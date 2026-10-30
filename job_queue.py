"""Durable queue storage. Inference never runs inside a SQL transaction."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
from storage_paths import require_storage

TERMINAL={'succeeded','failed','cancelled'}
# Settings-table key for the Keep weights loaded switch, shared by both services.
KEEP_WEIGHTS_SETTING='keep_weights_loaded'
CANCELLED_WHILE_RUNNING='Cancelled while running; any finished images in a batch are kept'


class QueueConflict(Exception):
    pass


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class JobStore:
    def __init__(self,path):
        self.path=Path(path)

    @contextmanager
    def connection(self,write=False):
        try:
            require_storage(self.path)
        except RuntimeError as error:
            raise sqlite3.OperationalError('Queue storage volume is unavailable') from error
        connection=sqlite3.connect(self.path,timeout=10)
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA synchronous=FULL')
        try:
            if write:
                connection.execute('BEGIN IMMEDIATE')
            yield connection
            if write:
                connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        require_storage(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,1,2,3,4):
                raise RuntimeError('Unsupported queue database version; use a compatible server')
            # Short rollback-journal transactions; all queue files stay project-local.
            db.execute('PRAGMA journal_mode=DELETE')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
                    payload TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    adapter_sha TEXT,
                    idempotency_key TEXT UNIQUE,
                    created REAL NOT NULL,
                    started REAL,
                    finished REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    error_status INTEGER,
                    immediate INTEGER NOT NULL DEFAULT 0,
                    retry_of TEXT
                );
                CREATE INDEX IF NOT EXISTS job_order ON jobs(status,immediate DESC,sequence);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES('paused','false');

            ''')
            with self.connection(write=True) as migration:
                columns={row[1] for row in migration.execute('PRAGMA table_info(jobs)')}
                if 'prompt_expansion' not in columns:
                    migration.execute('ALTER TABLE jobs ADD COLUMN prompt_expansion TEXT')
                if 'prepared' not in columns:
                    migration.execute('ALTER TABLE jobs ADD COLUMN prepared INTEGER NOT NULL DEFAULT 0')
                    if version in (1,2):
                        # Older admissions had already resolved defaults, seeds and adapter checksums.
                        migration.execute('UPDATE jobs SET prepared=1')
                if 'queue_position' not in columns:
                    migration.execute('ALTER TABLE jobs ADD COLUMN queue_position INTEGER')
                    identifiers=[row[0] for row in migration.execute(
                        "SELECT id FROM jobs WHERE status='queued' ORDER BY immediate DESC,sequence")]
                    self._set_order(migration,identifiers)
                if 'progress' not in columns:
                    # Nullable live progress; older readers ignore it, so the schema version is unchanged.
                    migration.execute('ALTER TABLE jobs ADD COLUMN progress TEXT')
                if 'cancel_requested' not in columns:
                    # Set by either service; only the generator stops the runtime and records the cancellation.
                    migration.execute('ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0')
                migration.execute('CREATE INDEX IF NOT EXISTS queue_position_order ON jobs(status,queue_position,sequence)')
                migration.execute('PRAGMA user_version=4')

    @staticmethod
    def _waiting(db):
        return db.execute("SELECT id,immediate FROM jobs WHERE status='queued' ORDER BY queue_position,sequence").fetchall()

    @staticmethod
    def _set_order(db,identifiers):
        # Dense one-based positions are scoped to waiting work, independent of history.
        db.executemany('UPDATE jobs SET queue_position=? WHERE id=? AND queue_position IS NOT ?',
                       [(position,identifier,position) for position,identifier in enumerate(identifiers,1)])

    def _insert_order(self,db,identifier,immediate):
        rows=[row for row in self._waiting(db) if row['id']!=identifier]
        index=next((index for index,row in enumerate(rows) if not row['immediate']),len(rows)) if immediate else len(rows)
        identifiers=[row['id'] for row in rows]
        identifiers.insert(index,identifier)
        self._set_order(db,identifiers)

    def move(self,identifier,position):
        """Move waiting work, including recovered attempts, in one transaction."""
        with self.connection(write=True) as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None:
                raise FileNotFoundError('Unknown job ID')
            if row['status']!='queued':
                raise QueueConflict('Only waiting jobs can be moved; active work is not interrupted')
            identifiers=[row['id'] for row in self._waiting(db)]
            if type(position) is not int or not 1<=position<=len(identifiers):
                raise QueueConflict(f'position must be between 1 and {len(identifiers)} waiting jobs; refresh the queue and retry')
            identifiers.remove(identifier)
            identifiers.insert(position-1,identifier)
            self._set_order(db,identifiers)
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    def run_now(self,identifier):
        """Prioritize this identity without changing frozen request/checkpoints."""
        with self.connection(write=True) as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None:
                raise FileNotFoundError('Unknown job ID')
            if row['status']!='queued':
                raise QueueConflict('Only waiting jobs can run now; active work is not interrupted')
            identifiers=[identifier]+[row['id'] for row in self._waiting(db) if row['id']!=identifier]
            db.execute('UPDATE jobs SET immediate=1 WHERE id=?',(identifier,))
            self._set_order(db,identifiers)
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    @staticmethod
    def record(row):
        if row is None:
            raise FileNotFoundError('Unknown job ID')
        item=dict(row)
        item['request']=json.loads(item.pop('payload'))
        item['prompt_expansion']=json.loads(item['prompt_expansion']) if item.get('prompt_expansion') else None
        item['result']=json.loads(item['result']) if item['result'] else None
        item['generate_immediately']=bool(item.pop('immediate'))
        item['progress']=JobStore.progress(item,item.get('progress'))
        item['cancel_requested']=bool(item.get('cancel_requested'))
        for key in ['adapter_sha','idempotency_key','fingerprint','prepared']:
            item.pop(key,None)
        return item

    @staticmethod
    def progress(item,saved):
        if item['status']=='succeeded':
            count=item['request'].get('n',1);count=count if type(count) is int and 1<=count<=5 else 1
            return {'percent':100,'phase':'done','image':count,'total_images':count,'updated':item.get('finished')}
        if item['status']!='running':return None
        try:value=json.loads(saved) if saved else None
        except ValueError:value=None
        return value if isinstance(value,dict) else {'percent':0,'phase':'starting','total_images':item['request'].get('n',1)}

    def get(self,identifier):
        with self.connection() as db:
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    @contextmanager
    def guard_artifact_deletion(self, path, owner_job_id=None):
        """Keep pending-reference checks and filesystem deletion ordered with CRUD."""
        path = Path(path).resolve()
        with self.connection(write=True) as db:
            for row in db.execute("SELECT id,payload FROM jobs WHERE status IN ('queued','running')"):
                if row['id'] == owner_job_id:
                    raise QueueConflict('This image belongs to an unfinished job. Wait for it to finish before deleting it.')
                payload = json.loads(row['payload'])
                for field in ('reference_image', 'mask_image'):
                    value = payload.get(field)
                    if isinstance(value, str) and value.strip() and Path(value).expanduser().resolve() == path:
                        raise QueueConflict('A queued or running job uses this image. Finish or cancel that job before deleting it.')
            yield

    def existing(self,key,request_fingerprint):
        if key is None:
            return None
        with self.connection() as db:
            row=db.execute('SELECT * FROM jobs WHERE idempotency_key=?',(key,)).fetchone()
            if row and row['fingerprint']!=request_fingerprint:
                raise QueueConflict('Idempotency-Key was already used for different settings')
            return self.record(row) if row else None

    def enqueue(self,payload,request_fingerprint,*,key=None,adapter_sha=None,retry_of=None):
        with self.connection(write=True) as db:
            if key:
                row=db.execute('SELECT * FROM jobs WHERE idempotency_key=?',(key,)).fetchone()
                if row:
                    if row['fingerprint']!=request_fingerprint:
                        raise QueueConflict('Idempotency-Key was already used for different settings')
                    return self.record(row)
            immediate=payload.get('generate_immediately',False)
            identifier=uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,status,payload,fingerprint,adapter_sha,idempotency_key,created,immediate,retry_of) VALUES(?,?,?,?,?,?,?,?,?)',
                (identifier,'queued',json.dumps(payload,allow_nan=False),request_fingerprint,adapter_sha,key,time.time(),int(immediate),retry_of))
            self._insert_order(db,identifier,immediate)
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    def update(self,identifier,payload):
        """Replace a never-started waiting request atomically with its priority/hash."""
        with self.connection(write=True) as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None:
                raise FileNotFoundError('Unknown job ID')
            if row['status']!='queued' or row['attempts']:
                raise QueueConflict('Only never-started waiting jobs can be edited; retry an interrupted job instead')
            db.execute('UPDATE jobs SET payload=?,fingerprint=?,immediate=?,adapter_sha=NULL,prepared=0 WHERE id=?',
                       (json.dumps(payload,allow_nan=False),fingerprint(payload),int(payload.get('generate_immediately',False)),identifier))
            if bool(row['immediate'])!=payload.get('generate_immediately',False):
                self._insert_order(db,identifier,payload.get('generate_immediately',False))
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    def prepare(self,identifier,payload,adapter_sha=None):
        """Freeze deferred defaults/seed before any model work, once per job."""
        with self.connection(write=True) as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None or row['status']!='running':
                raise QueueConflict('Cannot prepare an inactive job')
            if not row['prepared']:
                db.execute('UPDATE jobs SET payload=?,adapter_sha=?,prepared=1 WHERE id=?',
                           (json.dumps(payload,allow_nan=False),adapter_sha,identifier))
            record=self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
            record['_adapter_sha']=db.execute('SELECT adapter_sha FROM jobs WHERE id=?',(identifier,)).fetchone()[0]
            record['_prepared']=True
            return record

    def recover(self):
        # Called only after acquiring the process-wide worker lease.
        with self.connection(write=True) as db:
            waiting=[row['id'] for row in self._waiting(db)]
            # A cancel requested before shutdown is honoured instead of retrying the job.
            db.execute("UPDATE jobs SET status='cancelled',finished=?,error=?,cancel_requested=0 WHERE status='running' AND cancel_requested=1",(time.time(),CANCELLED_WHILE_RUNNING))
            running=[row[0] for row in db.execute("SELECT id FROM jobs WHERE status='running' ORDER BY started,sequence")]
            changed=db.execute("UPDATE jobs SET status='queued',started=NULL,progress=NULL,error='Interrupted by a previous shutdown; retrying' WHERE status='running'").rowcount
            self._set_order(db,running+waiting)
            return changed

    @staticmethod
    def _eligible(db,loras_enabled,llm_enabled,eligible_models):
        paused=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
        condition=" AND immediate=1" if paused else ''
        for row in db.execute("SELECT * FROM jobs WHERE status='queued'"+condition+" ORDER BY queue_position,sequence"):
            request=json.loads(row['payload'])
            if eligible_models is not None and request.get('model') not in eligible_models:continue
            enabled=(loras_enabled or not request.get('lora')) and (llm_enabled or not request.get('expand_prompt'))
            if enabled:return row
            if eligible_models is None:return None
        return None

    def has_work(self,*,loras_enabled=True,llm_enabled=True,eligible_models=None):
        with self.connection() as db:
            return self._eligible(db,loras_enabled,llm_enabled,eligible_models) is not None

    def claim(self,*,loras_enabled=True,llm_enabled=True,eligible_models=None):
        with self.connection(write=True) as db:
            row=self._eligible(db,loras_enabled,llm_enabled,eligible_models)
            if row is None:return None
            db.execute("UPDATE jobs SET status='running',queue_position=NULL,started=?,attempts=attempts+1,error=NULL,progress=NULL,cancel_requested=0 WHERE id=?",(time.time(),row['id']))
            self._set_order(db,[item['id'] for item in self._waiting(db)])
            item=self.record(db.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone())
            item['_adapter_sha']=row['adapter_sha'];item['_prepared']=bool(row['prepared'])
            return item

    def setting(self,key,default=None):
        with self.connection() as db:
            row=db.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_setting(self,key,value):
        with self.connection(write=True) as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,json.dumps(value)))

    def save_expansion(self,identifier,expansion):
        with self.connection(write=True) as db:
            changed=db.execute("UPDATE jobs SET prompt_expansion=? WHERE id=? AND status='running' AND prompt_expansion IS NULL",
                               (json.dumps(expansion,allow_nan=False),identifier)).rowcount
            if changed != 1:
                raise QueueConflict('Cannot replace a saved expansion or change an inactive job')
        return self.get(identifier)['prompt_expansion']

    def save_progress(self,identifier,result):
        # Only fully published PNGs/sidecars enter this checkpoint; no GPU work in SQL.
        with self.connection(write=True) as db:
            if db.execute("UPDATE jobs SET result=? WHERE id=? AND status='running'",
                          (json.dumps(result,allow_nan=False),identifier)).rowcount != 1:
                raise QueueConflict('Cannot update progress for an inactive job')

    def save_live_progress(self,identifier,attempt,progress):
        """Best-effort percentage for the running attempt; returns False instead of raising."""
        try:
            with self.connection(write=True) as db:
                return db.execute("UPDATE jobs SET progress=? WHERE id=? AND status='running' AND attempts=?",
                                  (json.dumps(progress,allow_nan=False),identifier,attempt)).rowcount==1
        except (sqlite3.Error,OSError,ValueError):
            return False

    def cancel_requested(self,identifier,attempt):
        with self.connection() as db:
            row=db.execute("SELECT cancel_requested FROM jobs WHERE id=? AND status='running' AND attempts=?",(identifier,attempt)).fetchone()
            return bool(row and row[0])

    def finish(self,identifier,*,result=None,error=None,error_status=None,cancelled=False):
        status='cancelled' if cancelled else 'failed' if error else 'succeeded'
        error=CANCELLED_WHILE_RUNNING if cancelled else error
        with self.connection(write=True) as db:
            # Cancelled jobs keep any saved batch checkpoint as their partial result.
            db.execute("UPDATE jobs SET status=?,result=COALESCE(?,result),error=?,error_status=?,finished=?,cancel_requested=0 WHERE id=? AND status='running'",
                (status,json.dumps(result,allow_nan=False) if result else None,error,error_status,time.time(),identifier))

    def cancel(self,identifier):
        with self.connection(write=True) as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row:
                raise FileNotFoundError('Unknown job ID')
            if row['status']=='running':
                # The generator stops the runtime within about a second, then records the job as cancelled.
                db.execute('UPDATE jobs SET cancel_requested=1 WHERE id=?',(identifier,))
            elif row['status']!='queued':
                raise QueueConflict('Only waiting or running jobs can be cancelled')
            else:
                db.execute("UPDATE jobs SET status='cancelled',queue_position=NULL,finished=? WHERE id=?",(time.time(),identifier))
                self._set_order(db,[item['id'] for item in self._waiting(db)])
        return self.get(identifier)

    def clear(self,model=None):
        with self.connection(write=True) as db:
            count=db.execute("UPDATE jobs SET status='cancelled',queue_position=NULL,finished=? WHERE status='queued'"+(" AND json_extract(payload,'$.model')=?" if model else ''),[time.time(),*([model] if model else [])]).rowcount
            self._set_order(db,[item['id'] for item in self._waiting(db)])
            return count

    def requeue(self,identifier):
        with self.connection(write=True) as db:
            waiting=[row['id'] for row in self._waiting(db)]
            changed=db.execute("UPDATE jobs SET status='queued',started=NULL,progress=NULL WHERE id=? AND status='running'",(identifier,)).rowcount
            if changed:
                self._set_order(db,[identifier]+waiting)

    def pause(self,value):
        with self.connection(write=True) as db:
            db.execute("UPDATE settings SET value=? WHERE key='paused'",(json.dumps(value),))
        return self.summary()

    def summary(self):
        with self.connection() as db:
            counts={key:0 for key in ['queued','running','succeeded','failed','cancelled']}
            counts.update({row[0]:row[1] for row in db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status')})
            counts['paused']=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
            counts['database_path']=str(self.path)
            condition=" AND immediate=1" if counts['paused'] else ''
            head=db.execute("SELECT id,payload FROM jobs WHERE status='queued'"+condition+" ORDER BY queue_position,sequence LIMIT 1").fetchone()
            counts['next_job_id']=head['id'] if head else None
            counts['next_model']=json.loads(head['payload']).get('model') if head else None
            counts['next_requires_llm']=bool(head and json.loads(head['payload']).get('expand_prompt'))
            counts['next_requires_lora']=bool(head and json.loads(head['payload']).get('lora'))
            return counts

    def listing(self,limit=24,offset=0,status=None,model=None):
        where=" WHERE status IN ('queued','running')" if status=='pending' else (' WHERE status=?' if status else '')
        values=[status] if status and status!='pending' else []
        if model:
            where+=(" AND " if where else " WHERE ")+"json_extract(payload,'$.model')=?";values.append(model)
        with self.connection() as db:
            total=db.execute('SELECT COUNT(*) FROM jobs'+where,values).fetchone()[0]
            rows=db.execute("SELECT * FROM jobs"+where+" ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, CASE WHEN status='queued' THEN queue_position END ASC, CASE WHEN status IN ('queued','running') THEN sequence END ASC, sequence DESC LIMIT ? OFFSET ?",[*values,limit,offset]).fetchall()
            return [self.record(row) for row in rows],total
