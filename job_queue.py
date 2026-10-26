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
        for key in ['adapter_sha','idempotency_key','fingerprint','prepared']:
            item.pop(key,None)
        return item

    def get(self,identifier):
        with self.connection() as db:
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

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
            running=[row[0] for row in db.execute("SELECT id FROM jobs WHERE status='running' ORDER BY started,sequence")]
            waiting=[row['id'] for row in self._waiting(db)]
            changed=db.execute("UPDATE jobs SET status='queued',started=NULL,error='Interrupted by a previous shutdown; retrying' WHERE status='running'").rowcount
            self._set_order(db,running+waiting)
            return changed

    def has_work(self,*,loras_enabled=True,llm_enabled=True):
        with self.connection() as db:
            paused=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
            condition=" AND immediate=1" if paused else ''
            row=db.execute("SELECT payload FROM jobs WHERE status='queued'"+condition+" ORDER BY queue_position,sequence LIMIT 1").fetchone()
            if not row: return False
            request=json.loads(row['payload'])
            return (loras_enabled or not request.get('lora')) and (llm_enabled or not request.get('expand_prompt'))

    def claim(self,*,loras_enabled=True,llm_enabled=True):
        with self.connection(write=True) as db:
            paused=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
            condition=" AND immediate=1" if paused else ''
            row=db.execute("SELECT * FROM jobs WHERE status='queued'"+condition+" ORDER BY queue_position,sequence LIMIT 1").fetchone()
            if not row:
                return None
            if json.loads(row['payload']).get('lora') and not loras_enabled:
                return None  # Preserve the head job until adapters are enabled or it is cancelled.
            if json.loads(row['payload']).get('expand_prompt') and not llm_enabled:
                return None  # Preserve jobs requiring an optional startup capability.
            db.execute("UPDATE jobs SET status='running',queue_position=NULL,started=?,attempts=attempts+1,error=NULL WHERE id=?",(time.time(),row['id']))
            self._set_order(db,[item['id'] for item in self._waiting(db)])
            item=self.record(db.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone())
            item['_adapter_sha']=row['adapter_sha']
            item['_prepared']=bool(row['prepared'])
            return item

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

    def finish(self,identifier,*,result=None,error=None,error_status=None):
        with self.connection(write=True) as db:
            db.execute("UPDATE jobs SET status=?,result=COALESCE(?,result),error=?,error_status=?,finished=? WHERE id=? AND status='running'",
                ('failed' if error else 'succeeded',json.dumps(result,allow_nan=False) if result else None,error,error_status,time.time(),identifier))

    def cancel(self,identifier):
        with self.connection(write=True) as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row:
                raise FileNotFoundError('Unknown job ID')
            if row['status']!='queued':
                raise QueueConflict('Only waiting jobs can be cancelled; running inference is not interrupted')
            db.execute("UPDATE jobs SET status='cancelled',queue_position=NULL,finished=? WHERE id=?",(time.time(),identifier))
            self._set_order(db,[item['id'] for item in self._waiting(db)])
        return self.get(identifier)

    def clear(self):
        with self.connection(write=True) as db:
            return db.execute("UPDATE jobs SET status='cancelled',queue_position=NULL,finished=? WHERE status='queued'",(time.time(),)).rowcount

    def requeue(self,identifier):
        with self.connection(write=True) as db:
            waiting=[row['id'] for row in self._waiting(db)]
            changed=db.execute("UPDATE jobs SET status='queued',started=NULL WHERE id=? AND status='running'",(identifier,)).rowcount
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
            counts['next_requires_llm']=bool(head and json.loads(head['payload']).get('expand_prompt'))
            counts['next_requires_lora']=bool(head and json.loads(head['payload']).get('lora'))
            return counts

    def listing(self,limit=24,offset=0,status=None):
        where=" WHERE status IN ('queued','running')" if status=='pending' else (' WHERE status=?' if status else '')
        values=[status] if status and status!='pending' else []
        with self.connection() as db:
            total=db.execute('SELECT COUNT(*) FROM jobs'+where,values).fetchone()[0]
            rows=db.execute("SELECT * FROM jobs"+where+" ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, CASE WHEN status='queued' THEN queue_position END ASC, CASE WHEN status IN ('queued','running') THEN sequence END ASC, sequence DESC LIMIT ? OFFSET ?",[*values,limit,offset]).fetchall()
            return [self.record(row) for row in rows],total
