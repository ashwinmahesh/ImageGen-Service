"""Durable single-worker image jobs. Inference never runs inside a SQL transaction."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from storage_paths import require_storage

log=logging.getLogger('image-queue')
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
            if version not in (0,1,2):
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
                migration.execute('PRAGMA user_version=2')

    @staticmethod
    def record(row):
        if row is None:
            raise FileNotFoundError('Unknown job ID')
        item=dict(row)
        item['request']=json.loads(item.pop('payload'))
        item['prompt_expansion']=json.loads(item['prompt_expansion']) if item.get('prompt_expansion') else None
        item['result']=json.loads(item['result']) if item['result'] else None
        item['generate_immediately']=bool(item.pop('immediate'))
        for key in ['adapter_sha','idempotency_key','fingerprint']:
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
            immediate=payload['generate_immediately']
            identifier=uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,status,payload,fingerprint,adapter_sha,idempotency_key,created,immediate,retry_of) VALUES(?,?,?,?,?,?,?,?,?)',
                (identifier,'queued',json.dumps(payload,allow_nan=False),request_fingerprint,adapter_sha,key,time.time(),int(immediate),retry_of))
            return self.record(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    def recover(self):
        # Called only after acquiring the process-wide worker lease.
        with self.connection(write=True) as db:
            return db.execute("UPDATE jobs SET status='queued',started=NULL,error='Interrupted by a previous shutdown; retrying' WHERE status='running'").rowcount

    def has_work(self,*,loras_enabled=True,llm_enabled=True):
        with self.connection() as db:
            paused=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
            condition=" AND immediate=1" if paused else ''
            row=db.execute("SELECT payload FROM jobs WHERE status='queued'"+condition+" ORDER BY immediate DESC,sequence LIMIT 1").fetchone()
            if not row: return False
            request=json.loads(row['payload'])
            return (loras_enabled or not request.get('lora')) and (llm_enabled or not request.get('expand_prompt'))

    def claim(self,*,loras_enabled=True,llm_enabled=True):
        with self.connection(write=True) as db:
            paused=json.loads(db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])
            condition=" AND immediate=1" if paused else ''
            row=db.execute("SELECT * FROM jobs WHERE status='queued'"+condition+" ORDER BY immediate DESC,sequence LIMIT 1").fetchone()
            if not row:
                return None
            if json.loads(row['payload']).get('lora') and not loras_enabled:
                return None  # Preserve the head job until adapters are enabled or it is cancelled.
            if json.loads(row['payload']).get('expand_prompt') and not llm_enabled:
                return None  # Preserve jobs requiring an optional startup capability.
            db.execute("UPDATE jobs SET status='running',started=?,attempts=attempts+1,error=NULL WHERE id=?",(time.time(),row['id']))
            item=self.record(db.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone())
            item['_adapter_sha']=row['adapter_sha']
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
            db.execute("UPDATE jobs SET status='cancelled',finished=? WHERE id=?",(time.time(),identifier))
        return self.get(identifier)

    def clear(self):
        with self.connection(write=True) as db:
            return db.execute("UPDATE jobs SET status='cancelled',finished=? WHERE status='queued'",(time.time(),)).rowcount

    def requeue(self,identifier):
        with self.connection(write=True) as db:
            db.execute("UPDATE jobs SET status='queued',started=NULL WHERE id=? AND status='running'",(identifier,))

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
            head=db.execute("SELECT id,payload FROM jobs WHERE status='queued' ORDER BY immediate DESC,sequence LIMIT 1").fetchone()
            counts['next_job_id']=head['id'] if head else None
            counts['next_requires_llm']=bool(head and json.loads(head['payload']).get('expand_prompt'))
            counts['next_requires_lora']=bool(head and json.loads(head['payload']).get('lora'))
            return counts

    def listing(self,limit=24,offset=0,status=None):
        where=" WHERE status IN ('queued','running')" if status=='pending' else (' WHERE status=?' if status else '')
        values=[status] if status and status!='pending' else []
        with self.connection() as db:
            total=db.execute('SELECT COUNT(*) FROM jobs'+where,values).fetchone()[0]
            rows=db.execute("SELECT * FROM jobs"+where+" ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, CASE WHEN status='queued' THEN immediate END DESC, CASE WHEN status IN ('queued','running') THEN sequence END ASC, sequence DESC LIMIT ? OFFSET ?",[*values,limit,offset]).fetchall()
            return [self.record(row) for row in rows],total


class QueueWorker:
    def __init__(self,store,execute,*,loras_enabled=lambda:True,llm_enabled=lambda:True,busy=None,gpu_lock_file=None):
        self.store,self.execute=store,execute
        self.loras_enabled=loras_enabled
        self.llm_enabled=llm_enabled
        self.busy=busy or threading.Lock()
        self.stopping=threading.Event()
        self.wake=threading.Event()
        self.future=None
        self.lease=None
        self.error=None
        self.gpu_lock_file=Path(gpu_lock_file) if gpu_lock_file else None
        self.stage='idle'

    def acquire(self, *, prepare=None):
        self.store.path.parent.mkdir(parents=True,exist_ok=True)
        self.lease=(self.store.path.parent/'worker.lock').open('a+')
        warned=False
        while True:
            try:
                fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.lease.seek(0)
                try:
                    owner=int(self.lease.read().strip())
                    os.kill(owner,0)
                except (ValueError,ProcessLookupError):
                    if not warned:
                        log.warning('Waiting for the previous native generation to release this queue')
                        warned=True
                    if self.stopping.wait(.5):
                        raise RuntimeError('Startup interrupted while waiting for previous inference')
                else:
                    self.lease.close(); self.lease=None
                    raise RuntimeError('This model queue already has a live worker; use its existing server') from None
        self.lease.seek(0); self.lease.truncate(); self.lease.write(str(os.getpid())); self.lease.flush()
        try:
            if prepare:
                prepare()
            self.store.recover()
        except BaseException:
            self.release()
            raise

    def start(self,executor):
        self.future=executor.submit(self.run)

    def run(self):
        try:
            while not self.stopping.is_set():
                self.wake.clear()
                if not self.store.has_work(loras_enabled=self.loras_enabled(),llm_enabled=self.llm_enabled()):
                    self.stage='idle'
                    self.wake.wait(.5)
                    continue
                self.stage='waiting_for_accelerator'
                gpu=self.acquire_accelerator()
                if self.stopping.is_set():
                    if gpu: gpu.close()
                    break
                job=None
                try:
                    # Claim only when the GPU is available. Waiting jobs stay removable.
                    job=self.store.claim(loras_enabled=self.loras_enabled(),llm_enabled=self.llm_enabled())
                    if job is None:
                        if gpu: gpu.close()
                        continue
                    try:
                        self.stage='generating'
                        job['_inherited_fds']=tuple(handle.fileno() for handle in [self.lease,gpu] if handle)
                        with self.busy:
                            result=self.execute(job)
                    finally:
                        if gpu: gpu.close()
                        self.stage='idle'
                except Exception as error:
                    if job is None:
                        if gpu: gpu.close()
                        raise
                    log.exception('Job %s failed',job['id'])
                    self.store.finish(job['id'],error=str(error)[:2000] or type(error).__name__,error_status=getattr(error,'http_status',500))
                else:
                    # If this commit fails, leave it running for recovery from its saved artifact.
                    self.store.finish(job['id'],result=result)
        except Exception:
            self.error='Queue worker stopped; inspect storage and GPU-lock configuration, then restart. Accepted jobs remain persisted.'
            log.exception('Queue worker stopped')

    def acquire_accelerator(self):
        if self.gpu_lock_file is None:
            return None
        require_storage(self.gpu_lock_file)
        self.gpu_lock_file.parent.mkdir(parents=True,exist_ok=True)
        handle=self.gpu_lock_file.open('a+')
        while not self.stopping.is_set():
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                self.stopping.wait(.25)
        handle.close()
        return None

    def stop(self):
        self.stopping.set(); self.wake.set()
        if self.future:
            self.future.result()  # Finish the current image; leave unclaimed jobs persisted.
        self.release()

    def release(self):
        if self.lease:
            self.lease.close(); self.lease=None
