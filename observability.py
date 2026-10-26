"""Independent project-local resource history; no queue schema or inference imports."""
from contextlib import contextmanager,closing
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from resource_probe import FIELDS,ResourceProbe

WINDOWS={'5m':300,'30m':1800,'1h':3600,'6h':21600,'24h':86400,'3d':259200,'7d':604800}
JOB_WINDOWS={'full':None,'1m':60,'5m':300,'15m':900}
RETENTION_SECONDS=8*86400
log=logging.getLogger('image-observability')


def database_path(project):
    project=Path(project).resolve()
    path=project/'state/observability/metrics.sqlite3'
    if not path.resolve().is_relative_to(project):
        raise RuntimeError('Observability state must remain inside the project; remove the external state symlink')
    return path


class MetricStore:
    def __init__(self,project):
        self.project=Path(project).resolve();self.path=database_path(project)

    @contextmanager
    def connection(self):
        if database_path(self.project)!=self.path:raise RuntimeError('Observability path changed')
        db=sqlite3.connect(f'{self.path.as_uri()}?mode=rw',uri=True,timeout=.25)
        db.row_factory=sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=NORMAL')
            yield db
            db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()

    def initialize(self):
        database_path(self.project)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path,timeout=.25)) as db:
            if db.execute('PRAGMA user_version').fetchone()[0] not in (0,1):
                raise RuntimeError('Unsupported observability database version')
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,model TEXT NOT NULL,pid INTEGER NOT NULL,started REAL NOT NULL,ended REAL);
                CREATE TABLE IF NOT EXISTS runs(session TEXT NOT NULL,job_id TEXT NOT NULL,attempt INTEGER NOT NULL,model TEXT NOT NULL,
                    started REAL NOT NULL,finished REAL,status TEXT NOT NULL,summary TEXT,PRIMARY KEY(session,job_id,attempt));
                CREATE INDEX IF NOT EXISTS run_time ON runs(started);
                CREATE INDEX IF NOT EXISTS run_job ON runs(model,job_id);
            ''')
            db.execute('CREATE TABLE IF NOT EXISTS samples(t REAL NOT NULL,session TEXT NOT NULL,model TEXT NOT NULL,job_id TEXT,attempt INTEGER,stage TEXT NOT NULL,'+
                       ','.join(name+' REAL' for name in FIELDS)+',process_count INTEGER,process_tree_partial INTEGER NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS sample_time ON samples(t)')
            db.execute('CREATE INDEX IF NOT EXISTS sample_job ON samples(job_id,model,t)')
            db.execute('PRAGMA user_version=1')
            db.commit()

    def session(self,identifier,model,pid,started):
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO sessions VALUES(?,?,?,?,NULL)',(identifier,model,pid,started))

    def save(self,sample):
        names=('t','session','model','job_id','attempt','stage',*FIELDS,'process_count','process_tree_partial')
        with self.connection() as db:
            db.execute('INSERT INTO samples('+','.join(names)+') VALUES('+','.join('?' for _ in names)+')',
                       [sample.get(name) for name in names])

    def begin(self,session,model,job,now):
        self.recover(model,job_id=job['id'])
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?,?,NULL)',
                       (session,job['id'],job['attempts'],model,now,None,'running'))

    def recover(self,model,*,job_id=None):
        where=' WHERE status=\'running\' AND model=?';args=[model]
        if job_id is not None:where+=' AND job_id=?';args.append(job_id)
        with self.connection() as db:rows=db.execute('SELECT * FROM runs'+where,args).fetchall()
        for row in rows:
            summary=self.summary(model=model,job_id=row['job_id'],session=row['session'],attempt=row['attempt'])
            with self.connection() as db:
                db.execute("UPDATE runs SET status='interrupted',finished=?,summary=? WHERE session=? AND job_id=? AND attempt=? AND status='running'",
                           (summary['last_sample'] or row['started'],json.dumps(summary),row['session'],row['job_id'],row['attempt']))

    def finish(self,session,model,job,status,now):
        summary=self.summary(job_id=job['id'],model=model,session=session,attempt=job['attempts'])
        with self.connection() as db:
            db.execute('UPDATE runs SET finished=?,status=?,summary=? WHERE session=? AND job_id=? AND attempt=?',
                       (now,status,json.dumps(summary),session,job['id'],job['attempts']))

    @staticmethod
    def where(*,start=None,end=None,model=None,job_id=None,session=None,attempt=None):
        parts=[];args=[]
        for key,value,op in [('t',start,'>='),('t',end,'<='),('model',model,'='),('job_id',job_id,'='),('session',session,'='),('attempt',attempt,'=')]:
            if value is not None:parts.append(key+op+'?');args.append(value)
        return (' WHERE '+' AND '.join(parts) if parts else ''),args

    def summary(self,**filters):
        where,args=self.where(**filters)
        expressions=['COUNT(*) AS count','MIN(t) AS first','MAX(t) AS last']
        for name in FIELDS:expressions.extend([f'AVG({name}) AS {name}_mean',f'MAX({name}) AS {name}_peak',f'COUNT({name}) AS {name}_count'])
        with self.connection() as db:row=db.execute('SELECT '+','.join(expressions)+' FROM samples'+where,args).fetchone()
        return {'sample_count':row['count'],'first_sample':row['first'],'last_sample':row['last'],
                'metrics':{name:{'mean':row[name+'_mean'],'peak':row[name+'_peak'],'count':row[name+'_count']} for name in FIELDS}}

    def series(self,start,end,*,model=None,job_id=None,max_points=600):
        width=max(.1,(end-start)/max_points)
        where,args=self.where(start=start,end=end,model=model,job_id=job_id)
        expressions=[f'MIN(CAST((t-?)/? AS INTEGER),{int(max_points)-1}) AS bucket','MIN(t) AS first','MAX(t) AS last','COUNT(*) AS count',
                     'GROUP_CONCAT(DISTINCT model) AS models','GROUP_CONCAT(DISTINCT session) AS sessions','GROUP_CONCAT(DISTINCT stage) AS stages']
        for name in FIELDS:expressions.extend([f'AVG({name}) AS {name}',f'MAX({name}) AS {name}_peak'])
        with self.connection() as db:
            rows=db.execute('SELECT '+','.join(expressions)+' FROM samples'+where+' GROUP BY bucket ORDER BY bucket',
                            [start,width,*args]).fetchall()
        points=[{'timestamp':(row['first']+row['last'])/2,'first_sample':row['first'],'last_sample':row['last'],'count':row['count'],
                 'models':row['models'].split(','),'sessions':row['sessions'].split(','),'stages':row['stages'].split(','),
                 'values':{name:row[name] for name in FIELDS},'peaks':{name:row[name+'_peak'] for name in FIELDS}} for row in rows]
        return {'start':start,'end':end,'bucket_seconds':width,'points':points,'summary':self.summary(start=start,end=end,model=model,job_id=job_id)}

    def jobs(self,*,model=None,limit=20,offset=0):
        where,args=(' WHERE model=?',[model]) if model else ('',[])
        with self.connection() as db:
            total=db.execute('SELECT COUNT(*) FROM runs'+where,args).fetchone()[0]
            rows=db.execute('SELECT * FROM runs'+where+' ORDER BY started DESC,job_id LIMIT ? OFFSET ?',[*args,limit,offset]).fetchall()
        data=[]
        for row in rows:
            item=dict(row)
            item['summary']=json.loads(item['summary']) if item['summary'] else self.summary(model=item['model'],job_id=item['job_id'],session=item['session'],attempt=item['attempt'])
            data.append(item)
        return {'data':data,'total':total,'limit':limit,'offset':offset,'has_more':offset+len(data)<total}

    def job(self,identifier,*,model=None,window='full',max_points=600,now=None):
        where,args=self.where(job_id=identifier,model=model)
        with self.connection() as db:rows=db.execute('SELECT * FROM runs'+where+' ORDER BY started',args).fetchall()
        if not rows:raise FileNotFoundError('No resource samples recorded for this job yet')
        now=time.time() if now is None else now
        runs=[{**dict(row),'summary':json.loads(row['summary']) if row['summary'] else self.summary(job_id=identifier,model=row['model'],session=row['session'],attempt=row['attempt'])} for row in rows]
        end=max(row['finished'] or now for row in runs)
        start=min(row['started'] for row in runs)
        if JOB_WINDOWS[window] is not None:start=max(start,end-JOB_WINDOWS[window])
        result=self.series(start,max(start+.1,end),model=model,job_id=identifier,max_points=max_points)
        result.update(job_id=identifier,runs=runs)
        return result

    def prune(self,now):
        with self.connection() as db:db.execute('DELETE FROM samples WHERE t<?',(now-RETENTION_SECONDS,))

    def end_session(self,session,now):
        with self.connection() as db:db.execute('UPDATE sessions SET ended=? WHERE id=?',(now,session))


class ResourceMonitor:
    def __init__(self,project,model,*,stage=lambda:'idle',probe=None,clock=time.time):
        self.store=MetricStore(project);self.model=model;self.stage=stage;self.probe=probe or ResourceProbe(project)
        self.clock=clock;self.session=uuid.uuid4().hex;self.started=clock()
        self.lock=threading.RLock();self.stopping=threading.Event();self.wake=threading.Event();self.thread=None;self.active=None
        self.ready=False;self.last_error=None;self.last_sample=None;self.next_prune=0
        self.last_tick=0

    def ensure_store(self):
        if not self.ready:
            self.store.initialize();self.store.session(self.session,self.model,self.probe.pid,self.started);self.ready=True

    def sample(self):
        with self.lock:
            try:
                self.ensure_store();now=self.clock();job=self.active
                self.store.save({'t':now,'session':self.session,'model':self.model,'job_id':job['id'] if job else None,
                    'attempt':job['attempts'] if job else None,'stage':self.stage(),**self.probe.collect()})
                self.last_sample=now;self.last_error=None
                if now>=self.next_prune:self.store.prune(now);self.next_prune=now+300
            except Exception:
                if self.last_error is None:log.warning('Resource sampling unavailable; inference continues',exc_info=True)
                self.last_error='Resource history unavailable; check project state directory permissions and disk space.'
            finally:self.last_tick=time.monotonic()

    def start(self):
        self.thread=threading.Thread(target=self.run,name='resource-sampler',daemon=True);self.thread.start()

    def run(self):
        while not self.stopping.is_set():
            self.wake.clear()
            interval=1 if self.active else 5
            remaining=self.last_tick+interval-time.monotonic()
            if remaining<=0:self.sample()
            else:self.wake.wait(remaining)

    def recover_jobs(self):
        # Called only after this model's exclusive worker lease was acquired.
        with self.lock:
            try:
                from model_profiles import PROFILES
                self.ensure_store()
                for identifier in PROFILES:self.store.recover(identifier)
            except Exception:self.last_error='Could not reconcile old resource attempts; inference continues.'

    def start_job(self,job):
        with self.lock:
            self.model=job.get('request',{}).get('model',self.model)
            self.active={'id':job['id'],'attempts':job['attempts']}
            try:self.ensure_store();self.store.begin(self.session,self.model,job,self.clock())
            except Exception:self.last_error='Could not record job resource history; inference continues.'
            self.sample()
            self.wake.set()

    def finish_job(self,job,status):
        with self.lock:
            try:
                self.sample();self.store.finish(self.session,self.model,job,status,self.clock())
            except Exception:self.last_error='Could not finish job resource history; inference continues.'
            finally:self.active=None;self.wake.set()

    def stop(self):
        self.stopping.set();self.wake.set()
        if self.thread:self.thread.join(timeout=3)
        try:
            if self.ready:self.store.end_session(self.session,self.clock())
        except Exception:pass

    def status(self):
        return {'recording':self.thread is not None and self.thread.is_alive(),'session':self.session,'model':self.model,
                'last_sample':self.last_sample,'error':self.last_error,'sources':dict(self.probe.status),
                'idle_interval_seconds':5,'job_interval_seconds':1,'retention_days':8}
