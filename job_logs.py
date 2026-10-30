"""Best-effort job runtime capture and bounded reads of attributable logs."""
from contextlib import closing,contextmanager,redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import traceback
import uuid
from model_profiles import PROFILES
from queue_storage import queue_database
from storage_paths import data_directory

IDENTIFIER=re.compile(r'[0-9a-f]{32}')
ANSI=re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))')


def runtime_folder(project,identifier):
    if not IDENTIFIER.fullmatch(identifier):raise FileNotFoundError('Unknown job ID')
    project=Path(project).resolve();folder=project/'state'/'job-logs'/identifier
    if not folder.resolve().is_relative_to(project):raise RuntimeError('Job logs must stay inside the project')
    return folder


def candidate_identifier(identifier,index,count):
    """Image ID, and native log name, of one option in a batch."""
    return identifier if count==1 else uuid.uuid5(uuid.UUID(identifier),f'image-{index+1}').hex


def event(message):
    try:print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}',flush=True)
    except (OSError,ValueError):pass


class SafeOutput:
    """A full/missing log disk must not turn a rendering success into failure."""
    def __init__(self,file,original):self.file,self.original=file,original
    def write(self,text):
        for stream in (self.file,self.original):
            try:
                if stream is not None:stream.write(text);stream.flush()
            except (OSError,ValueError):pass
        return len(text)
    def flush(self):
        for stream in (self.file,self.original):
            try:
                if stream is not None:stream.flush()
            except (OSError,ValueError):pass
    def __getattr__(self,name):return getattr(self.original,name)


def capture_file(project,job):
    folder=runtime_folder(project,job['id']);folder.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(folder/f"attempt-{int(job['attempts'])}.log",os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor);raise OSError('Job log is not a regular file')
    return os.fdopen(descriptor,'a',encoding='utf-8',buffering=1)


@contextmanager
def capture_runtime(project,job):
    """Only used inside the serial model child; never redirect HTTP threads."""
    stream=None
    try:stream=capture_file(project,job)
    except (OSError,RuntimeError,ValueError):
        logging.getLogger(__name__).warning('Job log capture unavailable; inference continues')
    if stream is None:
        yield
        return
    output=SafeOutput(stream,sys.stdout);errors=SafeOutput(stream,sys.stderr)
    handler=logging.StreamHandler(SafeOutput(stream,None))
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger=logging.getLogger();logger.addHandler(handler)
    try:
        with redirect_stdout(output),redirect_stderr(errors):
            event(f"Starting job {job['id']} · model {job['request']['model']} · attempt {job['attempts']}")
            try:yield
            except BaseException:
                event('Generation attempt failed or was interrupted');traceback.print_exc();raise
            else:event('Generation attempt completed')
    finally:
        logger.removeHandler(handler)
        try:stream.close()
        except OSError:pass


def record_event(project,job,message,*,error=False):
    """Scheduler diagnostics without redirecting any shared HTTP streams."""
    try:
        with capture_file(project,job) as stream:
            stream.write(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}\n')
            if error:traceback.print_exc(file=stream)
    except (OSError,RuntimeError,ValueError):pass


def job_record(project,identifier):
    if not IDENTIFIER.fullmatch(identifier):raise FileNotFoundError('Unknown job ID')
    path=queue_database(project)
    if not path.is_file():raise FileNotFoundError('Unknown job ID')
    # Read-only: this endpoint never creates state or recovers/updates jobs.
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2)) as db:
        row=db.execute('SELECT payload,status,attempts,error FROM jobs WHERE id=?',(identifier,)).fetchone()
    if row is None:raise FileNotFoundError('Unknown job ID')
    return {'request':json.loads(row[0]),'status':row[1],'attempts':row[2],'error':row[3]}


def read_tail(path,budget):
    # Accept only a regular, directly named file; no arbitrary file or pipe reads.
    if path.is_symlink() or path.resolve().parent!=path.parent.resolve():return None
    try:descriptor=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    except FileNotFoundError:return None
    try:
        metadata=os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):return None
        with os.fdopen(descriptor,'rb',closefd=False) as source:
            source.seek(max(0,metadata.st_size-budget));data=source.read(budget)
        text=ANSI.sub('',data.decode('utf-8',errors='replace')).replace('\r','\n')
        text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',text)
        return {'text':text,'size_bytes':metadata.st_size,'bytes_read':len(data),'truncated':metadata.st_size>budget,'modified':metadata.st_mtime}
    finally:os.close(descriptor)


def read_job_logs(project,identifier,*,model=None,max_bytes=65536):
    if not 1024<=max_bytes<=262144:raise ValueError('max_bytes must be between 1024 and 262144')
    record=job_record(project,identifier);selected=record['request'].get('model')
    if model is not None and model!=selected:raise FileNotFoundError('Unknown job ID for this model')
    folder=runtime_folder(project,identifier);candidates=[];warnings=[]
    attempts=record['attempts']
    for attempt in range(attempts,max(0,attempts-10),-1):
        candidates.append((folder/f'attempt-{attempt}.log',{'kind':'runtime','label':f'Runtime · attempt {attempt}','attempt':attempt,'candidate_index':None}))
    if selected in PROFILES and (PROFILES[selected]['backend']=='sd-cpp' or PROFILES[selected].get('cli_module')):
        try:
            logs=data_directory(project)/selected/'logs'
            count=record['request'].get('n',1);count=count if type(count) is int and 1<=count<=5 else 1
            for index in range(count):
                candidate=candidate_identifier(identifier,index,count)
                candidates.append((logs/(candidate+'.log'),{'kind':'native','label':f'Image runtime · option {index+1}','attempt':None,'candidate_index':index+1}))
            warnings.append('Native output logs contain the latest file contents; they are not separated by attempt.')
        except (OSError,RuntimeError,ValueError):warnings.append('Native logs are unavailable while model storage is inaccessible.')
    # Allocate one shared byte budget across the bounded source list.
    sources=[];budget=max_bytes//max(1,len(candidates))
    for path,info in candidates:
        try:tail=read_tail(path,budget)
        except OSError:
            warnings.append('A job log could not be read; check storage access.');continue
        if tail is not None:sources.append({**info,'file_path':str(path),**tail})
    return {'job_id':identifier,'model':selected,'status':record['status'],'error':record['error'],
            'available':bool(sources),'sources':sources,'max_bytes':max_bytes,
            'note':' '.join(warnings) if sources else 'No job-specific logs are available yet. Older shared runtime logs cannot be safely attributed to this job.'}
