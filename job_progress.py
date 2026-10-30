"""Durable job percentages from runtime step counters. Failures never affect generation."""
import logging
import re
import threading
import time
from job_logs import candidate_identifier, runtime_folder
from model_profiles import PROFILES
from storage_paths import data_directory

log=logging.getLogger(__name__)
# tqdm (mflux): "5/8 [01:53<01:05, 21.87s/it]"; stable-diffusion.cpp: "7/8 -   1.73s/it - ETA 00:01".
STEP=re.compile(r'(\d+)/(\d+)\s*(?:\[[^\]\n]*?,\s*|-\s*)([0-9.]+)\s*(s/it|it/s)')
# Weight-loading bars share the same format; their descriptions precede the counter.
LOADING=re.compile(r'load|fetch|shard|download',re.I)
STAGES={'waiting_for_accelerator','loading_model','expanding_prompt','planning_variations'}
# Sampling fills 95% of each image; decoding and saving the PNG take the rest.
SAMPLING_SHARE=.95


def last_step(text):
    """Return (step, total, seconds_per_step) for the newest sampler counter, or None."""
    for line in reversed(re.split(r'[\r\n]',text)):
        match=STEP.search(line)
        if match is None or LOADING.search(line[:match.start()]):continue
        step,total,rate=int(match[1]),int(match[2]),float(match[3])
        if 0<total and step<=total and rate>0:
            return step,total,rate if match[4]=='s/it' else 1/rate
    return None


def read_tail(path,size=65536):
    try:
        with open(path,'rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-size))
            return stream.read().decode('utf-8',errors='replace')
    except OSError:
        return None


def image_count(job):
    count=job['request'].get('n',1)
    return count if type(count) is int and 1<=count<=5 else 1


def runtime_steps(project,job,index):
    """Read the counter the running image's runtime writes: a native CLI log or the captured attempt log."""
    model=job['request'].get('model');profile=PROFILES.get(model)
    if profile is None:return None
    if profile['backend']=='sd-cpp' or profile.get('cli_module'):
        path=data_directory(project)/model/'logs'/(candidate_identifier(job['id'],index,image_count(job))+'.log')
        try:
            # Native logs are reused by retries; ignore one written before this attempt started.
            if path.stat().st_mtime<(job.get('started') or 0)-1:return None
        except OSError:
            return None
        text=read_tail(path)
    else:
        text=read_tail(runtime_folder(project,job['id'])/f"attempt-{int(job['attempts'])}.log")
        # Resident runtimes share one attempt log; only count steps after the latest stage change.
        if text is not None and 'Stage: ' in text:text=text[text.rindex('Stage: '):]
    return last_step(text) if text else None


class ProgressTracker:
    """Watches the active job about once a second: saves its percentage and stops it when either service asks to cancel it."""
    def __init__(self,store,worker,project,*,interval=1.0,steps=runtime_steps,clock=time.monotonic,cancel=None):
        self.store,self.worker,self.project,self.cancel=store,worker,project,cancel
        self.interval,self.steps,self.clock=interval,steps,clock
        self.stopping=threading.Event();self.thread=None
        self.current=None;self.attempt=None;self.best=0;self.started={};self.stopping_job=False

    def start(self):
        self.thread=threading.Thread(target=self.run,name='job-progress',daemon=True);self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:self.thread.join(timeout=5)

    def run(self):
        while not self.stopping.wait(self.interval):
            try:self.poll()
            except Exception:log.debug('Progress update skipped; generation continues',exc_info=True)

    def poll(self):
        job=self.worker.job
        if job is None:
            self.attempt=None;return None
        key=(job['id'],job['attempts'])
        if key!=self.attempt:
            self.attempt,self.current,self.best,self.started,self.stopping_job=key,None,0,{},False
        if self.cancel and not self.stopping_job and self.store.cancel_requested(*key):
            self.stopping_job=True;self.cancel()
        progress=self.measure(job,self.worker.stage)
        if self.changed(progress):
            if self.store.save_live_progress(job['id'],job['attempts'],progress):self.current=progress
        return progress

    def measure(self,job,stage):
        count=image_count(job);stage=stage or 'starting'
        option=re.fullmatch(r'generating_option_(\d+)_of_\d+',stage)
        index=int(option[1])-1 if option else 0 if stage=='generating' else None
        step=None;fraction=0
        if index is None:
            phase=stage if stage in STAGES else 'starting'
        else:
            self.started.setdefault(index,self.clock())
            step=self.steps(self.project,job,index)
            if step is None or step[0]==0:phase='loading_model'
            elif step[0]<step[1]:phase='sampling';fraction=step[0]/step[1]
            else:phase='finishing';fraction=1
        done=index or 0
        # Never move backwards within an attempt, and reach 100 only once the job succeeds.
        self.best=max(self.best,min(99,int(100*(done+SAMPLING_SHARE*fraction)/count)))
        remaining=None
        if step is not None:
            finished=[self.started[i+1]-self.started[i] for i in range(done) if i in self.started and i+1 in self.started]
            per_image=sum(finished)/len(finished) if finished else step[1]*step[2]
            remaining=round((step[1]-step[0])*step[2]+(count-done-1)*per_image)
        return {'percent':self.best,'phase':phase,'image':None if index is None else index+1,'total_images':count,
                'step':step[0] if step else None,'total_steps':step[1] if step else None,
                'estimated_remaining_seconds':remaining,'updated':time.time()}

    def changed(self,progress):
        if self.current is None:return True
        keys=('percent','phase','image','step','total_steps')
        if any(progress[key]!=self.current[key] for key in keys):return True
        before,after=self.current['estimated_remaining_seconds'],progress['estimated_remaining_seconds']
        if (before is None)!=(after is None) or (after is not None and abs(after-before)>=5):return True
        return progress['updated']-self.current['updated']>=15
