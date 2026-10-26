"""Generator-only queue consumption and accelerator ownership."""
import fcntl
import logging
import os
from pathlib import Path
import threading
from storage_paths import require_storage

log=logging.getLogger('image-generator')

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
