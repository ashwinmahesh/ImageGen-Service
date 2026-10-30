"""Cancelling running jobs: shared request flag, generator-side stop and recorded outcome, without inference."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from generation_worker import QueueWorker
from job_progress import ProgressTracker
from job_queue import JobStore, QueueConflict, CANCELLED_WHILE_RUNNING
from model_process import ModelProcess
import queue_api
from tests.model_fixtures import setup_model


def until(condition,timeout=5):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if condition():return True
        time.sleep(.02)
    raise AssertionError('Condition was not reached')


class CancelStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=JobStore(Path(self.temp.name)/'jobs.sqlite3');self.store.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def add(self,prompt='A fox'):
        return self.store.enqueue({'prompt':prompt,'model':'z-image-turbo','n':2},prompt)

    def test_running_cancel_is_a_request_until_the_generator_records_it(self):
        job=self.add();running=self.store.claim()
        self.assertFalse(running['cancel_requested'])
        stopping=self.store.cancel(job['id'])
        self.assertEqual((stopping['status'],stopping['cancel_requested']),('running',True))
        self.assertTrue(self.store.cancel_requested(job['id'],running['attempts']))
        self.assertFalse(self.store.cancel_requested(job['id'],running['attempts']+1))
        self.store.save_progress(job['id'],{'created':1,'images':['first option']})
        self.store.finish(job['id'],cancelled=True)
        cancelled=self.store.get(job['id'])
        self.assertEqual((cancelled['status'],cancelled['error'],cancelled['cancel_requested']),('cancelled',CANCELLED_WHILE_RUNNING,False))
        self.assertEqual(cancelled['result'],{'created':1,'images':['first option']})
        self.assertIsNotNone(cancelled['finished'])
        with self.assertRaises(QueueConflict):self.store.cancel(job['id'])

    def test_waiting_cancel_is_immediate_and_retry_attempts_start_clear(self):
        waiting=self.add('waiting')
        self.assertEqual(self.store.cancel(waiting['id'])['status'],'cancelled')
        job=self.add();self.store.claim();self.store.cancel(job['id'])
        self.store.requeue(job['id']);again=self.store.claim()
        self.assertFalse(again['cancel_requested'])

    def test_restart_honours_a_pending_cancel_instead_of_retrying(self):
        stopped=self.add('stopped');other=self.add('other')
        self.store.claim();self.store.cancel(stopped['id']);self.store.claim()
        self.assertEqual(self.store.recover(),1)
        self.assertEqual(self.store.get(stopped['id'])['status'],'cancelled')
        self.assertEqual(self.store.get(other['id'])['status'],'queued')


class CancelWorkerTest(unittest.TestCase):
    def test_tracker_stops_the_runtime_and_the_next_job_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=JobStore(Path(temporary)/'jobs.sqlite3');store.initialize()
            stop=threading.Event();ran=[]
            def execute(job):
                ran.append(job['request']['prompt'])
                if job['request']['prompt']=='long':
                    # A killed runtime surfaces as a disconnected child.
                    if stop.wait(5):raise EOFError('Model process disconnected')
                return {'created':1}
            worker=QueueWorker(store,execute);worker.acquire()
            tracker=ProgressTracker(store,worker,Path(temporary),interval=.02,steps=lambda *args:None,cancel=stop.set)
            executor=ThreadPoolExecutor(max_workers=1)
            try:
                worker.start(executor);tracker.start()
                long=store.enqueue({'prompt':'long','model':'z-image-turbo'},'long')
                until(lambda:store.get(long['id'])['status']=='running')
                after=store.enqueue({'prompt':'after','model':'z-image-turbo'},'after')
                store.cancel(long['id'])
                until(lambda:store.get(after['id'])['status']=='succeeded')
            finally:
                tracker.stop();worker.stop();executor.shutdown()
            cancelled=store.get(long['id'])
            self.assertEqual((cancelled['status'],cancelled['error']),('cancelled',CANCELLED_WHILE_RUNNING))
            self.assertEqual(ran,['long','after'])


class CancelAPITest(unittest.TestCase):
    def test_queue_api_requests_running_cancels_and_rejects_finished_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();setup_model(root,'z-image-turbo')
            client=TestClient(queue_api.create_app(root))
            try:
                job=client.post('/v1/jobs',json={'model':'z-image-turbo','request':{'prompt':'A castle'}}).json()
                store=JobStore(root/'state'/'queue'/'jobs.sqlite3');running=store.claim()
                stopping=client.delete('/v1/jobs/'+job['id'])
                self.assertEqual(stopping.status_code,200,stopping.text)
                self.assertEqual((stopping.json()['status'],stopping.json()['cancel_requested']),('running',True))
                store.finish(job['id'],cancelled=True)
                self.assertEqual(client.get('/v1/jobs/'+job['id']).json()['status'],'cancelled')
                self.assertEqual(client.delete('/v1/jobs/'+job['id']).status_code,409)
            finally:client.close()


class InterruptTest(unittest.TestCase):
    def test_interrupt_ends_the_owned_runtime_group(self):
        manager=ModelProcess(Path(tempfile.gettempdir()))
        self.assertFalse(manager.interrupt())
        child=('import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);time.sleep(30)')
        manager.process=subprocess.Popen([sys.executable,'-c',child],start_new_session=True)
        try:
            time.sleep(.3)
            self.assertTrue(manager.interrupt());self.assertTrue(manager.cancelling)
            manager.process.wait(timeout=5)
            until(lambda:not ModelProcess.group_has_live_members(manager.process.pid))
        finally:
            if manager.process.poll() is None:os.killpg(manager.process.pid,signal.SIGKILL)


if __name__=='__main__':
    unittest.main()
