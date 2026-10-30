"""Restart, priority, cancellation, idempotency and worker ownership without GPU work."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import server
from job_queue import JobStore, QueueConflict, fingerprint
from generation_worker import QueueWorker
from tests import test_api as api_tests


def until(predicate):
    deadline=time.monotonic()+5
    while not predicate():
        if time.monotonic()>deadline: raise AssertionError('Worker did not reach expected state')
        time.sleep(.01)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.store=JobStore(self.root/'queue/jobs.sqlite3'); self.store.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def add(self,name='first',immediate=False):
        data={'model':'z-image-turbo','prompt':name,'generate_immediately':immediate,'seed':42}
        return self.store.enqueue(data,fingerprint(data))

    def test_persistence_priority_fifo_pause_and_recovery(self):
        first=self.add('first'); second=self.add('second'); urgent1=self.add('urgent1',True); urgent2=self.add('urgent2',True)
        self.store=JobStore(self.store.path); self.store.initialize()
        self.store.pause(True)
        claimed=self.store.claim(); self.assertEqual(claimed['id'],urgent1['id'])
        self.store.finish(claimed['id'],result={'ok':True})
        claimed=self.store.claim(); self.assertEqual(claimed['id'],urgent2['id'])
        self.assertEqual(self.store.recover(),1)
        retried=self.store.claim()
        self.assertEqual(retried['id'],urgent2['id']); self.assertEqual(retried['attempts'],2)
        self.assertEqual(retried['request']['seed'],42)
        self.store.finish(retried['id'],result={'ok':True})
        self.assertIsNone(self.store.claim())
        self.store.pause(False)
        for expected in [first,second]:
            row=self.store.claim(); self.assertEqual(row['id'],expected['id']); self.store.finish(row['id'],result={'ok':True})

    def test_clear_cancel_and_pagination_preserve_running_and_results(self):
        first=self.add(); active=self.store.claim(); self.add('second'); self.add('third')
        stopping=self.store.cancel(active['id'])  # Running work is stopped by the generator, not cleared.
        self.assertEqual((stopping['status'],stopping['cancel_requested']),('running',True))
        self.assertEqual(self.store.clear(),2)
        self.assertEqual(self.store.get(first['id'])['status'],'running')
        page,total=self.store.listing(limit=1,offset=1,status='cancelled')
        self.assertEqual(total,2); self.assertEqual(len(page),1)
        self.store.finish(first['id'],result={'path':'kept'})
        self.assertEqual(self.store.clear(),0)
        self.assertEqual(self.store.get(first['id'])['result'],{'path':'kept'})

    def test_concurrent_same_idempotency_key_has_one_record(self):
        data={'model':'z-image-turbo','prompt':'fox','seed':42,'generate_immediately':False}
        with ThreadPoolExecutor(max_workers=8) as pool:
            records=list(pool.map(lambda _:self.store.enqueue(data,fingerprint(data),key='stable'),range(20)))
        self.assertEqual(len({item['id'] for item in records}),1)
        self.assertEqual(self.store.summary()['queued'],1)
        with self.assertRaises(QueueConflict): self.store.enqueue({**data,'prompt':'other'},fingerprint({**data,'prompt':'other'}),key='stable')

    def test_live_worker_lease_blocks_another_owner(self):
        first=QueueWorker(self.store,lambda _:None); first.acquire()
        second=QueueWorker(self.store,lambda _:None)
        try:
            with self.assertRaisesRegex(RuntimeError,'live worker'): second.acquire()
        finally:
            first.release(); second.release()

    def test_external_mount_failure_does_not_create_fallback_database(self):
        missing=JobStore(Path('/Volumes/image-queue-test-missing/jobs.sqlite3'))
        with patch('storage_paths.Path.is_mount',return_value=False),patch('job_queue.sqlite3.connect') as connect:
            with self.assertRaisesRegex(RuntimeError,'not mounted'): missing.initialize()
            connect.assert_not_called()

    def test_future_database_schema_is_not_silently_downgraded(self):
        with closing(sqlite3.connect(self.store.path)) as db: db.execute('PRAGMA user_version=99')
        with self.assertRaisesRegex(RuntimeError,'Unsupported'): self.store.initialize()
        with closing(sqlite3.connect(self.store.path)) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],99)

    def test_abrupt_process_exit_recovers_the_original_job(self):
        job=self.add(); self.add('second')
        code="from pathlib import Path; from job_queue import JobStore; import os,sys; store=JobStore(Path(sys.argv[1])); store.claim(); os._exit(17)"
        process=subprocess.run([sys.executable,'-c',code,str(self.store.path)],cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(process.returncode,17)
        self.assertEqual(self.store.get(job['id'])['status'],'running')
        worker=QueueWorker(self.store,lambda _:None); worker.acquire()
        try:
            self.assertEqual(self.store.claim()['id'],job['id'])
        finally: worker.release()

    def test_native_child_keeps_leases_after_parent_crashes(self):
        job=self.add()
        code="""from pathlib import Path
import os,subprocess,sys
from job_queue import JobStore
from generation_worker import QueueWorker
store=JobStore(Path(sys.argv[1]));worker=QueueWorker(store,lambda _:None,gpu_lock_file=Path(sys.argv[2]))
worker.acquire();gpu=worker.acquire_accelerator();store.claim()
subprocess.Popen([sys.executable,'-c','import time;time.sleep(1)'],pass_fds=(worker.lease.fileno(),gpu.fileno()),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
os._exit(18)
"""
        gpu_path=self.root/'native-gpu.lock'
        process=subprocess.run([sys.executable,'-c',code,str(self.store.path),str(gpu_path)],cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(process.returncode,18)
        with gpu_path.open('a+') as handle:
            with self.assertRaises(BlockingIOError): fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recovered=QueueWorker(self.store,lambda _:None)
        recovered.acquire()  # Waits for the inherited native lease, then recovers exactly this job.
        try:
            self.assertEqual(self.store.get(job['id'])['status'],'queued')
            with gpu_path.open('a+') as handle: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally: recovered.release()

    def test_gpu_contention_waits_and_stop_preserves_unstarted_job(self):
        gpu=self.root/'gpu.lock'; handle=gpu.open('a+'); fcntl.flock(handle,fcntl.LOCK_EX)
        called=threading.Event(); job=self.add()
        worker=QueueWorker(self.store,lambda _:called.set(),gpu_lock_file=gpu); worker.acquire()
        pool=ThreadPoolExecutor(max_workers=1); worker.start(pool)
        try:
            until(lambda:worker.stage=='waiting_for_accelerator')
            self.assertFalse(called.is_set())
            self.assertEqual(self.store.get(job['id'])['status'],'queued')
            worker.stop()
            self.assertEqual(self.store.get(job['id'])['status'],'queued')
            self.assertFalse(called.is_set())
        finally:
            handle.close(); worker.stop(); pool.shutdown(wait=True)

    def test_worker_finishes_active_then_runs_priority_fifo_without_overlap(self):
        release=threading.Event(); started=threading.Event(); order=[]
        first=self.add('first')
        def execute(job):
            order.append(job['request']['prompt'])
            if job['id']==first['id']:
                started.set(); self.assertTrue(release.wait(4))
            return {'ok':True}
        worker=QueueWorker(self.store,execute); worker.acquire()
        pool=ThreadPoolExecutor(max_workers=1); worker.start(pool)
        try:
            self.assertTrue(started.wait(2))
            self.add('normal'); self.add('urgent1',True); self.add('urgent2',True)
            worker.wake.set(); release.set()
            until(lambda:self.store.summary()['succeeded']==4)
            self.assertEqual(order,['first','urgent1','urgent2','normal'])
        finally:
            release.set(); worker.stop(); pool.shutdown(wait=True)


class QueueAPITest(unittest.TestCase):
    setUp=api_tests.APITest.setUp
    tearDown=api_tests.APITest.tearDown

    def enqueue(self,**changes):
        return self.client.post('/v1/jobs',json={**self.payload,**changes})

    def test_async_pagination_delete_clear_and_persisted_pause(self):
        self.assertEqual(self.client.patch('/v1/queue',json={'paused':True}).status_code,200)
        jobs=[]
        for _ in range(3):
            response=self.enqueue(); self.assertEqual(response.status_code,202,response.text); jobs.append(response.json())
        self.assertFalse(jobs[0]['generate_immediately'])
        self.assertEqual(self.client.get('/v1/jobs?limit=2').json()['next_offset'],2)
        self.assertEqual(self.client.get('/v1/jobs?status=queued&offset=2&limit=2').json()['previous_offset'],0)
        self.assertEqual(self.client.delete('/v1/jobs/'+jobs[0]['id']).json()['status'],'cancelled')
        self.assertEqual(self.client.delete('/v1/jobs').json(),{'cancelled':2})
        self.assertEqual(self.client.get('/health').json()['queue']['queued'],0)
        self.assertTrue(self.store.summary()['paused'])
        for bad in [{'paused':'yes'},{'extra':True,'paused':False}]:
            self.assertEqual(self.client.patch('/v1/queue',json=bad).status_code,422)

    def test_http_idempotency_keeps_random_seed_and_catches_conflict(self):
        self.store.pause(True)
        payload={**self.payload,'seed':None}
        headers={'Idempotency-Key':'same-image'}
        first=self.client.post('/v1/jobs',json=payload,headers=headers)
        second=self.client.post('/v1/jobs',json=payload,headers=headers)
        self.assertEqual(first.status_code,202,first.text); self.assertEqual(second.status_code,202,second.text)
        self.assertEqual(first.json()['id'],second.json()['id'])
        self.assertEqual(first.json()['request']['seed'],second.json()['request']['seed'])
        self.assertIsNone(first.json()['request']['seed'])  # Deferred until the consumer freezes defaults.
        self.assertEqual(self.client.post('/v1/jobs',json={**payload,'prompt':'other'},headers=headers).status_code,409)
        self.assertEqual(self.enqueue(generate_immediately='true').status_code,422)
        for query in ['limit=0','offset=-1','status=invalid']:
            self.assertEqual(self.client.get('/v1/jobs?'+query).status_code,422)

    def test_complete_artifact_is_reconciled_after_missing_sql_success_commit(self):
        self.worker.stop(); self.job_executor.shutdown(wait=True)
        record=server.submit_job(server.GenerateRequest(**self.payload))
        claimed=self.store.claim()
        metadata=server.execute_job(claimed)
        original=Path(metadata['file_path']).read_bytes()
        self.assertEqual(self.store.get(record['id'])['status'],'running')
        self.store.recover(); claimed=self.store.claim()
        self.fake.calls.clear()
        result=server.execute_job(claimed)
        self.store.finish(claimed['id'],result=result)
        self.assertEqual(self.fake.calls,[])
        self.assertEqual(Path(metadata['file_path']).read_bytes(),original)
        read=self.client.get('/v1/jobs/'+record['id'])
        self.assertEqual(read.status_code,200,read.text)
        self.assertEqual(read.json()['status'],'succeeded')
        self.assertEqual(self.client.get(read.json()['result']['data'][0]['url']).status_code,200)

    def test_deleted_reference_fails_visibly_and_later_jobs_continue(self):
        self.store.pause(True)
        reference=self.root/'ref.png'; Image.new('RGB',(256,256),'red').save(reference)
        first=self.enqueue(reference_image=str(reference)).json()
        reference.unlink()
        second=self.enqueue().json()
        self.store.pause(False); self.worker.wake.set()
        until(lambda:self.store.get(second['id'])['status']=='succeeded')
        self.assertEqual(self.store.get(first['id'])['status'],'failed')
        retry=self.client.post('/v1/jobs/'+first['id']+'/retry')
        self.assertEqual(retry.status_code,400)  # Missing input must be restored before retry.

    def test_immediate_waits_next_and_active_is_not_preempted(self):
        entered=threading.Event(); release=threading.Event()
        original=self.fake.generate_image
        def blocked(**settings):
            if settings['prompt']=='first': entered.set(); self.assertTrue(release.wait(4))
            return original(**settings)
        self.fake.generate_image=blocked
        first=self.enqueue(prompt='first').json()
        try:
            self.assertTrue(entered.wait(2))
            normal=self.enqueue(prompt='normal').json()
            urgent=self.enqueue(prompt='urgent',generate_immediately=True)
            self.assertEqual(urgent.status_code,202,urgent.text)
            self.assertEqual(self.client.get('/health').json()['queue']['running'],1)
        finally: release.set()
        until(lambda:self.store.get(normal['id'])['status']=='succeeded')
        self.assertEqual([call['prompt'] for call in self.fake.calls],['first','urgent','normal'])

    def test_read_and_clear_remain_available_if_worker_stops(self):
        self.store.pause(True)
        job=self.enqueue().json()
        self.worker.error='Stopped for a configuration problem'
        try:
            self.assertEqual(self.client.get('/v1/jobs').status_code,200)
            self.assertEqual(self.client.get('/v1/jobs/'+job['id']).status_code,200)
            self.assertEqual(self.enqueue().status_code,503)
            self.assertEqual(self.client.delete('/v1/jobs').json(),{'cancelled':1})
        finally:self.worker.error=None

    def test_unavailable_queue_volume_returns_503(self):
        with patch('job_queue.require_storage',side_effect=RuntimeError('Disconnected volume')):
            self.assertEqual(self.client.get('/v1/jobs').status_code,503)
            self.assertEqual(self.client.delete('/v1/jobs').status_code,503)
