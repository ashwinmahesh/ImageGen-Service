"""Cross-model scheduling, real child IPC/reuse and readiness, with fake inference."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from generation_worker import QueueWorker
from job_queue import JobStore,fingerprint
from model_process import ModelProcess
from queue_storage import initialize_shared_queue
import queue_api
import server
from tests.model_fixtures import setup_model
from tests.test_queue import until


class RuntimeModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.env=patch.dict(os.environ,{'DATA_DIR':str(self.root),'PYTHONPATH':str(Path(__file__).resolve().parents[1])});self.env.start()
        for model in ('z-image-turbo','flux2-klein-4b','chroma1-flash'):setup_model(self.root,model)
        self.store=initialize_shared_queue(self.root)
        self.client=TestClient(queue_api.create_app(self.root))

    def tearDown(self):
        self.client.close();self.env.stop();self.temp.cleanup()

    def add(self,model='z-image-turbo',**settings):
        response=self.client.post('/v1/jobs',json={'model':model,'request':{'prompt':'A lighthouse','size':'256x256','response_format':'url',**settings}})
        self.assertEqual(response.status_code,202,response.text);return response.json()

    def test_discovery_and_admission_only_allow_ready_models(self):
        self.assertEqual({item['id'] for item in self.client.get('/v1/models').json()['data']},{'z-image-turbo','flux2-klein-4b','chroma1-flash'})
        schema=self.client.get('/openapi.json').json()['components']['schemas']['Submission']['properties']['model']
        self.assertEqual(set(schema['anyOf'][0]['enum']),{'z-image-turbo','flux2-klein-4b','chroma1-flash'})
        self.assertEqual(self.client.post('/v1/jobs',json={'model':'qwen-image-2.1-turbo','request':{'prompt':'A fox'}}).status_code,422)
        self.assertEqual(self.client.get('/v1/loras?model=qwen-image-2.1-turbo').status_code,422)
        (self.root/'flux2-klein-4b/weights/transformer/weights.safetensors').unlink()
        self.assertNotIn('flux2-klein-4b',{item['id'] for item in self.client.get('/v1/models').json()['data']})
        self.assertEqual(self.client.post('/v1/jobs',json={'model':'flux2-klein-4b','request':{'prompt':'A fox'}}).status_code,422)

    def test_shared_default_global_order_priority_pause_and_filters(self):
        self.store.set_setting('default_model','flux2-klein-4b')
        default=self.client.post('/v1/jobs',json={'request':{'prompt':'Default'}}).json()
        self.assertEqual(default['model'],'flux2-klein-4b')
        second=self.add();third=self.add('chroma1-flash')
        page=self.client.get('/v1/jobs').json();self.assertEqual(page['total'],3)
        self.assertEqual([item['id'] for item in page['data']],[default['id'],second['id'],third['id']])
        self.client.post('/v1/jobs/'+third['id']+'/move',json={'position':1})
        self.assertEqual(self.client.get('/v1/jobs?model=chroma1-flash').json()['data'][0]['queue_position'],1)
        self.client.patch('/v1/queue',json={'paused':True})
        self.assertIsNone(self.store.claim(eligible_models=['z-image-turbo','chroma1-flash','flux2-klein-4b']))
        self.client.post('/v1/jobs/'+second['id']+'/run-now')
        self.assertEqual(self.store.claim(eligible_models=['z-image-turbo'])['id'],second['id'])
        self.assertEqual(self.client.delete('/v1/jobs').json(),{'cancelled':2})
        self.assertEqual(self.store.get(second['id'])['status'],'running')

    def test_unavailable_model_and_disabled_capability_do_not_block_other_work(self):
        blocked=self.add('flux2-klein-4b');llm=self.add(expand_prompt=True);normal=self.add()
        selected=self.store.claim(eligible_models=['z-image-turbo'],llm_enabled=False)
        self.assertEqual(selected['id'],normal['id'])
        self.assertEqual(self.store.get(blocked['id'])['status'],'queued')
        self.assertEqual(self.store.get(llm['id'])['status'],'queued')

    def test_same_key_is_scoped_to_model_and_settings_stay_immutable(self):
        headers={'Idempotency-Key':'agent-job'}
        for model in ('z-image-turbo','flux2-klein-4b'):
            body={'model':model,'request':{'prompt':'Fox'}}
            first=self.client.post('/v1/jobs',json=body,headers=headers).json()
            replay=self.client.post('/v1/jobs',json=body,headers=headers).json()
            self.assertEqual(first['id'],replay['id'])
            changed=self.client.post('/v1/jobs',json={'model':model,'request':{'prompt':'Owl'}},headers=headers)
            self.assertEqual(changed.status_code,409)
        self.assertEqual(self.store.listing()[1],2)

    def child_command(self):
        script=self.root/'fake-runtime.py'
        script.write_text('''import sys,types
from pathlib import Path
import server,model_child
from tests.test_api import FakeModel
server.PROJECT_ROOT=Path('''+repr(str(self.root))+''')
sys.modules['mlx']=types.ModuleType('mlx')
sys.modules['mlx.core']=types.SimpleNamespace(clear_cache=lambda:None)
def load():
    if server.API_MODEL=='chroma1-flash':raise RuntimeError('Synthetic model-load failure')
    server.model=FakeModel()
server.load_model=load
model_child.main()
''')
        return [sys.executable,str(script)]

    def test_real_subprocess_reuse_switch_release_failure_and_recovery(self):
        stages=[]
        manager=ModelProcess(self.root,llm_enabled=False,loras_enabled=False,stage=stages.append,command=self.child_command())
        first=self.add(seed=123);second=self.add(seed=124);third=self.add('flux2-klein-4b',seed=125)
        try:
            one=self.store.claim(eligible_models=['z-image-turbo','flux2-klein-4b']);result=manager.execute(one)
            pid=manager.process.pid;self.assertTrue(manager.loaded)
            self.store.finish(one['id'],result=result)
            two=self.store.claim(eligible_models=['z-image-turbo','flux2-klein-4b']);result2=manager.execute(two)
            self.assertEqual(manager.process.pid,pid);self.store.finish(two['id'],result=result2)
            three=self.store.claim(eligible_models=['z-image-turbo','flux2-klein-4b']);result3=manager.execute(three)
            self.assertNotEqual(manager.process.pid,pid)
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            self.assertEqual([result['model'],result2['model'],result3['model']],['z-image-turbo','z-image-turbo','flux2-klein-4b'])
            self.assertEqual(result3['steps'],4);self.assertEqual(Path(result3['file_path']).parent,self.root/'flux2-klein-4b/images')
            # Completed artifact is reconciled after interruption without loading weights.
            manager.close();self.store.recover();recovered=self.store.claim(eligible_models=['flux2-klein-4b'])
            self.assertEqual(manager.execute(recovered),result3);self.assertFalse(manager.loaded)
            self.store.finish(recovered['id'],result=result3)
            self.assertIn('loading_model',stages)
        finally:manager.close()

    def test_failed_model_start_does_not_stop_later_jobs(self):
        manager=ModelProcess(self.root,llm_enabled=False,stage=lambda value:None,command=self.child_command())
        failed=self.add('chroma1-flash');good=self.add()
        worker=QueueWorker(self.store,manager.execute,eligible_models=lambda:['z-image-turbo','chroma1-flash'])
        executor=ThreadPoolExecutor(max_workers=1);worker.acquire()
        try:
            with self.assertLogs('image-generator',level='ERROR'):
                worker.start(executor);until(lambda:self.store.get(good['id'])['status']=='succeeded')
            self.assertEqual(self.store.get(failed['id'])['status'],'failed');self.assertIsNone(worker.error)
        finally:worker.stop();executor.shutdown();manager.close()

    def test_generator_validation_defaults_and_both_apis_use_same_queue(self):
        with patch.object(server,'PROJECT_ROOT',self.root),patch.object(server,'job_store',self.store),patch.object(server,'job_worker',types_worker()),patch.object(server,'MODEL_ROOTS',{key:self.root/key for key in server.PROFILES}),patch.object(server,'ROOT',self.root/'z-image-turbo'):
            self.store.set_setting('default_model','flux2-klein-4b')
            client=TestClient(server.app,base_url='http://127.0.0.1')
            try:
                accepted=client.post('/v1/jobs',json={'prompt':'A fox'});self.assertEqual(accepted.status_code,202,accepted.text)
                job=accepted.json();self.assertEqual(job['model'],'flux2-klein-4b');self.assertEqual(job['request']['steps'],4)
                self.assertEqual(self.client.get('/v1/jobs/'+job['id']).json()['model'],'flux2-klein-4b')
                self.assertEqual(client.post('/v1/jobs',json={'model':'flux2-klein-4b','prompt':'Fox','steps':9}).status_code,422)
                self.assertEqual(client.post('/v1/jobs',json={'model':'qwen-image-2.1-turbo','prompt':'Fox'}).status_code,422)
                self.assertEqual(client.get('/v1/models').json()['default_model'],'flux2-klein-4b')
                (self.root/'flux2-klein-4b/weights/transformer/weights.safetensors').unlink()
                self.assertIn('not set up',client.get('/v1/jobs/'+job['id']).json()['blocked_reason'])
            finally:client.close()

    def test_warm_runtime_retains_worker_lease_but_releases_job_gpu_lock(self):
        import fcntl
        manager=ModelProcess(self.root,llm_enabled=False,command=self.child_command())
        lease=(self.root/'worker-test.lock').open('a+');gpu=(self.root/'gpu-test.lock').open('a+')
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.add();job=self.store.claim();job['_inherited_fds']=(lease.fileno(),gpu.fileno())
            manager.execute(job);gpu.close();lease.close()
            with (self.root/'gpu-test.lock').open('a+') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (self.root/'worker-test.lock').open('a+') as probe:
                with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            manager.close()
            with (self.root/'worker-test.lock').open('a+') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:lease.close();gpu.close();manager.close()

    def test_native_descendant_locks_release_after_runtime_crash(self):
        import fcntl
        script=self.root/'crash-runtime.py'
        script.write_text("""import argparse,os,socket,subprocess,sys
from runtime_protocol import receive
parser=argparse.ArgumentParser();parser.add_argument('--socket-fd',type=int);args=parser.parse_args()
channel=socket.socket(fileno=args.socket_fd);message,fds=receive(channel)
subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],pass_fds=tuple(fds))
os._exit(3)
""")
        manager=ModelProcess(self.root,command=[sys.executable,str(script)])
        path=self.root/'gpu.lock'
        with path.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            job=self.store.claim() if self.store.has_work() else None
            queued=self.add();job=self.store.claim();job['_inherited_fds']=(lock.fileno(),)
            with self.assertRaises(EOFError):manager.execute(job)
        # A surviving native descendant would keep the flock after this parent close.
        with path.open('a+') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertIsNone(manager.process)

    def test_observability_labels_each_actual_job_model(self):
        from observability import ResourceMonitor
        from tests.test_observability import FakeProbe
        monitor=ResourceMonitor(self.root,'z-image-turbo',probe=FakeProbe())
        try:
            for identifier in ('z-image-turbo','flux2-klein-4b'):
                queued=self.add(identifier);job=self.store.claim()
                monitor.start_job(job);monitor.finish_job(job,'succeeded');self.store.finish(job['id'],result={'ok':True})
            self.assertEqual({row['model'] for row in monitor.store.jobs()['data']},{'z-image-turbo','flux2-klein-4b'})
        finally:monitor.stop()


def types_worker():
    import threading,types
    return types.SimpleNamespace(error=None,wake=threading.Event(),stage='idle')
