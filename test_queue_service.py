"""Independent queue CRUD, model routing, deferred preparation and startup lifecycle."""
import contextlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from job_queue import JobStore, fingerprint
import queue_api
import queue_launch
import launch
import server
import test_api


class QueueServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.app=queue_api.create_app(self.root)
        self.client=TestClient(self.app)

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def add(self,model='z-image-turbo',**request):
        return self.client.post('/v1/jobs',json={'model':model,'request':{'prompt':'A castle',**request}})

    def test_offline_model_scoped_crud_pagination_and_history(self):
        # No setup, mounted SSD, generator process or model assets exist in this project.
        z=self.add(); self.assertEqual(z.status_code,202,z.text)
        z=z.json(); flash=self.add('chroma1-flash').json()
        self.assertEqual(self.client.get('/').status_code,404)  # UI belongs only to generator.
        for model,expected in [('z-image-turbo',z['id']),('chroma1-flash',flash['id'])]:
            page=self.client.get('/v1/jobs',params={'model':model,'limit':1}).json()
            self.assertEqual(page['total'],1);self.assertEqual(page['data'][0]['id'],expected)
            self.assertTrue((self.root/'state'/'queues'/model/'jobs.sqlite3').is_file())
        detail=f"/v1/jobs/{z['id']}?model=z-image-turbo"
        updated=self.client.patch(detail,json={'request':{'prompt':'A forest','generate_immediately':True}})
        self.assertEqual(updated.status_code,200,updated.text)
        self.assertTrue(updated.json()['generate_immediately'])
        self.assertEqual(updated.json()['request']['model'],'z-image-turbo')
        self.assertEqual(self.client.get(f"/v1/jobs/{z['id']}?model=chroma1-flash").status_code,404)
        self.assertEqual(self.client.delete(detail).json()['status'],'cancelled')
        retried=self.client.post(f"/v1/jobs/{z['id']}/retry?model=z-image-turbo").json()
        self.assertNotEqual(retried['id'],z['id']);self.assertEqual(retried['retry_of'],z['id'])
        self.assertEqual(self.client.delete('/v1/jobs?model=z-image-turbo').json(),{'cancelled':1})
        self.assertEqual(self.client.get('/v1/jobs?model=chroma1-flash&status=pending').json()['total'],1)
        for path in ['/v1/jobs','/v1/queue',f"/v1/jobs/{z['id']}"]:
            self.assertEqual(self.client.get(path).status_code,200)
        implicit=self.client.post('/v1/jobs',json={'request':{'prompt':'Default model'}})
        self.assertEqual(implicit.status_code,202,implicit.text)
        self.assertEqual(implicit.json()['model'],'z-image-turbo')
        self.assertEqual(self.add('../bad').status_code,422)
        self.assertEqual(self.add(model='z-image-turbo',generate_immediately='true').status_code,422)
        mismatch=self.client.post('/v1/jobs',json={'model':'z-image-turbo','request':{'model':'chroma1-flash'}})
        self.assertEqual(mismatch.status_code,422)

    def test_idempotency_updates_racing_claims_and_clear_preserve_running(self):
        body={'model':'z-image-turbo','request':{'prompt':'Fox'}};headers={'Idempotency-Key':'agent-1'}
        first=self.client.post('/v1/jobs',json=body,headers=headers).json()
        again=self.client.post('/v1/jobs',json=body,headers=headers).json()
        self.assertEqual(first['id'],again['id'])
        path=f"/v1/jobs/{first['id']}?model=z-image-turbo"
        edit=self.client.patch(path,json={'request':{'prompt':'Owl'}})
        self.assertEqual(edit.status_code,200)
        self.assertEqual(self.client.post('/v1/jobs',json=body,headers=headers).status_code,409)
        body['request']['prompt']='Owl'
        self.assertEqual(self.client.post('/v1/jobs',json=body,headers=headers).json()['id'],first['id'])
        store=self.app.state.store('z-image-turbo');store.claim()
        self.assertEqual(self.client.patch(path,json={'request':{'prompt':'Other'}}).status_code,409)
        self.assertEqual(self.client.delete(path).status_code,409)
        self.add()
        self.assertEqual(self.client.delete('/v1/jobs?model=z-image-turbo').json(),{'cancelled':1})
        self.assertEqual(store.get(first['id'])['status'],'running')
        # Queue API restart must NOT recover running inference or acquire its worker lease.
        other=TestClient(queue_api.create_app(self.root))
        try:
            self.assertEqual(other.get(path).json()['status'],'running')
        finally:other.close()
        store.recover()  # Generator-only operation after acquiring lease in production.
        self.assertEqual(self.client.patch(path,json={'request':{'prompt':'Other'}}).status_code,409)

    def test_v2_migration_preserves_already_resolved_requests(self):
        import sqlite3
        store=JobStore(self.root/'legacy-v2.sqlite3');store.initialize()
        data={'model':'z-image-turbo','prompt':'Legacy','seed':42,'generate_immediately':False}
        old=store.enqueue(data,fingerprint(data),adapter_sha='a'*64)
        editable=store.enqueue({**data,'prompt':'Can edit'},fingerprint({**data,'prompt':'Can edit'}))
        with sqlite3.connect(store.path) as db:
            db.execute('ALTER TABLE jobs DROP COLUMN prepared')
            db.execute('PRAGMA user_version=2')
        store.initialize()
        claimed=store.claim()
        frozen=store.prepare(claimed['id'],{**data,'seed':99},'b'*64)
        self.assertEqual(frozen['request'],data)
        self.assertEqual(frozen['_adapter_sha'],'a'*64)
        self.assertEqual(frozen['id'],old['id'])
        replacement={'model':'z-image-turbo','prompt':'Edited legacy request','generate_immediately':False}
        store.update(editable['id'],replacement)
        second=store.claim()
        self.assertEqual(store.prepare(second['id'],{**replacement,'seed':99})['request']['seed'],99)
        with sqlite3.connect(store.path) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],3)

    def test_queue_imports_do_not_load_inference_or_worker(self):
        code="import sys,queue_api; assert not any(k in sys.modules for k in ['server','config','lora_catalog','generation_worker','mlx','torch','mflux','prompt_expander','engine','chroma_engine'])"
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class DeferredGeneratorTest(unittest.TestCase):
    setUp=test_api.APITest.setUp
    tearDown=test_api.APITest.tearDown

    def add(self,**settings):
        return self.queue_http.post('/v1/jobs',json={'model':server.API_MODEL,
            'request':{'prompt':'A fox','size':'256x256','output_dir':str(self.root/'images'),**settings}})

    def test_raw_agent_request_is_frozen_once_and_reconciles_after_restart(self):
        self.worker.stop(); self.job_executor.shutdown(wait=True)
        queued=self.add(n=2).json()
        self.assertNotIn('seed',queued['request'])
        job=self.store.claim();first=server.execute_job(job)
        frozen=self.store.get(job['id'])['request'];self.assertIsInstance(frozen['seed'],int)
        self.assertEqual(first['images'][1]['seed'],(first['images'][0]['seed']+1)%2**32)
        self.store.recover();claimed=self.store.claim();self.fake.calls.clear()
        second=server.execute_job(claimed)
        self.assertEqual(first,second);self.assertEqual(self.fake.calls,[])
        self.assertEqual(self.store.get(job['id'])['request'],frozen)

    def test_bad_raw_request_fails_and_later_request_generates(self):
        invalid=self.add(steps=0).json();valid=self.add().json()
        from test_queue import until
        until(lambda:self.store.get(valid['id'])['status']=='succeeded')
        self.assertEqual(self.store.get(invalid['id'])['status'],'failed')
        self.assertIn('steps',self.store.get(invalid['id'])['error'])
        self.assertEqual(self.client.post('/v1/jobs/'+invalid['id']+'/retry').status_code,422)
        response=self.client.get('/v1/jobs/'+valid['id'])
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(Path(response.json()['result']['data'][0]['file_path']).is_file())

    def test_generator_crud_proxies_and_queue_failure_retains_accepted_jobs(self):
        self.store.pause(True)
        accepted=self.client.post('/v1/jobs',json=self.payload).json()
        changed=self.client.patch('/v1/jobs/'+accepted['id'],json={**self.payload,'prompt':'A raven'})
        self.assertEqual(changed.status_code,200,changed.text)
        self.assertEqual(self.store.get(accepted['id'])['request']['prompt'],'A raven')
        with patch.object(server.queue_client,'call',side_effect=server.HTTPException(503,'Queue unavailable')):
            self.assertEqual(self.client.post('/v1/jobs',json=self.payload).status_code,503)
        self.assertEqual(self.store.summary()['queued'],1)


class QueueStartupTest(unittest.TestCase):
    def test_existing_correct_queue_is_reused_and_unrelated_listener_is_preserved(self):
        with patch.object(queue_launch,'start_lock',return_value=contextlib.nullcontext(Path('/tmp'))),patch.object(queue_launch,'state',return_value={'status':'ready'}),patch.object(queue_launch.subprocess,'Popen') as start:
            queue_launch.ensure_queue();start.assert_not_called()
        with patch.object(queue_launch,'start_lock',return_value=contextlib.nullcontext(Path('/tmp'))),patch.object(queue_launch,'state',return_value=None),patch.object(queue_launch,'port_free',return_value=False),patch.object(queue_launch.subprocess,'Popen') as start:
            with self.assertRaisesRegex(SystemExit,'another service'):queue_launch.ensure_queue()
            start.assert_not_called()

    def test_generator_requires_distinct_ports_and_does_not_stop_queue(self):
        with patch.object(launch.sys,'argv',['launch.py','--model','z-image-turbo','--no-llm','--queue-port','5732']),patch.object(launch,'check_assets'),patch.object(launch,'ensure_queue') as ensure:
            with self.assertRaisesRegex(SystemExit,'different'):launch.main()
            ensure.assert_not_called()


if __name__=='__main__':unittest.main()
