"""Persisted manual scheduling shared by the queue API and generator/UI."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from job_queue import JobStore, QueueConflict, fingerprint
import queue_api
from tests import test_api


class QueueOrderTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=JobStore(Path(self.temp.name)/'jobs.sqlite3'); self.store.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def add(self,prompt,immediate=False,key=None):
        payload={'model':'z-image-turbo','prompt':prompt,'generate_immediately':immediate,'seed':42}
        return self.store.enqueue(payload,fingerprint(payload),key=key)

    def waiting(self):
        rows,total=self.store.listing(status='queued',limit=100)
        self.assertEqual([row['queue_position'] for row in rows],list(range(1,total+1)))
        return [row['id'] for row in rows]

    def test_manual_order_overrides_priority_persists_and_drives_worker(self):
        a=self.add('first'); b=self.add('second'); urgent=self.add('urgent',True)
        self.assertEqual(self.waiting(),[urgent['id'],a['id'],b['id']])
        self.store.move(b['id'],1)
        self.store=JobStore(self.store.path); self.store.initialize()
        self.assertEqual(self.waiting(),[b['id'],urgent['id'],a['id']])
        self.assertEqual(self.store.summary()['next_job_id'],b['id'])
        self.store.move(urgent['id'],3)
        self.assertEqual(self.waiting(),[b['id'],a['id'],urgent['id']])
        self.assertTrue(self.store.has_work())
        for expected in [b,a,urgent]:
            claimed=self.store.claim()
            self.assertEqual(claimed['id'],expected['id']); self.assertIsNone(claimed['queue_position'])
            self.store.finish(claimed['id'],result={'ok':True})
        self.assertFalse(self.store.has_work())

    def test_run_now_preserves_identity_request_idempotency_and_pause(self):
        self.add('first'); urgent=self.add('urgent',True); last=self.add('last',key='same')
        self.store.pause(True)
        updated=self.store.run_now(last['id'])
        self.assertEqual(updated['queue_position'],1); self.assertTrue(updated['generate_immediately'])
        self.assertEqual(updated['request'],last['request'])
        self.assertEqual(updated['sequence'],last['sequence'])
        self.assertEqual(self.store.existing('same',fingerprint(last['request']))['id'],last['id'])
        self.assertEqual(self.store.enqueue(last['request'],fingerprint(last['request']),key='same')['id'],last['id'])
        self.assertEqual(self.store.summary()['queued'],3)
        self.assertEqual(self.store.summary()['next_job_id'],last['id'])
        self.assertEqual(self.store.claim()['id'],last['id'])
        self.assertEqual(self.store.claim()['id'],urgent['id'])
        self.assertFalse(self.store.has_work()); self.assertIsNone(self.store.claim())
        self.assertIsNone(self.store.summary()['next_job_id'])

    def test_recovered_jobs_can_move_without_losing_seed_expansion_or_progress(self):
        original=self.add('first'); later=self.add('later')
        self.store.claim(); frozen=self.store.prepare(original['id'],{**original['request'],'seed':123},'a'*64)
        self.store.save_expansion(original['id'],{'expanded_prompt':'kept'})
        self.store.save_progress(original['id'],{'images':[{'path':'kept.png'}]})
        self.store.recover(); self.store.move(original['id'],2); self.store.run_now(original['id'])
        claimed=self.store.claim()
        self.assertTrue(claimed['_prepared']); self.assertEqual(claimed['_adapter_sha'],'a'*64)
        self.assertEqual(claimed['request'],frozen['request']); self.assertEqual(claimed['attempts'],2)
        self.assertEqual(claimed['prompt_expansion'],{'expanded_prompt':'kept'})
        self.assertEqual(claimed['result'],{'images':[{'path':'kept.png'}]})
        self.assertEqual(self.waiting(),[later['id']])

    def test_invalid_moves_cancel_and_request_priority_changes(self):
        first=self.add('first'); second=self.add('second')
        for position in [0,-1,3,True,1.5]:
            with self.assertRaises(QueueConflict): self.store.move(second['id'],position)
        self.assertEqual(self.waiting(),[first['id'],second['id']])
        for method in [lambda:self.store.move('a'*32,1),lambda:self.store.run_now('a'*32)]:
            with self.assertRaises(FileNotFoundError): method()
        self.store.update(second['id'],{**second['request'],'generate_immediately':True})
        self.assertEqual(self.waiting(),[second['id'],first['id']])
        active=self.store.claim()
        for method in [lambda:self.store.move(active['id'],1),lambda:self.store.run_now(active['id'])]:
            with self.assertRaises(QueueConflict): method()
        self.store.finish(active['id'],result={'ok':True})
        with self.assertRaises(QueueConflict): self.store.run_now(active['id'])
        self.store.cancel(first['id']); self.assertEqual(self.waiting(),[])
        self.assertIsNone(self.store.get(first['id'])['queue_position'])

    def test_concurrent_moves_claims_and_admissions_keep_one_order(self):
        jobs=[self.add(str(index)) for index in range(12)]
        def action(index):
            try:
                if index%3==0: return self.store.claim()
                if index%3==1: return self.store.run_now(jobs[index%12]['id'])
                return self.store.move(jobs[index%12]['id'],1)
            except QueueConflict: return None  # A concurrent claim legitimately wins.
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(action,range(24)))
        self.waiting()
        rows,total=self.store.listing(limit=100)
        self.assertEqual(total,12); self.assertEqual(len({row['id'] for row in rows}),12)
        self.assertEqual(sum(row['attempts'] for row in rows),8)

    def test_schema3_migration_preserves_priority_history_pause_and_payload(self):
        a=self.add('first'); urgent=self.add('urgent',True); active=self.store.claim()
        self.store.pause(True)
        with closing(sqlite3.connect(self.store.path)) as db:
            db.execute('DROP INDEX queue_position_order')
            db.execute('ALTER TABLE jobs DROP COLUMN queue_position')
            db.execute('PRAGMA user_version=3'); db.commit()
        self.store.initialize()
        self.assertTrue(self.store.summary()['paused'])
        self.assertEqual(self.store.get(active['id'])['status'],'running')
        self.assertEqual(self.store.get(a['id'])['request'],a['request'])
        self.store.recover()
        self.assertEqual(self.waiting(),[urgent['id'],a['id']])
        self.assertEqual(self.store.claim()['id'],urgent['id'])
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],4)


class QueueSchedulingAPITest(unittest.TestCase):
    def test_shared_queue_api_scheduling_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            from tests.model_fixtures import setup_model
            for identifier in ('z-image-turbo','chroma1-flash'):setup_model(Path(folder),identifier)
            app=queue_api.create_app(folder)
            with TestClient(app) as client:
                def add(model='z-image-turbo'):
                    return client.post('/v1/jobs',json={'model':model,'request':{'prompt':'Lighthouse'}}).json()
                first=add(); second=add(); flash=add('chroma1-flash')
                path=f"/v1/jobs/{second['id']}"
                moved=client.post(path+'/move',json={'position':1})
                self.assertEqual(moved.status_code,200,moved.text); self.assertEqual(moved.json()['queue_position'],1)
                for body in [{'position':0},{'position':True},{'position':'1'},{'position':1,'unexpected':1},{}]:
                    self.assertEqual(client.post(path+'/move',json=body).status_code,422)
                self.assertEqual(client.post(path+'/move',json={'position':9}).status_code,409)
                self.assertEqual(client.post(path+'/run-now').status_code,200)
                app.state.store().pause(True)
                run=client.post(f"/v1/jobs/{first['id']}/run-now")
                self.assertEqual(run.status_code,200,run.text); self.assertTrue(run.json()['generate_immediately'])
                self.assertEqual(run.json()['request'],first['request'])
                app.state.store().claim()
                self.assertEqual(client.post(f"/v1/jobs/{first['id']}/run-now").status_code,409)
                self.assertEqual(client.post(f"/v1/jobs/{first['id']}/move",json={'position':1}).status_code,409)
                self.assertEqual(client.get('/v1/jobs?model=chroma1-flash').json()['data'][0]['id'],flash['id'])


class GeneratorSchedulingAPITest(unittest.TestCase):
    def setUp(self):
        test_api.APITest.setUp(self)
        self.worker.stop()  # Only the fixture's fake worker, so scheduling can be inspected deterministically.

    tearDown=test_api.APITest.tearDown

    def test_direct_generator_scheduling_survives_queue_http_outage(self):
        with patch.object(__import__('server').queue_client.transport,'request',side_effect=AssertionError('Must not proxy CRUD')):
            first=self.client.post('/v1/jobs',json=self.payload,headers={'Idempotency-Key':'stable'}).json()
            second=self.client.post('/v1/jobs',json={**self.payload,'prompt':'Mountain village'}).json()
            moved=self.client.post(f"/v1/jobs/{second['id']}/move",json={'position':1})
            self.assertEqual(moved.status_code,200,moved.text); self.assertEqual(moved.json()['queue_position'],1)
            run=self.client.post(f"/v1/jobs/{first['id']}/run-now")
            self.assertEqual(run.status_code,200,run.text); self.assertEqual(run.json()['queue_position'],1)
            self.assertEqual(run.json()['request'],first['request'])
            self.assertTrue(run.json()['generate_immediately'])
            again=self.client.post('/v1/jobs',json=self.payload,headers={'Idempotency-Key':'stable'})
            self.assertEqual(again.status_code,202,again.text); self.assertEqual(again.json()['id'],first['id'])
            page=self.client.get('/v1/jobs?status=queued&limit=1&offset=1').json()
            self.assertEqual(page['data'][0]['id'],second['id']); self.assertEqual(page['data'][0]['queue_position'],2)
            self.assertEqual(page['total'],2)
            self.store.claim()
            for action,kwargs in [('move',{'json':{'position':1}}),('run-now',{})]:
                self.assertEqual(self.client.post(f"/v1/jobs/{first['id']}/{action}",**kwargs).status_code,409)
                self.assertEqual(self.client.post('/v1/jobs/unknown/'+action,**kwargs).status_code,404)
            self.assertEqual(self.client.post(f"/v1/jobs/{second['id']}/move",json={'position':0}).status_code,422)
