"""Boundary timing, durable history, runtime attribution and nonfatal telemetry."""
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from engine import ZImageEngine
from model_lifecycle import Timing,operation
from model_process import ModelProcess,ModelProcessError
from observability import MetricStore,ResourceMonitor
from observability_api import create_router
from tests.test_observability import FakeProbe

JOB='a'*32
PROJECT=Path(__file__).resolve().parents[1]


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.monitor=ResourceMonitor(self.temp.name,'z-image-turbo',probe=FakeProbe());self.monitor.ensure_store();self.store=self.monitor.store;self.now=time.time()
        app=FastAPI();app.include_router(create_router(lambda:self.monitor,PROJECT));self.client=TestClient(app)

    def tearDown(self):self.client.close();self.monitor.stop();self.temp.cleanup()

    def event(self,number,*,model='z-image-turbo',action='load',scope='weights',status='succeeded',started=None,seconds=2,job=JOB,session=None):
        event={'id':f'{number:032x}','phase':'started','action':action,'scope':scope,'reason':'initial_load','started':self.now-100 if started is None else started}
        session=session or self.monitor.session;owner={'id':job,'attempts':1} if job else None
        self.store.lifecycle(session,model,event,owner)
        if status!='running':self.store.lifecycle(session,model,{**event,'phase':'finished','status':status,'finished':None if status=='interrupted' else event['started']+seconds,'duration_seconds':None if status=='interrupted' else seconds},owner)
        return event

    def test_persistence_aggregates_exclude_failed_and_interrupted_timings(self):
        self.event(1,seconds=2);self.event(2,seconds=4);self.event(3,status='failed',seconds=99);self.event(4,status='interrupted');self.event(5,status='running')
        other=MetricStore(self.temp.name);data=other.lifecycle_history(limit=2);self.assertEqual(data['total'],5);self.assertEqual(len(data['data']),2);self.assertTrue(data['has_more'])
        summary=data['summary'][0];self.assertEqual((summary['succeeded'],summary['failed'],summary['interrupted'],summary['running']),(2,1,1,1));self.assertEqual(summary['mean_seconds'],3);self.assertEqual(summary['peak_seconds'],4)
        self.assertEqual(len(other.lifecycle_history(limit=2,offset=2)['data']),2)
        self.assertIsNone(next(item for item in other.lifecycle_history()['data'] if item['status']=='interrupted')['duration_seconds'])

    def test_retention_filters_overlap_and_unknown_job_are_independent_of_samples(self):
        self.event(1,started=self.now-100*86400);self.event(2,action='unload',scope='process',started=self.now-50,model='chroma1-flash')
        self.event(3,started=self.now-200,seconds=150);self.event(4,started=self.now-200,status='running');self.event(5,started=self.now-200,status='interrupted')
        self.store.prune(self.now);self.assertEqual(self.store.lifecycle_history()['total'],5)
        filtered=self.store.lifecycle_history(start=self.now-100,end=self.now,model='z-image-turbo');self.assertEqual({item['id'] for item in filtered['data']},{f'{n:032x}' for n in [3,4]})
        self.assertEqual(self.store.lifecycle_history(action='unload',scope='process')['total'],1)
        self.assertEqual(self.client.get('/v1/observability/lifecycle?window=all&job_id='+'b'*32).json()['total'],0)
        self.assertEqual(self.client.get('/v1/observability/lifecycle?window=all').json()['total'],5)
        self.assertEqual(self.client.get('/v1/observability/lifecycle?window=30d').json()['total'],4)

    def test_reconciliation_only_explicitly_interrupts_previous_sessions_with_unknown_duration(self):
        self.event(1,status='running',session='old');self.event(2,status='running')
        self.store.initialize();self.client.get('/v1/observability/lifecycle');self.assertTrue(all(item['status']=='running' for item in self.store.lifecycle_history()['data']))
        self.monitor.recover_jobs();data={item['session']:item for item in self.store.lifecycle_history()['data']}
        self.assertEqual(data['old']['status'],'interrupted');self.assertIsNone(data['old']['finished']);self.assertIsNone(data['old']['duration_seconds']);self.assertEqual(data[self.monitor.session]['status'],'running')

    def test_v1_upgrade_preserves_samples_and_sessions_without_backfill(self):
        with self.store.connection() as db:
            db.execute('DROP TABLE lifecycle_events');db.execute('PRAGMA user_version=1');before=db.execute('SELECT * FROM sessions').fetchall()
        upgraded=MetricStore(self.temp.name);upgraded.initialize()
        with upgraded.connection() as db:self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM sessions')],[tuple(row) for row in before]);self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],2)
        self.assertEqual(upgraded.lifecycle_history()['total'],0)
        self.assertEqual(upgraded.series(self.now-100,self.now)['points'],[])

    def test_api_bounds_definitions_and_event_filters(self):
        self.event(1);self.event(2,action='unload',scope='process',model='chroma1-flash')
        for window in ['24h','7d','30d','90d','all']:
            result=self.client.get('/v1/observability/lifecycle',params={'window':window,'limit':1});self.assertEqual(result.status_code,200,result.text);self.assertEqual(result.json()['total'],2);self.assertEqual(len(result.json()['data']),1)
        result=self.client.get('/v1/observability/lifecycle?model=chroma1-flash&scope=process&action=unload').json();self.assertEqual(result['total'],1);self.assertEqual(result['data'][0]['model'],'chroma1-flash')
        for params in [{'window':'5m'},{'model':'missing'},{'scope':'gpu'},{'action':'restart'},{'limit':101},{'offset':-1},{'job_id':'bad'},{'end':'nan'},{'end':self.now+100},{'end':'inf'}]:self.assertEqual(self.client.get('/v1/observability/lifecycle',params=params).status_code,422,params)
        definitions=self.client.get('/v1/observability').json();self.assertIn('lifecycle_scopes',definitions);self.assertEqual(definitions['lifecycle_windows'],['24h','7d','30d','90d','all'])
        self.assertIn('/v1/observability/lifecycle',self.client.get('/openapi.json').json()['paths'])

    def test_lifecycle_store_failure_does_not_change_inference_or_resource_reads(self):
        with patch.object(self.store,'lifecycle',side_effect=sqlite3.OperationalError('locked')),self.assertLogs('image-observability',level='WARNING'):
            self.monitor.lifecycle({'phase':'started'},'z-image-turbo')
        self.assertIsNotNone(self.monitor.status()['lifecycle_error']);self.monitor.sample();self.assertIsNotNone(self.monitor.status()['lifecycle_error'])
        with patch.object(self.store,'lifecycle_history',side_effect=sqlite3.OperationalError('disk')):
            response=self.client.get('/v1/observability/lifecycle');self.assertEqual(response.status_code,503);self.assertEqual(response.json()['detail']['code'],'observability_unavailable');self.assertEqual(self.client.get('/v1/observability/metrics').status_code,200)

    def test_monotonic_timing_ignores_wall_clock_changes_and_finishes_once(self):
        events=[];timer=Timing('load','weights','initial_load',emit=events.append,clock=Mock(side_effect=[100,50]),monotonic=Mock(side_effect=[10,13.25]));timer.finish();timer.finish('failed')
        self.assertEqual(len(events),2);self.assertEqual(events[1]['duration_seconds'],3.25);self.assertEqual(events[1]['finished'],50)
        with patch('model_lifecycle._sink',side_effect=OSError('no history')),self.assertLogs('model_lifecycle',level='WARNING'):
            with operation('load','weights','initial_load'):pass

    def test_resident_model_reuse_adapter_changes_and_load_failure_have_real_boundaries(self):
        events=[]
        class Fake:
            def generate_image(self,**settings):return 'image'
        selection={'file_path':'style','sha256':'hash','strength':1}
        with patch('model_lifecycle._sink',events.append):
            engine=ZImageEngine(factory=lambda selection:Fake(),clear_cache=lambda:None);engine.generate_image(prompt='Lake');engine.generate_image(prompt='Lake',lora=selection);engine.generate_image(prompt='Lake',lora=selection)
            with patch.object(engine,'_factory',side_effect=RuntimeError('bad weights')):
                with self.assertRaises(RuntimeError):engine.generate_image(prompt='Lake',lora=None)
        ends=[event for event in events if event['phase']=='finished'];self.assertEqual([(event['action'],event['status']) for event in ends],[('load','succeeded'),('unload','succeeded'),('load','succeeded'),('unload','succeeded'),('load','failed')])
        self.assertTrue(all(event['duration_seconds']>=0 for event in ends))

    def test_abrupt_child_release_marks_unfinished_weight_event_interrupted(self):
        event=self.event(1,status='running');manager=ModelProcess(self.temp.name,observer=self.monitor);manager.process=Mock(pid=123);manager.active_model='z-image-turbo';manager.pending_lifecycle[event['id']]=(event,'z-image-turbo',{'id':JOB,'attempts':1})
        with patch.object(manager,'group_has_live_members',return_value=False):manager.close(reason='error')
        weight=next(row for row in self.store.lifecycle_history()['data'] if row['scope']=='weights');self.assertEqual(weight['status'],'interrupted');self.assertIsNone(weight['duration_seconds'])

    def test_failed_release_stays_fatal_and_is_recorded_without_claiming_unload(self):
        manager=ModelProcess(self.temp.name,observer=self.monitor);manager.process=Mock(pid=123);manager.active_model='z-image-turbo';(Path(self.temp.name)/'state/model-process').mkdir()
        with patch.object(manager,'group_has_live_members',return_value=True),patch('os.killpg',side_effect=PermissionError('denied')),self.assertLogs('model_process',level='ERROR'):
            with self.assertRaises(ModelProcessError) as failure:manager.close()
        self.assertTrue(failure.exception.fatal);self.assertIsNotNone(manager.process);event=self.store.lifecycle_history()['data'][0];self.assertEqual(event['status'],'failed');self.assertEqual(event['action'],'unload')

    def test_updated_child_does_not_send_new_events_to_a_preupgrade_parent(self):
        from tests.test_runtime_models import RuntimeModelsTest
        from runtime_protocol import send,receive
        from queue_storage import queue_database
        fixture=RuntimeModelsTest();fixture.setUp();manager=ModelProcess(fixture.root,llm_enabled=False,command=fixture.child_command())
        try:
            fixture.add();job=fixture.store.claim();manager.select('z-image-turbo')
            send(manager.channel,{'action':'execute','database_path':str(queue_database(fixture.root)),'job':job,'llm_enabled':False,'loras_enabled':False})
            kinds=[]
            while True:
                event,fds=receive(manager.channel);self.assertFalse(fds);kinds.append(event['event'])
                if event['event']=='error':self.fail(event['message'])
                if event['event']=='result':break
            self.assertNotIn('ready',kinds);self.assertNotIn('lifecycle',kinds)
        finally:manager.close();fixture.tearDown()

    def test_real_fake_runtime_switch_reuse_shutdown_and_job_attribution(self):
        # Exercise real IPC/process exit using the existing fake-backend fixture.
        from tests.test_runtime_models import RuntimeModelsTest
        fixture=RuntimeModelsTest();fixture.setUp();monitor=ResourceMonitor(fixture.root,'z-image-turbo',probe=FakeProbe());manager=ModelProcess(fixture.root,llm_enabled=False,observer=monitor,command=fixture.child_command())
        try:
            for model in ['z-image-turbo','z-image-turbo','flux2-klein-4b']:fixture.add(model)
            jobs=[]
            for _ in range(3):
                job=fixture.store.claim();jobs.append(job);result=manager.execute(job);fixture.store.finish(job['id'],result=result)
            manager.close()
            events=monitor.store.lifecycle_history()['data'];self.assertEqual(len(events),4);self.assertTrue(all(event['status']=='succeeded' and event['duration_seconds']>=0 for event in events))
            self.assertFalse(any(event['job_id']==jobs[1]['id'] for event in events))
            switch=next(event for event in events if event['action']=='unload' and event['model']=='z-image-turbo');self.assertEqual(switch['job_id'],jobs[2]['id']);self.assertEqual(switch['reason'],'model_switch')
            stop=next(event for event in events if event['action']=='unload' and event['model']=='flux2-klein-4b');self.assertIsNone(stop['job_id']);self.assertEqual(stop['reason'],'shutdown')
        finally:manager.close();monitor.stop();fixture.tearDown()


if __name__=='__main__':unittest.main()
