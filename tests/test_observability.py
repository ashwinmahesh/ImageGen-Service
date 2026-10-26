"""Durable resource history, source semantics and failure isolation; no ML/GPU load."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import plistlib
import sqlite3
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from generation_worker import QueueWorker
from job_queue import JobStore,fingerprint
from observability import MetricStore,ResourceMonitor,WINDOWS,RETENTION_SECONDS
from observability_api import create_router
from resource_probe import FIELDS,ResourceProbe,gpu_counters

ROOT=Path(__file__).resolve().parents[1]
JOB='a'*32


class FakeProbe:
    pid=123
    status={'cpu_memory':'available','gpu':'available','process_tree_partial':False}
    def collect(self):
        return dict(zip(FIELDS,[12,33,2*1024**3,8*1024**3,24*1024**3,70,1024**3]),process_count=2,process_tree_partial=False)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.store=MetricStore(self.root);self.store.initialize();self.now=time.time()
        self.store.session('s1','z-image-turbo',12,self.now-100)

    def tearDown(self):self.temp.cleanup()

    def sample(self,t,*,model='z-image-turbo',session='s1',job=JOB,cpu=20,gpu=None):
        self.store.save({'t':t,'model':model,'session':session,'job_id':job,'attempt':1,'stage':'generating',
                         **FakeProbe().collect(),'service_cpu_percent':cpu,'gpu_utilization_percent':gpu})

    def test_persistence_buckets_peaks_gaps_models_and_missing_values(self):
        self.sample(self.now-299,cpu=10);self.sample(self.now-298,cpu=80)
        self.sample(self.now-10,model='chroma1-flash',session='s2',cpu=40)
        store=MetricStore(self.root)
        data=store.series(self.now-300,self.now,max_points=30)
        self.assertEqual(data['summary']['sample_count'],3)
        self.assertEqual(len(data['points']),2)  # no synthetic points in the long gap
        self.assertEqual(data['points'][0]['values']['service_cpu_percent'],45)
        self.assertEqual(data['points'][0]['peaks']['service_cpu_percent'],80)
        self.assertIsNone(data['points'][0]['values']['gpu_utilization_percent'])
        self.assertEqual(data['points'][1]['models'],['chroma1-flash'])
        self.assertEqual(data['points'][1]['sessions'],['s2'])
        self.assertEqual(store.series(self.now-300,self.now,model='z-image-turbo')['summary']['sample_count'],2)

    def test_series_stays_bounded_including_exact_endpoints(self):
        for index in range(101):self.sample(self.now-100+index)
        self.assertLessEqual(len(self.store.series(self.now-100,self.now,max_points=30)['points']),30)

    def test_retention_keeps_completed_summary_and_pagination(self):
        start=self.now-RETENTION_SECONDS-100
        job={'id':JOB,'attempts':1};self.store.begin('s1','z-image-turbo',job,start)
        self.sample(start+1,gpu=61);self.store.finish('s1','z-image-turbo',job,'succeeded',start+2)
        self.store.prune(self.now)
        result=self.store.job(JOB,now=self.now)
        self.assertEqual(result['points'],[])
        self.assertEqual(result['runs'][0]['summary']['metrics']['gpu_utilization_percent']['peak'],61)
        self.assertEqual(self.store.jobs(limit=1)['total'],1)
        self.assertEqual(self.store.jobs(limit=1,offset=1)['data'],[])

    def test_recovery_closes_old_attempt_without_touching_another_model(self):
        old={'id':JOB,'attempts':1};self.store.begin('s1','z-image-turbo',old,self.now-10)
        self.sample(self.now-5,gpu=80)
        other={'id':'b'*32,'attempts':1};self.store.begin('other','chroma1-flash',other,self.now-8)
        self.store.recover('z-image-turbo')
        runs=self.store.jobs()['data'];first=next(run for run in runs if run['job_id']==JOB)
        self.assertEqual(first['status'],'interrupted');self.assertEqual(first['finished'],self.now-5)
        self.assertEqual(first['summary']['metrics']['gpu_utilization_percent']['peak'],80)
        self.assertEqual(next(run for run in runs if run['job_id']==other['id'])['status'],'running')
        self.store.begin('new','z-image-turbo',{'id':JOB,'attempts':2},self.now-1)
        self.assertEqual(len(self.store.job(JOB,now=self.now)['runs']),2)

    def test_future_schema_and_external_state_symlink_are_preserved(self):
        with closing(sqlite3.connect(self.store.path)) as db:db.execute('PRAGMA user_version=99')
        with self.assertRaisesRegex(RuntimeError,'Unsupported'):self.store.initialize()
        with closing(sqlite3.connect(self.store.path)) as db:self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],99)
        other=self.root/'fresh';other.mkdir();(other/'state').symlink_to(self.root/'state',target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'inside the project'):MetricStore(other)


class ProbeTest(unittest.TestCase):
    def process(self,pid,created,cpu,rss,command=None):
        process=Mock(pid=pid);process.oneshot.return_value.__enter__=Mock();process.oneshot.return_value.__exit__=Mock()
        process.create_time.return_value=created;process.cpu_times.return_value=types.SimpleNamespace(user=cpu,system=0)
        process.memory_info.return_value=types.SimpleNamespace(rss=rss);process.cmdline.return_value=command or []
        return process

    def test_service_tree_cpu_normalization_ram_and_queue_exclusion(self):
        root=self.process(1,50,1,100);child=self.process(2,50,.5,50)
        queue=self.process(3,50,1000,999,command=['python','/project/queue_api.py'])
        root.children.return_value=[child,queue]
        library=Mock();library.Process.return_value=root;library.cpu_count.return_value=4
        library.NoSuchProcess=ProcessLookupError;library.ZombieProcess=ProcessLookupError;library.AccessDenied=PermissionError
        library.virtual_memory.return_value=types.SimpleNamespace(total=1000,available=300)
        library.cpu_times.side_effect=[types.SimpleNamespace(_asdict=lambda:{'user':10,'idle':90}),types.SimpleNamespace(_asdict=lambda:{'user':20,'idle':100})]
        probe=ResourceProbe('/project',pid=1,clock=Mock(side_effect=[10,12]),wall=Mock(side_effect=[100,102]),gpu=lambda:(0,0))
        with patch('resource_probe.importlib.import_module',return_value=library):
            first=probe.collect();self.assertIsNone(first['service_cpu_percent'])
            root.cpu_times.return_value.user=3;child.cpu_times.return_value.user=2.5
            second=probe.collect()
        self.assertEqual(second['service_cpu_percent'],50)
        self.assertEqual(second['system_cpu_percent'],50)
        self.assertEqual(second['service_rss_bytes'],150)
        self.assertEqual(second['system_memory_used_bytes'],700)
        self.assertEqual(second['process_count'],2)
        self.assertEqual(second['gpu_utilization_percent'],0) # measured zero is valid

    def test_missing_dependency_or_gpu_is_null_and_backed_off(self):
        gpu=Mock(side_effect=OSError('restricted'))
        probe=ResourceProbe('/project',clock=lambda:10,gpu=gpu)
        with patch('resource_probe.importlib.import_module',side_effect=ImportError):
            first=probe.collect();second=probe.collect()
        self.assertIsNone(first['service_rss_bytes']);self.assertIsNone(second['gpu_utilization_percent'])
        self.assertEqual(gpu.call_count,1)
        self.assertIn('setup-env',probe.status['cpu_memory'])

    def test_gpu_driver_counters_are_parsed_and_unknown_is_not_zero(self):
        payload=plistlib.dumps([{'PerformanceStatistics':{'Device Utilization %':42,'In use system memory':1024}}])
        with patch('resource_probe.platform.system',return_value='Darwin'),patch('resource_probe.subprocess.run',return_value=types.SimpleNamespace(stdout=payload)) as run:
            self.assertEqual(gpu_counters(),(42,1024));self.assertEqual(run.call_args.kwargs['timeout'],1)
        with patch('resource_probe.platform.system',return_value='Darwin'),patch('resource_probe.subprocess.run',return_value=types.SimpleNamespace(stdout=plistlib.dumps([]))):
            self.assertEqual(gpu_counters(),(None,None))


class MonitorTest(unittest.TestCase):
    def test_short_job_boundary_samples_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=ResourceMonitor(temp,'z-image-turbo',probe=FakeProbe())
            job={'id':JOB,'attempts':1};monitor.start_job(job);monitor.finish_job(job,'succeeded');monitor.stop()
            result=MetricStore(temp).job(JOB)
            self.assertEqual(result['summary']['sample_count'],2)
            self.assertEqual(result['runs'][0]['status'],'succeeded')
            self.assertTrue(all(point['models']==['z-image-turbo'] for point in result['points']))

    def test_job_cadence_wakes_an_idle_sampler_and_shutdown_is_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            probe=FakeProbe();probe.collect=Mock(wraps=probe.collect)
            monitor=ResourceMonitor(temp,'z-image-turbo',probe=probe);monitor.start()
            try:
                deadline=time.monotonic()+2
                while probe.collect.call_count<1 and time.monotonic()<deadline:time.sleep(.01)
                monitor.start_job({'id':JOB,'attempts':1});count=probe.collect.call_count
                deadline=time.monotonic()+2
                while probe.collect.call_count==count and time.monotonic()<deadline:time.sleep(.01)
                self.assertGreater(probe.collect.call_count,count)
            finally:
                started=time.monotonic();monitor.stop();self.assertLess(time.monotonic()-started,1)
                self.assertFalse(monitor.thread.is_alive())

    def test_worker_records_success_failure_and_does_not_depend_on_observer(self):
        for broken in (False,True):
            with self.subTest(broken=broken),tempfile.TemporaryDirectory() as temp:
                store=JobStore(Path(temp)/'queue/jobs.sqlite3');store.initialize()
                monitor=ResourceMonitor(temp,'z-image-turbo',probe=FakeProbe())
                observer=Mock() if broken else monitor
                if broken:observer.start_job.side_effect=RuntimeError('observer failed');observer.finish_job.side_effect=RuntimeError('observer failed')
                request={'model':'z-image-turbo','prompt':'test'};job=store.enqueue(request,fingerprint(request))
                def execute(_):return {'ok':True}
                worker=QueueWorker(store,execute,observer=observer);worker.acquire()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    worker.start(executor)
                    try:
                        deadline=time.monotonic()+3
                        while store.get(job['id'])['status']!='succeeded' and time.monotonic()<deadline:time.sleep(.01)
                        self.assertEqual(store.get(job['id'])['status'],'succeeded');self.assertIsNone(worker.error)
                    finally:worker.stop()
                if not broken:self.assertEqual(monitor.store.job(job['id'])['runs'][0]['status'],'succeeded')

    def test_failed_execution_still_records_its_resource_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            store=JobStore(Path(temp)/'queue/jobs.sqlite3');store.initialize()
            monitor=ResourceMonitor(temp,'chroma1-flash',probe=FakeProbe())
            request={'model':'chroma1-flash','prompt':'test'};job=store.enqueue(request,fingerprint(request))
            execute=Mock(side_effect=RuntimeError('fake inference failure'))
            worker=QueueWorker(store,execute,observer=monitor);worker.acquire()
            with patch('generation_worker.log'),ThreadPoolExecutor(max_workers=1) as executor:
                worker.start(executor)
                try:
                    deadline=time.monotonic()+3
                    while store.get(job['id'])['status']!='failed' and time.monotonic()<deadline:time.sleep(.01)
                    self.assertEqual(store.get(job['id'])['status'],'failed')
                finally:worker.stop()
            result=monitor.store.job(job['id']);self.assertEqual(result['runs'][0]['status'],'failed')
            self.assertEqual(result['summary']['sample_count'],2)

    def test_sampling_storage_outage_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=ResourceMonitor(temp,'z-image-turbo',probe=FakeProbe());job={'id':JOB,'attempts':1}
            with patch.object(monitor.store,'initialize',side_effect=sqlite3.OperationalError('disk')):
                monitor.start_job(job);monitor.finish_job(job,'succeeded')
            self.assertIsNone(monitor.active);self.assertIsNotNone(monitor.status()['error'])


class LifecycleTest(unittest.TestCase):
    def test_startup_failure_stops_sampling_and_unavailable_sampling_does_not_block_startup(self):
        import server
        async def enter():
            async with server.lifespan(server.app):pass
        # Inference startup failure must not leave a sampler thread behind.
        monitor=Mock();worker=Mock(future=None)
        with patch.object(server,'ResourceMonitor',return_value=monitor),patch.object(server,'initialize_service',side_effect=RuntimeError('fake load failure')), \
             patch.object(server,'executor',ThreadPoolExecutor(max_workers=1)),patch.object(server,'job_worker',worker), \
             patch.object(server,'observability_monitor',None),patch.object(server,'observability_error',None):
            with self.assertRaisesRegex(RuntimeError,'fake load failure'):asyncio.run(enter())
            monitor.start.assert_called_once();monitor.stop.assert_called_once()
        # An invalid/unwritable observability path is independent of generation.
        worker=Mock(future=object())
        with patch.object(server,'ResourceMonitor',side_effect=RuntimeError('invalid metrics directory')),patch.object(server,'initialize_service') as initialize, \
             patch.object(server,'executor',ThreadPoolExecutor(max_workers=1)),patch.object(server,'job_worker',worker), \
             patch.object(server,'observability_monitor',None),patch.object(server,'observability_error',None),patch.object(server,'log'):
            asyncio.run(enter());initialize.assert_called_once();worker.start.assert_called_once();worker.stop.assert_called_once()


class APITest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.monitor=ResourceMonitor(self.temp.name,'z-image-turbo',probe=FakeProbe())
        self.monitor.start_job({'id':JOB,'attempts':1});self.monitor.finish_job({'id':JOB,'attempts':1},'succeeded')
        app=FastAPI();app.include_router(create_router(lambda:self.monitor,ROOT));self.client=TestClient(app)

    def tearDown(self):self.client.close();self.monitor.stop();self.temp.cleanup()

    def test_all_windows_job_history_pagination_and_schema(self):
        status=self.client.get('/v1/observability').json();self.assertEqual(status['windows'],list(WINDOWS))
        self.assertIn('Whole-device',status['scopes']['gpu_utilization_percent'])
        for window in WINDOWS:
            response=self.client.get('/v1/observability/metrics',params={'window':window});self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['summary']['sample_count'],2)
        self.assertEqual(self.client.get('/v1/observability/jobs?limit=1').json()['total'],1)
        self.assertEqual(self.client.get('/v1/observability/jobs?offset=1').json()['data'],[])
        detail=self.client.get('/v1/observability/jobs/'+JOB+'?window=1m').json()
        self.assertEqual(detail['runs'][0]['model'],'z-image-turbo');self.assertEqual(detail['summary']['sample_count'],2)
        self.assertIn('/v1/observability/metrics',self.client.get('/openapi.json').json()['paths'])

    def test_input_validation_and_unknown_job(self):
        for params in [{'window':'2h'},{'model':'unknown'},{'max_points':1201},{'end':'nan'},{'end':time.time()+1000}]:
            self.assertEqual(self.client.get('/v1/observability/metrics',params=params).status_code,422)
        self.assertEqual(self.client.get('/v1/observability/jobs/not-a-job').status_code,404)
        self.assertEqual(self.client.get('/v1/observability/jobs/'+'f'*32).status_code,404)

    def test_storage_failure_has_its_own_error_and_page_is_local(self):
        with patch.object(self.monitor.store,'series',side_effect=sqlite3.OperationalError('disk')):
            response=self.client.get('/v1/observability/metrics');self.assertEqual(response.status_code,503)
            self.assertEqual(response.json()['detail']['code'],'observability_unavailable')
        for path in ['/observability','/assets/observability.js','/assets/observability.css']:
            self.assertEqual(self.client.get(path).status_code,200)
        self.assertIn('Live updates',self.client.get('/observability').text)


class GeneratorIntegrationTest(unittest.TestCase):
    def test_real_generator_routes_and_fake_inference_record_the_same_job(self):
        import server
        from tests.test_api import APITest as GeneratorFixture
        fixture=GeneratorFixture();fixture.setUp()
        monitor=ResourceMonitor(fixture.root,'z-image-turbo',probe=FakeProbe())
        fixture.worker.observer=monitor
        patcher=patch.object(server,'observability_monitor',monitor);patcher.start();fixture.patches.append(patcher)
        try:
            accepted=fixture.client.post('/v1/jobs',json=fixture.payload)
            self.assertEqual(accepted.status_code,202,accepted.text);identifier=accepted.json()['id']
            deadline=time.monotonic()+3;response=None
            while time.monotonic()<deadline:
                response=fixture.client.get('/v1/observability/jobs/'+identifier)
                if response.status_code==200 and response.json()['runs'][0]['status']=='succeeded':break
                time.sleep(.01)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['runs'][0]['status'],'succeeded')
            self.assertEqual(response.json()['runs'][0]['model'],server.API_MODEL)
            self.assertEqual(response.json()['summary']['sample_count'],2)
            generated=fixture.client.get('/v1/jobs/'+identifier).json()
            self.assertTrue(Path(generated['result']['data'][0]['file_path']).is_file())
            self.assertIn('observability',fixture.client.get('/health').json())
        finally:
            fixture.tearDown();monitor.stop()


if __name__=='__main__':unittest.main()
