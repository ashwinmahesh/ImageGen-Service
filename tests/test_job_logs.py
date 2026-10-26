"""Attributable, bounded job diagnostics, independent of resource sampling."""
from contextlib import redirect_stdout,redirect_stderr
import io
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient
from job_logs import capture_runtime,read_job_logs,record_event,runtime_folder
from job_queue import fingerprint
from queue_storage import initialize_shared_queue
from observability_api import create_router


class JobLogsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.env=patch.dict(os.environ,{'DATA_DIR':str(self.root/'data')});self.env.start()
        self.store=initialize_shared_queue(self.root)
        app=FastAPI();app.include_router(create_router(lambda:None,self.root));self.client=TestClient(app)
    def tearDown(self):self.client.close();self.env.stop();self.temp.cleanup()
    def job(self,model='z-image-turbo',**settings):
        payload={'model':model,'prompt':'A lighthouse',**settings}
        job=self.store.enqueue(payload,fingerprint(payload));return self.store.claim()
    def test_capture_isolates_jobs_attempts_and_restores_streams_after_error(self):
        one=self.job();terminal=io.StringIO()
        with redirect_stdout(terminal),redirect_stderr(terminal):
            with capture_runtime(self.root,one):
                print('First output');logging.warning('A test warning')
            self.store.finish(one['id'],result={})
            two=self.job()
            with self.assertRaisesRegex(ValueError,'synthetic failure'):
                with capture_runtime(self.root,two):
                    print('Second output');raise ValueError('synthetic failure')
            print('Outside capture')
        text=read_job_logs(self.root,one['id'])['sources'][0]['text']
        self.assertIn('First output',text);self.assertIn('test warning',text)
        self.assertNotIn('Second output',text);self.assertNotIn('Outside capture',text)
        second=read_job_logs(self.root,two['id'])['sources'][0]['text']
        self.assertIn('ValueError: synthetic failure',second);self.assertNotIn('First output',second)
        one['attempts']=2;record_event(self.root,one,'Second attempt')
        with self.store.connection(write=True) as db:db.execute('UPDATE jobs SET attempts=2 WHERE id=?',(one['id'],))
        self.assertEqual([s['attempt'] for s in read_job_logs(self.root,one['id'])['sources']],[2,1])
    def test_tail_budget_model_scope_and_legacy_native_batch_ids(self):
        job=self.job('qwen-image-2.1-turbo',n=2)
        folder=self.root/'data/qwen-image-2.1-turbo/logs';folder.mkdir(parents=True)
        for index in range(2):
            identifier=uuid.uuid5(uuid.UUID(job['id']),f'image-{index+1}').hex
            (folder/(identifier+'.log')).write_text('x'*3000+'\x1b[31mFinish\x1b[0m\rDone\x00')
        result=read_job_logs(self.root,job['id'],max_bytes=1024)
        self.assertEqual(len(result['sources']),2)
        self.assertLessEqual(sum(s['bytes_read'] for s in result['sources']),1024)
        for source in result['sources']:
            self.assertTrue(source['truncated']);self.assertIn('Finish\nDone',source['text']);self.assertNotIn('\x1b',source['text'])
            self.assertIsNone(source['attempt'])
        self.assertIn('not separated by attempt',result['note'])
        with self.assertRaises(FileNotFoundError):read_job_logs(self.root,job['id'],model='z-image-turbo')
    def test_old_missing_and_unrelated_shared_logs_are_not_exposed(self):
        job=self.job();folder=self.root/'state/model-process';folder.mkdir()
        (folder/'server.log').write_text('unattributable private shared log')
        result=read_job_logs(self.root,job['id']);self.assertFalse(result['available']);self.assertEqual(result['sources'],[])
        with self.assertRaises(FileNotFoundError):read_job_logs(self.root,'f'*32)
        empty=self.root/'untouched'
        with self.assertRaises(FileNotFoundError):read_job_logs(empty,'f'*32)
        self.assertFalse(empty.exists())
    def test_symlinks_and_nonregular_logs_are_not_followed(self):
        job=self.job();folder=runtime_folder(self.root,job['id']);folder.mkdir(parents=True)
        secret=self.root/'secret';secret.write_text('private')
        log=folder/'attempt-1.log';log.symlink_to(secret)
        self.assertEqual(read_job_logs(self.root,job['id'])['sources'],[])
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            with capture_runtime(self.root,job):print('Inference still works')
        self.assertEqual(secret.read_text(),'private');log.unlink();os.mkfifo(log)
        self.assertEqual(read_job_logs(self.root,job['id'])['sources'],[])
    def test_capture_disk_failure_is_nonfatal_and_storage_read_errors_are_distinct(self):
        job=self.job()
        with patch('job_logs.capture_file',side_effect=OSError('Disk full')),self.assertLogs('job_logs',level='WARNING'):
            with capture_runtime(self.root,job):completed=True
            record_event(self.root,job,'Does not break scheduler')
        self.assertTrue(completed)
        with patch('observability_api.read_job_logs',side_effect=OSError('storage')):
            result=self.client.get('/v1/observability/jobs/'+job['id']+'/logs')
            self.assertEqual(result.status_code,503);self.assertEqual(result.json()['detail']['code'],'job_logs_unavailable')
    def test_logs_api_works_without_sampler_and_validates_limits(self):
        job=self.job();record_event(self.root,job,'<script>plain text</script>')
        path='/v1/observability/jobs/'+job['id']+'/logs'
        self.assertEqual(self.client.get('/v1/observability').status_code,503)
        result=self.client.get(path);self.assertEqual(result.status_code,200,result.text)
        self.assertIn('<script>',result.json()['sources'][0]['text'])
        for params in ({'max_bytes':0},{'max_bytes':262145},{'model':'unknown'}):
            self.assertEqual(self.client.get(path,params=params).status_code,422)
        self.assertEqual(self.client.get(path,params={'model':'chroma1-flash'}).status_code,404)
        self.assertEqual(self.client.get('/v1/observability/jobs/invalid/logs').status_code,404)
        self.assertIn(path.replace(job['id'],'{identifier}'),self.client.get('/openapi.json').json()['paths'])
