"""Queue HTTP failure must not gate the generator's direct SQLite API."""
import http.client
import io
import json
from pathlib import Path
import sqlite3
import threading
import unittest
import urllib.error
from unittest.mock import patch
from queue_client import QueueClient
import server
from tests import test_api
from tests.test_queue import until


class HealthResponse(io.BytesIO):
    status=200


class QueueHealthTest(unittest.TestCase):
    def reply(self,**changes):
        return HealthResponse(json.dumps({'service':'imagegen-queue','status':'ready','project_dir':str(Path('/tmp/project').resolve()),**changes}).encode())

    def test_network_protocol_and_invalid_response_failures_are_bounded_and_recover(self):
        client=QueueClient(project='/tmp/project')
        failures=[ConnectionRefusedError(),TimeoutError(),urllib.error.URLError('refused'),
                  http.client.IncompleteRead(b'partial'),http.client.BadStatusLine('broken')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__),patch('queue_client.urllib.request.urlopen',side_effect=failure) as request:
                state=client.status();self.assertFalse(state['available'])
                self.assertIn('make queue',state['error'])
                self.assertEqual(request.call_count,1)  # No retries or supervised restart.
                self.assertEqual(request.call_args.kwargs['timeout'],1)
        for raw in [b'not JSON',b'[]',b'null',b'{}']:
            with patch('queue_client.urllib.request.urlopen',return_value=HealthResponse(raw)):
                self.assertFalse(client.status()['available'])
        for changes in [{'service':'different-service'},{'project_dir':'/tmp/other'},{'status':'loading'}]:
            with patch('queue_client.urllib.request.urlopen',return_value=self.reply(**changes)):
                self.assertFalse(client.status()['available'])
        with patch('queue_client.urllib.request.urlopen',return_value=self.reply()):
            state=client.status();self.assertTrue(state['available']);self.assertIsNone(state['error'])

    def test_http_error_is_closed_and_recovery_command_preserves_port(self):
        body=io.BytesIO(b'upstream error')
        failure=urllib.error.HTTPError('http://localhost/health',503,'Unavailable',{},body)
        client=QueueClient(54321)
        with patch('queue_client.urllib.request.urlopen',side_effect=failure):
            state=client.status()
        self.assertFalse(state['available']);self.assertTrue(body.closed)
        self.assertEqual(state['recovery_command'],'make queue QUEUE_PORT=54321')
        self.assertIn('This page and already accepted jobs can continue',state['error'])


class GeneratorQueueOutageTest(unittest.TestCase):
    setUp=test_api.APITest.setUp
    tearDown=test_api.APITest.tearDown

    def outage(self):
        return patch.object(self.queue_http,'request',side_effect=ConnectionRefusedError('queue process is down'))

    def test_all_ui_crud_uses_sqlite_and_recovers_agent_visibility(self):
        with self.outage():
            self.assertEqual(self.client.patch('/v1/queue',json={'paused':True}).status_code,200)
            first=self.client.post('/v1/jobs',json=self.payload).json()
            changed=self.client.patch('/v1/jobs/'+first['id'],json={**self.payload,'prompt':'A raven'})
            self.assertEqual(changed.status_code,200,changed.text)
            self.assertEqual(self.store.get(first['id'])['request']['prompt'],'A raven')
            page=self.client.get('/v1/jobs?status=pending&limit=1').json()
            self.assertEqual(page['total'],1)
            self.assertEqual(self.client.get('/v1/jobs/'+first['id']).status_code,200)
            self.assertEqual(self.client.delete('/v1/jobs/'+first['id']).json()['status'],'cancelled')
            retry=self.client.post('/v1/jobs/'+first['id']+'/retry')
            self.assertEqual(retry.status_code,202,retry.text)
            self.assertEqual(retry.json()['retry_of'],first['id'])
            self.assertEqual(self.client.delete('/v1/jobs').json(),{'cancelled':1})
            health=self.client.get('/health').json()
            self.assertEqual(health['status'],'ready')
            self.assertFalse(health['queue_api']['available'])
            self.assertIsNone(health['queue']['error'])
        # Restored standalone API sees the exact same database/IDs, without a generator restart.
        self.assertTrue(self.client.get('/health').json()['queue_api']['available'])
        records=self.queue_http.get('/v1/jobs').json()['data']
        self.assertEqual({item['id'] for item in records},{first['id'],retry.json()['id']})

    def test_active_and_new_jobs_finish_and_results_remain_visible_during_outage(self):
        entered=threading.Event();release=threading.Event()
        original=self.fake.generate_image
        def blocked(**settings):
            if settings['prompt']=='first':
                entered.set();self.assertTrue(release.wait(5))
            return original(**settings)
        self.fake.generate_image=blocked
        first=self.client.post('/v1/jobs',json={**self.payload,'prompt':'first'}).json()
        try:
            self.assertTrue(entered.wait(2))
            with self.outage():
                state=self.client.get('/health').json()
                self.assertTrue(state['busy']);self.assertFalse(state['queue_api']['available'])
                self.assertEqual(self.client.get('/v1/jobs/'+first['id']).json()['status'],'running')
                second=self.client.post('/v1/jobs',json={**self.payload,'prompt':'second'})
                self.assertEqual(second.status_code,202,second.text)
                release.set()
                until(lambda:self.store.get(second.json()['id'])['status']=='succeeded')
                saved=self.client.get('/v1/jobs/'+second.json()['id'])
                self.assertEqual(saved.status_code,200,saved.text)
                self.assertEqual(self.client.get(saved.json()['result']['data'][0]['url']).status_code,200)
                self.assertEqual(self.client.get('/v1/images').json()['total'],2)
                self.assertIsNone(self.worker.error)
        finally:release.set()
        self.assertEqual([entry['prompt'] for entry in self.fake.calls],['first','second'])

    def test_synchronous_admission_and_idempotency_work_without_queue_http(self):
        with self.outage():
            headers={'Idempotency-Key':'during-outage'}
            self.client.patch('/v1/queue',json={'paused':True})
            first=self.client.post('/v1/jobs',json=self.payload,headers=headers)
            repeated=self.client.post('/v1/jobs',json=self.payload,headers=headers)
            self.assertEqual(first.status_code,202,first.text)
            self.assertEqual(first.json()['id'],repeated.json()['id'])
            self.assertEqual(self.store.summary()['queued'],1)
            response=self.client.post('/v1/images/generations',json=self.payload)
            self.assertEqual(response.status_code,200,response.text)
            self.assertTrue(Path(response.json()['data'][0]['file_path']).is_file())
            self.assertEqual(self.store.get(first.json()['id'])['status'],'queued')

    def test_local_storage_failure_is_distinct_from_queue_http_outage(self):
        with self.outage(),patch.object(self.store,'listing',side_effect=sqlite3.OperationalError('disk unavailable')):
            response=self.client.get('/v1/jobs')
            self.assertEqual(response.status_code,503)
            self.assertEqual(response.json()['code'],'queue_storage_unavailable')
            self.assertEqual(response.headers['Retry-After'],'2')
        with self.outage(),patch.object(self.store,'enqueue',side_effect=sqlite3.OperationalError('disk unavailable')):
            response=self.client.post('/v1/jobs',json=self.payload)
            self.assertEqual(response.status_code,503)
            self.assertEqual(response.json()['code'],'queue_storage_unavailable')


if __name__=='__main__':unittest.main()
