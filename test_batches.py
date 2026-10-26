"""Sequential multi-image results, durable recovery and shared prompt expansion."""
import base64
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import server
import test_api
from test_queue import until
from test_prompt_expansion import fake_expand


class BatchAPI(unittest.TestCase):
    setUp=test_api.APITest.setUp
    tearDown=test_api.APITest.tearDown
    post=test_api.APITest.post

    def test_default_and_strict_count_bounds(self):
        self.assertEqual(len(self.post().json()['data']),1)
        for n in [0,6,-1,True,1.0,'2',None]:
            response=self.post(n=n)
            self.assertEqual(response.status_code,422,(n,response.text))

    def test_five_options_use_unique_seeds_paths_and_gallery_entries(self):
        response=self.post(n=5,seed=2**32-2,tags=['five-options'])
        self.assertEqual(response.status_code,200,response.text)
        items=response.json()['data']
        self.assertEqual(len(items),5)
        self.assertEqual([item['seed'] for item in items],[2**32-2,2**32-1,0,1,2])
        self.assertEqual(len({item['file_path'] for item in items}),5)
        self.assertEqual([call['prompt'] for call in self.fake.calls],['A fox']*5)
        self.assertEqual([item['candidate_index'] for item in items],[1,2,3,4,5])
        batch_ids=set()
        for item in items:
            self.assertEqual(item['candidate_count'],5)
            self.assertEqual(self.client.get(item['url']).status_code,200)
            meta=json.loads(Path(item['metadata_path']).read_text())
            batch_ids.add(meta['batch_id']);self.assertEqual(meta['job_id'],meta['batch_id'])
        self.assertEqual(len(batch_ids),1)
        self.assertEqual(self.client.get('/v1/images?q=five-options').json()['total'],5)
        job=self.client.get('/v1/jobs/'+batch_ids.pop()).json()
        self.assertEqual((job['completed_images'],job['total_images']),(5,5))
        self.assertEqual(len(job['result']['data']),5)

    def test_batch_base64_and_reference_ratio(self):
        from PIL import Image
        import io
        reference=self.root/'reference.png';Image.new('RGB',(450,300),'blue').save(reference)
        response=self.post(n=2,reference_image=str(reference),response_format='b64_json')
        self.assertEqual(response.status_code,200,response.text)
        for item in response.json()['data']:
            with Image.open(io.BytesIO(base64.b64decode(item['b64_json']))) as image:
                self.assertEqual(image.size,(256,171))
        self.assertEqual([call['seed'] for call in self.fake.calls],[42,43])

    def test_llm_expands_once_and_all_options_use_same_expansion(self):
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand',side_effect=fake_expand) as expand:
            response=self.post(n=3,expand_prompt=True)
        self.assertEqual(response.status_code,200,response.text);expand.assert_called_once()
        expanded=fake_expand(self.payload['prompt'])['expanded_prompt']
        self.assertEqual([call['prompt'] for call in self.fake.calls],[expanded]*3)
        self.assertTrue(all(item['prompt_expansion']['expanded_prompt']==expanded for item in response.json()['data']))

    def test_batch_idempotency_keeps_count_random_seed_and_one_row(self):
        self.store.pause(True)
        payload={**self.payload,'n':3,'seed':None}
        headers={'Idempotency-Key':'same-batch'}
        one=self.client.post('/v1/jobs',json=payload,headers=headers).json()
        two=self.client.post('/v1/jobs',json=payload,headers=headers).json()
        self.assertEqual(one['id'],two['id'])
        self.assertEqual(one['request']['seed'],two['request']['seed'])
        self.assertEqual(self.store.listing()[1],1)
        self.assertEqual(self.client.post('/v1/jobs',json={**payload,'n':4},headers=headers).status_code,409)
        self.assertEqual(self.client.delete('/v1/jobs/'+one['id']).json()['status'],'cancelled')
        self.assertEqual(self.fake.calls,[])

    def test_recovery_adopts_completed_options_without_rerendering_or_reexpanding(self):
        self.worker.stop();self.job_executor.shutdown(wait=True)
        payload=server.GenerateRequest(**self.payload,n=3,expand_prompt=True)
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand',side_effect=fake_expand) as expand:
            admitted=server.submit_job(payload);claimed=self.store.claim()
            real_save=self.store.save_progress
            def crash_after_two(identifier,result):
                real_save(identifier,result)
                if len(result['images'])==2: raise KeyboardInterrupt('simulated crash')
            with patch.object(self.store,'save_progress',side_effect=crash_after_two):
                with self.assertRaises(KeyboardInterrupt): server.execute_job(claimed)
            self.assertEqual(len(self.fake.calls),2)
            self.store.recover();claimed=self.store.claim()
            result=server.execute_job(claimed);self.store.finish(claimed['id'],result=result)
        expand.assert_called_once()
        self.assertEqual([call['seed'] for call in self.fake.calls],[42,43,44])
        job=self.client.get('/v1/jobs/'+admitted['id']).json()
        self.assertEqual(len(job['result']['data']),3)
        self.assertEqual(job['attempts'],2)

    def test_mid_batch_failure_retains_completed_option_and_later_jobs_continue(self):
        original=self.fake.generate_image
        def fail_second(**kwargs):
            if kwargs['seed']==43: raise RuntimeError('deliberate option failure')
            return original(**kwargs)
        with patch.object(self.fake,'generate_image',side_effect=fail_second):
            response=self.post(n=3)
        self.assertEqual(response.status_code,500,response.text)
        failed=self.store.listing()[0][0]
        job=self.client.get('/v1/jobs/'+failed['id']).json()
        self.assertEqual(job['status'],'failed')
        self.assertNotIn('result',job)
        self.assertEqual(job['completed_images'],1)
        self.assertEqual(len(job['partial_result']['data']),1)
        self.assertEqual(self.client.get(job['partial_result']['data'][0]['url']).status_code,200)
        self.assertEqual(self.post().status_code,200)

    def test_progress_visible_during_batch_without_pretending_complete(self):
        release=threading.Event();second_started=threading.Event();original=self.fake.generate_image
        def block_second(**kwargs):
            if kwargs['seed']==43: second_started.set();self.assertTrue(release.wait(4))
            return original(**kwargs)
        with patch.object(self.fake,'generate_image',side_effect=block_second):
            admitted=self.client.post('/v1/jobs',json={**self.payload,'n':2}).json()
            try:
                self.assertTrue(second_started.wait(2))
                job=self.client.get('/v1/jobs/'+admitted['id']).json()
                self.assertEqual(job['status'],'running')
                self.assertEqual((job['completed_images'],job['total_images']),(1,2))
                self.assertNotIn('result',job)
                self.assertEqual(len(job['partial_result']['data']),1)
                self.assertEqual(self.client.delete('/v1/jobs/'+admitted['id']).status_code,409)
            finally:release.set()
            until(lambda:self.store.get(admitted['id'])['status']=='succeeded')


if __name__=='__main__':unittest.main()
