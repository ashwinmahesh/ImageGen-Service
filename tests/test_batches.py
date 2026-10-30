"""Sequential multi-image results, durable recovery and shared prompt expansion."""
import base64
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import server
from tests import test_api
from tests.test_queue import until
from tests.test_prompt_expansion import fake_expand


def fake_variations(prompt,*,count,**kwargs):
    return [{**fake_expand(prompt),'expanded_prompt':prompt+f'\nComposition option {index+1} with distinct framing.',
             'variation_index':index+1,'variation_count':count,'variation_goal':f'Goal {index+1}'} for index in range(count)]


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
            finally:release.set()
            until(lambda:self.store.get(admitted['id'])['status']=='succeeded')

    def test_varied_batch_plans_once_before_render_and_returns_each_applied_prompt(self):
        original=self.fake.generate_image
        def verify_saved(**kwargs):
            saved=self.store.listing()[0][0]['prompt_expansion']
            self.assertEqual(len(saved['candidates']),3)
            return original(**kwargs)
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand_batch',side_effect=fake_variations) as expand,patch.object(self.fake,'generate_image',side_effect=verify_saved):
            response=self.post(n=3,expand_prompt=True,vary_prompt=True)
        self.assertEqual(response.status_code,200,response.text);expand.assert_called_once()
        expected=fake_variations(self.payload['prompt'],count=3)
        self.assertEqual([call['prompt'] for call in self.fake.calls],[item['expanded_prompt'] for item in expected])
        for item,expansion in zip(response.json()['data'],expected):
            self.assertEqual(item['prompt_expansion'],expansion)
            metadata=json.loads(Path(item['metadata_path']).read_text())
            self.assertEqual(metadata['prompt'],self.payload['prompt'])
            self.assertEqual(metadata['effective_prompt'],expansion['expanded_prompt'])
        job=self.client.get('/v1/jobs/'+metadata['job_id']).json()
        self.assertNotIn('prompt_expansion',job)
        self.assertEqual(job['candidate_expansions'],expected)

    def test_varied_batch_dependencies_and_legacy_fingerprints(self):
        for values in ({'n':2},{'expand_prompt':True},{'n':2,'expand_prompt':True,'vary_prompt':'yes'}):
            self.assertEqual(self.post(**{'vary_prompt':True,**values}).status_code,422)
        self.assertEqual(self.post(n=2,vary_prompt=True,expand_prompt=True).status_code,422)  # Server LLM off.
        self.assertEqual(self.store.listing()[1],0)
        self.assertNotIn('vary_prompt',server.normalized_request(server.GenerateRequest(**self.payload)))

    def test_varied_restart_reuses_all_planned_prompts_and_completed_images(self):
        self.worker.stop();self.job_executor.shutdown(wait=True)
        payload=server.GenerateRequest(**self.payload,n=3,expand_prompt=True,vary_prompt=True)
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand_batch',side_effect=fake_variations) as expand:
            admitted=server.submit_job(payload);claimed=self.store.claim();real_save=self.store.save_progress
            def crash(identifier,result):
                real_save(identifier,result)
                raise KeyboardInterrupt('simulated shutdown after first saved image')
            with patch.object(self.store,'save_progress',side_effect=crash):
                with self.assertRaises(KeyboardInterrupt): server.execute_job(claimed)
            self.assertEqual(len(self.fake.calls),1)
            self.store.recover();claimed=self.store.claim()
            result=server.execute_job(claimed);self.store.finish(claimed['id'],result=result)
        expand.assert_called_once()
        self.assertEqual(len(self.fake.calls),3)
        self.assertEqual([call['prompt'] for call in self.fake.calls],[item['expanded_prompt'] for item in fake_variations(payload.prompt,count=3)])
        self.assertEqual(self.client.get('/v1/jobs/'+admitted['id']).json()['completed_images'],3)

    def test_duplicate_variation_prompts_fail_before_any_render(self):
        def repeated(prompt,*,count,**kwargs):
            values=fake_variations(prompt,count=count)
            for item in values: item['expanded_prompt']=values[0]['expanded_prompt']
            return values
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand_batch',side_effect=repeated):
            response=self.post(n=2,expand_prompt=True,vary_prompt=True)
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.fake.calls,[])

    def test_recovery_refuses_a_saved_image_with_a_different_composition(self):
        self.worker.stop();self.job_executor.shutdown(wait=True)
        payload=server.GenerateRequest(**self.payload,n=2,expand_prompt=True,vary_prompt=True)
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server.prompt_expander,'expand_batch',side_effect=fake_variations):
            admitted=server.submit_job(payload);claimed=self.store.claim();real_save=self.store.save_progress
            def crash(identifier,result):
                real_save(identifier,result)
                raise KeyboardInterrupt('interrupted after publishing first option')
            with patch.object(self.store,'save_progress',side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):server.execute_job(claimed)
            saved=self.store.get(admitted['id'])['result']['images'][0]
            sidecar=Path(saved['metadata_path']);data=json.loads(sidecar.read_text())
            data['prompt_expansion']['expanded_prompt']+=' Different prompt.';sidecar.write_text(json.dumps(data))
            self.store.recover();claimed=self.store.claim()
            with self.assertRaisesRegex(server.LocalFileError,'planned composition'):server.execute_job(claimed)
        self.assertEqual(len(self.fake.calls),1)


if __name__=='__main__':unittest.main()
