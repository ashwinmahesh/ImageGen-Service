"""Prompt expansion, persistence and mode changes with no model/GPU work."""
import json
from pathlib import Path
import sqlite3
import sys
import types
import unittest
from unittest.mock import patch,Mock
from contextlib import contextmanager
import prompt_expander
import server
import launch
from job_queue import JobStore, fingerprint
from tests import test_api
from tests import test_serving_mode


def fake_expand(prompt, **kwargs):
    return dict(original_prompt=prompt,expanded_prompt=prompt+'\nWarm rim light illuminates the soft mist.',
                model='test-local',revision='test-pinned',mode='expand',policy_version=1,seconds=.1,
                original_image_tokens=20,expanded_image_tokens=35,image_token_limit=512)


class ExpansionAPI(unittest.TestCase):
    setUp=test_api.APITest.setUp
    tearDown=test_api.APITest.tearDown
    post=test_api.APITest.post

    def test_off_by_default_rejected_before_queuing(self):
        self.assertFalse(self.client.get('/health').json()['prompt_expansion']['enabled'])
        response=self.post(expand_prompt=True)
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.store.listing()[1],0)
        self.assertEqual(self.client.post('/v1/prompts/expand',json={'prompt':'A fox'}).status_code,422)

    def test_original_and_applied_prompts_saved_and_returned(self):
        with patch.object(server.app.state,'llm_enabled',True), patch.object(prompt_expander,'expand',side_effect=fake_expand) as expander:
            response=self.post(expand_prompt=True)
            self.assertEqual(response.status_code,200,response.text)
            expansion=response.json()['data'][0]['prompt_expansion']
            self.assertEqual(expansion['original_prompt'],self.payload['prompt'])
            self.assertEqual(self.fake.calls[-1]['prompt'],expansion['expanded_prompt'])
            metadata=json.loads(Path(response.json()['data'][0]['metadata_path']).read_text())
            self.assertEqual(metadata['prompt'],self.payload['prompt'])
            self.assertEqual(metadata['expanded_prompt'],expansion['expanded_prompt'])
            self.assertEqual(metadata['effective_prompt'],expansion['expanded_prompt'])
            self.assertEqual(self.store.get(metadata['job_id'])['prompt_expansion'],expansion)
            expander.assert_called_once()

    def test_preview_exposes_safe_format_retry_provenance_without_creating_a_job(self):
        def repaired(prompt,**kwargs):return {**fake_expand(prompt),'structured_output_attempts':2,'structured_output_retry_reason':'invalid_json'}
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server,'GPU_LOCK_FILE',self.root/'gpu.lock'),patch.object(prompt_expander,'expand',side_effect=repaired):
            response=self.client.post('/v1/prompts/expand',json={'prompt':'A fox'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['structured_output_attempts'],2)
        self.assertEqual(response.json()['structured_output_retry_reason'],'invalid_json')
        self.assertEqual(self.store.listing()[1],0)

    def test_failed_expansion_does_not_render(self):
        with patch.object(server.app.state,'llm_enabled',True),patch.object(prompt_expander,'expand',side_effect=prompt_expander.ExpansionError('Too long; not compressed')):
            response=self.post(expand_prompt=True)
            self.assertEqual(response.status_code,422,response.text)
            self.assertEqual(self.fake.calls,[])
            self.assertEqual(self.store.listing()[0][0]['status'],'failed')

    def test_expansion_disabled_request_fingerprint_stays_compatible(self):
        a=server.normalized_request(server.GenerateRequest(**self.payload))
        b=server.normalized_request(server.GenerateRequest(**self.payload,expand_prompt=False))
        self.assertNotIn('expand_prompt',a)
        self.assertEqual(fingerprint(a),fingerprint(b))
        self.assertEqual(self.post(expand_prompt='yes').status_code,422)

    def test_preview_capacity_and_no_image_job(self):
        with patch.object(server.app.state,'llm_enabled',True),patch.object(server,'GPU_LOCK_FILE',self.root/'gpu.lock'),patch.object(prompt_expander,'expand',side_effect=fake_expand):
            response=self.client.post('/v1/prompts/expand',json={'prompt':'A fox'})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(self.store.listing()[1],0)
            server.busy.acquire()
            try:
                self.assertEqual(self.client.post('/v1/prompts/expand',json={'prompt':'A fox'}).status_code,409)
            finally:
                server.busy.release()
            self.assertFalse(server.preview_gate.locked())

    def test_restart_uses_saved_expansion_after_interrupted_render(self):
        self.worker.stop()  # Direct recovery fixture; no concurrent claimant.
        payload=server.GenerateRequest(**self.payload,expand_prompt=True)
        data=server.normalized_request(payload)
        job=self.store.enqueue(data,fingerprint(data))
        self.store.pause(False)
        job=self.store.claim()
        expansion=fake_expand(payload.prompt)
        self.store.save_expansion(job['id'],expansion)
        self.store.recover()
        job=self.store.claim()
        with patch.object(server.app.state,'llm_enabled',True),patch.object(prompt_expander,'expand',side_effect=AssertionError('Must reuse saved expansion')):
            metadata=server.execute_job(job)
        self.assertEqual(metadata['expanded_prompt'],expansion['expanded_prompt'])
        self.assertEqual(metadata['prompt'],payload.prompt)

    def test_setup_v1_schema_migration_and_feature_head_blocking(self):
        store=JobStore(self.root/'migration.sqlite3');store.initialize()
        payload={'prompt':'A fox','expand_prompt':True,'generate_immediately':False}
        job=store.enqueue(payload,fingerprint(payload),key='existing')
        with sqlite3.connect(store.path) as db:
            db.execute('ALTER TABLE jobs DROP COLUMN prompt_expansion')
            db.execute('PRAGMA user_version=1')
        store.initialize()
        self.assertEqual(store.existing('existing',fingerprint(payload))['id'],job['id'])
        self.assertTrue(store.summary()['next_requires_llm'])
        self.assertFalse(store.has_work(llm_enabled=False))
        self.assertIsNone(store.claim(llm_enabled=False))
        claimed=store.claim(llm_enabled=True)
        expansion=fake_expand('A fox');store.save_expansion(claimed['id'],expansion)
        store.recover()
        self.assertEqual(store.claim()['prompt_expansion'],expansion)


class ExpansionLauncher(unittest.TestCase):
    run_launcher=test_serving_mode.LauncherModeTest.run_launcher

    def test_flag_assets_mode_and_busy_reuse(self):
        with patch.object(launch,'check_enhancer_assets') as check:
            execute,_=self.run_launcher([],stub_enhancer=False)
            check.assert_called_once()
            self.assertNotIn('--no-llm',execute.call_args.args[1])
            state={'model':launch.API_MODEL,'busy':True,'loras_enabled':True,'prompt_expansion':{'enabled':True}}
            execute,kill=self.run_launcher([],state,stub_enhancer=False)
            execute.assert_not_called();kill.assert_not_called()
            with self.assertRaisesRegex(SystemExit,'processing'):
                self.run_launcher(['--no-llm'],state,stub_enhancer=False)
        with patch.object(launch,'check_enhancer_assets',side_effect=RuntimeError('make download-llm')):
            with self.assertRaisesRegex(SystemExit,'make download-llm'):
                self.run_launcher([],stub_enhancer=False)


class ExpansionContract(unittest.TestCase):
    @contextmanager
    def simulated_llm(self,outputs):
        messages=[];responses=iter(outputs)
        tokenizer=types.SimpleNamespace(apply_chat_template=lambda value,**kwargs:messages.append(json.loads(json.dumps(value))) or 'formatted')
        fake=types.ModuleType('mlx_lm');fake.load=Mock(return_value=('model',tokenizer))
        def generate(*args,**kwargs):
            text=next(responses)
            return iter([types.SimpleNamespace(text=text,finish_reason='stop',peak_memory=1.6)])
        fake.stream_generate=Mock(side_effect=generate)
        sampling=types.ModuleType('mlx_lm.sample_utils');sampling.make_sampler=lambda **kwargs:'greedy';sampling.make_logits_processors=lambda **kwargs:[]
        with patch.dict(sys.modules,{'mlx_lm':fake,'mlx_lm.sample_utils':sampling}),patch.object(prompt_expander,'image_token_count',return_value=40):
            yield fake,messages

    def test_malformed_generation_is_retried_once_without_reloading_or_rewriting_original(self):
        original='A fox in flat cel animation. No text.'
        valid=json.dumps({'style_details':'Clean ink contours enclose flat two-tone colors.','additional_composition':'Soft evening light separates the existing shapes.'})
        for invalid in ['plain prose', '{"style_details":"Flat colors."}',
                        '{"style_details":[],"additional_composition":""}',
                        '{"style_details":"","additional_composition":""}']:
            with self.subTest(invalid=invalid),self.simulated_llm([invalid,valid]) as (fake,messages):
                data=prompt_expander.run_worker({'prompt':original,'weights':'unused','image_weights':'unused','image_model':'z-image-turbo'})
                fake.load.assert_called_once();self.assertEqual(fake.stream_generate.call_count,2)
                self.assertEqual(data['structured_output_attempts'],2)
                self.assertIn(data['structured_output_retry_reason'],['invalid_json','invalid_fields','invalid_values','empty_details'])
                self.assertEqual(data['original_prompt'],original);self.assertTrue(data['expanded_prompt'].startswith(original+'\n'))
                self.assertEqual(data['expanded_prompt'].count(original),1)
                self.assertTrue(any(original in item['content'] for item in messages[1] if item['role']=='user'))
                self.assertIn('style_details',messages[1][-1]['content']);self.assertIn('same new-word allowance',messages[1][-1]['content'])
                self.assertEqual([call.kwargs['max_tokens'] for call in fake.stream_generate.call_args_list],[320,320])

    def test_repeated_bad_format_has_a_bounded_actionable_private_error(self):
        with self.simulated_llm(['private raw response','private raw response']) as (fake,_):
            with self.assertRaisesRegex(prompt_expander.ExpansionError,'one formatting retry') as result:
                prompt_expander.run_worker({'prompt':'A fox','weights':'unused','image_model':'z-image-turbo'})
        self.assertEqual(fake.stream_generate.call_count,2)
        self.assertIn('invalid JSON',str(result.exception));self.assertNotIn('private raw response',str(result.exception))

    def test_format_retry_applies_per_candidate_and_keeps_prior_compositions(self):
        wide='A wide view keeps the fox small against the snowy forest.'
        close='A close view lets the fox fill most of the frame.'
        valid=lambda composition:json.dumps({'style_details':'Translucent washes have soft pigment edges.','additional_composition':composition})
        with self.simulated_llm(['{}',valid(wide),valid(close)]) as (fake,messages):
            data=prompt_expander.run_worker({'prompt':'A watercolor fox in a snowy forest.','weights':'unused','image_model':'z-image-turbo','variation_count':2})
        fake.load.assert_called_once();self.assertEqual(fake.stream_generate.call_count,3)
        self.assertEqual([item['structured_output_attempts'] for item in data['candidates']],[2,1])
        self.assertEqual([item['variation_index'] for item in data['candidates']],[1,2]);self.assertIn(wide,messages[2][-1]['content'])

    def test_fidelity_errors_do_not_trigger_format_repair(self):
        invalid=json.dumps({'style_details':'Warm light illuminates their faces.','additional_composition':''})
        with self.simulated_llm([invalid]) as (fake,_):
            with self.assertRaisesRegex(prompt_expander.ExpansionError,'facial detail'):
                prompt_expander.run_worker({'prompt':'Two adults seen strictly from behind, flat cel cartoon.','weights':'unused','image_model':'z-image-turbo'})
        self.assertEqual(fake.stream_generate.call_count,1)

    def test_format_parser_accepts_complete_json_fences_and_rejects_ambiguous_fields(self):
        valid='{"style_details":"Flat color blocks.","additional_composition":""}'
        self.assertEqual(prompt_expander.parse_details('```JSON\n'+valid+'\n```','A fox')['style_details'],'Flat color blocks.')
        duplicate='{"style_details":"First.","style_details":"Second.","additional_composition":""}'
        with self.assertRaises(prompt_expander.GenerationFormatError) as result:prompt_expander.parse_details(duplicate,'A fox')
        self.assertEqual(result.exception.reason,'invalid_json')

    def test_varied_worker_reuses_one_loaded_llm_and_conditions_each_composition(self):
        original='Two adult women seen strictly from behind, a flat cel cartoon. No text.'
        messages=[]
        tokenizer=types.SimpleNamespace(apply_chat_template=lambda value,**kwargs:messages.append(value) or 'formatted')
        fake=types.ModuleType('mlx_lm');fake.load=lambda path:('model',tokenizer)
        outputs=iter(['A wide view keeps both rear silhouettes small against the surroundings.',
                      'A closer crop gives both rear silhouettes greater prominence without showing faces.'])
        def generate(*args,**kwargs):
            detail={'style_details':'Crisp ink outlines frame flat two-tone cel colors.','additional_composition':next(outputs)}
            return iter([types.SimpleNamespace(text=json.dumps(detail),finish_reason='stop',peak_memory=1.6)])
        fake.stream_generate=generate
        sampling=types.ModuleType('mlx_lm.sample_utils');sampling.make_sampler=lambda **kwargs:'greedy';sampling.make_logits_processors=lambda **kwargs:[]
        with patch.dict(sys.modules,{'mlx_lm':fake,'mlx_lm.sample_utils':sampling}),patch.object(prompt_expander,'image_token_count',return_value=40),patch.object(fake,'load',wraps=fake.load) as load:
            data=prompt_expander.run_worker({'prompt':original,'weights':'/unused','image_model':'z-image-turbo','variation_count':2})
        load.assert_called_once()
        candidates=data['candidates'];self.assertEqual(len(candidates),2)
        self.assertNotEqual(candidates[0]['expanded_prompt'],candidates[1]['expanded_prompt'])
        self.assertTrue(all(item['expanded_prompt'].startswith(original+'\n') for item in candidates))
        self.assertIn('wide environmental',messages[0][-1]['content'])
        self.assertIn('subject-dominant',messages[1][-1]['content'])
        self.assertIn(candidates[0]['details']['additional_composition'],messages[1][-1]['content'])
        self.assertIn('explicitly stated',messages[1][0]['content'])
        self.assertEqual([item['variation_index'] for item in candidates],[1,2])

    def test_worker_removes_exact_echo_and_releases_no_truncated_output(self):
        original='A fox in flat cel animation.'
        response=types.SimpleNamespace(text=json.dumps({'style_details':original+'\nBold ink contours frame broad flat color shapes.','additional_composition':''}),finish_reason='stop',peak_memory=1.6)
        tokenizer=types.SimpleNamespace(apply_chat_template=lambda *args,**kwargs:'formatted')
        fake=types.ModuleType('mlx_lm');fake.load=lambda path:('model',tokenizer)
        fake.stream_generate=lambda *args,**kwargs:iter([response])
        sampling=types.ModuleType('mlx_lm.sample_utils');sampling.make_sampler=lambda **kwargs:'greedy';sampling.make_logits_processors=lambda **kwargs:[]
        with patch.dict(sys.modules,{'mlx_lm':fake,'mlx_lm.sample_utils':sampling}),patch.object(prompt_expander,'image_token_count',side_effect=[30,50]):
            result=prompt_expander.run_worker({'prompt':original,'weights':'/unused','image_model':'z-image-turbo'})
        self.assertEqual(result['expanded_prompt'].count(original),1)
        self.assertEqual(result['peak_mlx_memory_gb'],1.6)
        response.finish_reason='length'
        with patch.dict(sys.modules,{'mlx_lm':fake,'mlx_lm.sample_utils':sampling}),patch.object(prompt_expander,'image_token_count',return_value=30):
            with self.assertRaisesRegex(prompt_expander.ExpansionError,'output limit'):
                prompt_expander.run_worker({'prompt':original,'weights':'/unused','image_model':'z-image-turbo'})
        with patch.object(prompt_expander,'image_token_count',return_value=500):
            with self.assertRaisesRegex(prompt_expander.ExpansionError,'not compressed'):
                prompt_expander.run_worker({'prompt':original,'weights':'/unused','image_model':'z-image-turbo'})

    def test_structured_details_and_narrow_regression_guards(self):
        details=prompt_expander.parse_details('{"style_details":"Flat color blocks.","additional_composition":""}','A flat cel fox')
        self.assertEqual(details['style_details'],'Flat color blocks.')
        for text in ['plain prose','{}','{"style_details":[],"additional_composition":""}','{"style_details":"","additional_composition":""}']:
            with self.assertRaises(prompt_expander.ExpansionError): prompt_expander.parse_details(text,'A fox')
        with self.assertRaisesRegex(prompt_expander.ExpansionError,'facial detail'):
            prompt_expander.check_added_details('Two adults seen strictly from behind.','Warm light illuminates their faces.')
        with self.assertRaisesRegex(prompt_expander.ExpansionError,'photographic detail'):
            prompt_expander.check_added_details('Flat cel colors.','Photorealistic fabric weave catches light.')
        prompt_expander.check_added_details('Strict rear view.','No facial features are visible.')

    def test_medium_routing_does_not_impose_cartoons_or_use_exclusions(self):
        from prompt_policy import style_context,style_anchor,example_for
        self.assertIn('photography',style_context('A natural-light photograph of a watch.'))
        self.assertNotIn('cel',style_context('A natural-light photograph of a watch.'))
        self.assertNotIn('cel',example_for('A natural-light photograph of a watch.')[1])
        self.assertIn('translucent',example_for('A watercolor painting.')[1].lower())
        self.assertEqual(style_anchor('A red fox beneath an oak tree. No text.'),'')
        self.assertIn('No rendering medium',style_context('A red fox beneath an oak tree.'))
        self.assertEqual(style_anchor('No photography. A flat vector logo.'),'vector')
        self.assertIn('different rendering treatments',style_context('2D cartoon figures in a 3D painted world.'))

    def test_each_backend_has_explicit_guidance_and_provenance(self):
        from model_profiles import PROFILES
        from prompt_policy import GUIDANCE,public_guidance
        self.assertEqual(set(PROFILES),set(GUIDANCE))
        self.assertEqual(public_guidance('z-image-turbo')['image_token_limit'],512)
        self.assertIsNone(public_guidance('chroma1-flash')['image_token_limit'])
        self.assertNotEqual(public_guidance('chroma1-hd')['id'],public_guidance('chroma1-flash')['id'])
        self.assertIn('not a reason',GUIDANCE['chroma1-flash']['instructions'])
        with self.assertRaises(KeyError): public_guidance('unregistered-model')

    def test_original_must_be_retained_and_limits_apply(self):
        original='A fox, flat cel shading. No text.'
        prompt_expander.validate_expansion(original,fake_expand(original))
        for value in ['A photorealistic fox', original+'\n', original+'\n'+('x'*4000)]:
            with self.assertRaises(ValueError):
                prompt_expander.validate_expansion(original,{'original_prompt':original,'expanded_prompt':value})
        self.assertIn('image-generation',prompt_expander.SYSTEM_PROMPT)
        self.assertIn('evocative',prompt_expander.SYSTEM_PROMPT)
        self.assertIn('expand the distinctive visual treatment',prompt_expander.SYSTEM_PROMPT)


if __name__=='__main__': unittest.main()
