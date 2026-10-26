"""Photo-edit task, scoped refinement and masks, with no model weights/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
import mflux_engine
import prompt_expander
from model_profiles import PROFILES
from photo_workflows import EDIT_SCOPE
from reference_geometry import prepare_geometry,prepare_mask,composite_edit,read_mask,ReferenceError

ROOT=Path(__file__).resolve().parents[1]


class EditRefinementTest(unittest.TestCase):
    def refine(self,details,**changes):
        messages=[]
        class Tokenizer:
            def apply_chat_template(self,value,**kwargs):
                messages.extend(value);return 'formatted edit'
        response=types.SimpleNamespace(text=json.dumps(details),finish_reason='stop',peak_memory=.1)
        lm=types.ModuleType('mlx_lm');lm.load=lambda _: (object(),Tokenizer());lm.stream_generate=lambda *a,**k:iter([response])
        samples=types.ModuleType('mlx_lm.sample_utils');samples.make_sampler=lambda **kwargs: None
        with patch.dict(sys.modules,{'mlx_lm':lm,'mlx_lm.sample_utils':samples}):
            data=prompt_expander.run_worker({'prompt':'Change the jacket to green.','weights':'unused',
                'image_model':'qwen-image-2.1-turbo','task':'edit','preserve':'face and background',**changes})
        return data,messages

    def test_edit_policy_has_no_scene_or_composition_expansion(self):
        data,messages=self.refine({'edit_details':'Change only the jacket color to green.','preservation':'Keep the face and background as they are.'})
        self.assertEqual(data['task'],'edit')
        self.assertEqual(data['guidance_profile'],'qwen-image-2.1-turbo-edit-v1')
        self.assertTrue(data['expanded_prompt'].startswith(data['original_prompt']+'\n'))
        self.assertIn('cannot see the photo',messages[0]['content'])
        self.assertIn('face and background',messages[-1]['content'])
        self.assertNotIn('Composition goal',str(messages))
        self.assertNotIn('style_details',str(messages))

    def test_generation_shaped_or_unrequested_camera_refinement_is_rejected(self):
        for details in [{'style_details':'Warm light','additional_composition':'Wide angle'},
                        {'edit_details':'Reframe into a wide shot.','preservation':'Keep the person.'}]:
            with self.subTest(details=details),self.assertRaises(prompt_expander.ExpansionError):
                self.refine(details)
        with self.assertRaisesRegex(prompt_expander.ExpansionError,'Composition variation'):
            self.refine({},variation_count=2)


class EditMaskGeometryTest(unittest.TestCase):
    def test_padding_alignment_and_composite_keep_unselected_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source.png'; mask=root/'mask.png'
            Image.new('RGB',(321,233),'red').save(source)
            image=Image.new('L',(321,233),0);ImageDraw.Draw(image).rectangle((70,40,140,100),fill=255);image.save(mask)
            with prepare_geometry(str(source),'512x512',root/'scratch') as geometry:
                with prepare_mask(str(mask),geometry,root/'scratch') as prepared:
                    prepared=Path(prepared)
                    with Image.open(prepared) as canvas:
                        self.assertEqual(canvas.size,geometry.model_size)
                        self.assertEqual(canvas.getpixel((0,0)),0)
                    output=root/'output.png';Image.new('RGBA',geometry.model_size,'green').save(output)
                    composite_edit(output,geometry.image_path,prepared)
                    geometry.finish(output)
                    with Image.open(output) as result:
                        self.assertEqual(result.size,geometry.output_size)
                        self.assertEqual(result.getpixel((0,0)),(255,0,0,255))
                        self.assertEqual(result.getpixel((160,90)),(0,128,0,255))
                self.assertFalse(prepared.exists())

    def test_invalid_empty_or_wrong_size_masks_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,size,color in [('empty',(100,100),0),('wrong',(99,100),255)]:
                path=root/(name+'.png');Image.new('L',size,color).save(path)
                with self.assertRaises(ReferenceError):read_mask(path,(100,100))

    def test_native_soft_mask_blend_is_not_applied_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'source.png';generated=root/'generated.png';mask=root/'mask.png'
            Image.new('RGB',(2,1),'red').save(source)
            Image.new('RGB',(2,1),(50,100,150)).save(generated)
            pixels=Image.new('L',(2,1));pixels.putpixel((1,0),128);pixels.save(mask)
            composite_edit(generated,source,mask)
            with Image.open(generated) as result:
                self.assertEqual(result.getpixel((0,0)),(255,0,0,255))
                self.assertEqual(result.getpixel((1,0)),(50,100,150,255))

    def test_mask_is_forwarded_to_qwen_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def run(command,**kwargs):
                self.assertEqual(command[command.index('--mask-image')+1],'/tmp/mask.png')
                Image.new('RGBA',(256,256),'green').save(command[command.index('--output')+1])
                return subprocess.CompletedProcess(command,0)
            with patch.object(mflux_engine,'ROOT',root),patch.object(mflux_engine,'PROFILE',PROFILES['qwen-image-2.1-turbo']),patch.object(mflux_engine.subprocess,'run',side_effect=run):
                mflux_engine.MfluxResult(dict(prompt='Edit',width=256,height=256,seed=1,num_inference_steps=8,image_path='/tmp/photo.png',mask_path='/tmp/mask.png')).save(root/'out.png')


class PhotoEditingContractTest(unittest.TestCase):
    def test_selected_area_on_both_async_apis_survives_recovery(self):
        script=r'''
import json
from pathlib import Path
from PIL import Image,ImageDraw
import server
from tests.test_api import APITest
fixture=APITest();fixture.setUp()
try:
    fixture.worker.stop();fixture.job_executor.shutdown(wait=True)
    source=fixture.root/'source.png';Image.new('RGB',(256,256),'red').save(source)
    mask=fixture.root/'mask.png';pixels=Image.new('L',(256,256))
    ImageDraw.Draw(pixels).rectangle((50,50,100,100),fill=255);pixels.save(mask)
    settings={**fixture.payload,'task':'edit','reference_image':str(source),'mask_image':str(mask),
              'prompt':'Change the jacket to green.','preserve':'Face and background','n':2}
    generator_schema=fixture.client.get('/openapi.json').json()
    assert 'mask_image' in generator_schema['components']['schemas']['GenerateRequest']['properties']
    queue_schema=fixture.queue_http.get('/openapi.json').json()
    request_schema=queue_schema['components']['schemas']['Submission']['properties']['request']
    assert any(example.get('mask_image') for example in request_schema['examples'])
    for origin in ('generator','queue'):
        client=fixture.client if origin=='generator' else fixture.queue_http
        body=settings if origin=='generator' else {'model':server.API_MODEL,'request':settings}
        headers={'Idempotency-Key':'area-'+origin}
        response=client.post('/v1/jobs',json=body,headers=headers)
        assert response.status_code==202,response.text
        job=response.json();identifier=job['id']
        assert job['request']['task']=='edit' and job['request']['mask_image']==str(mask)
        replay=client.post('/v1/jobs',json=body,headers=headers)
        assert replay.status_code==202 and replay.json()['id']==identifier,replay.text
        changed={**settings,'mask_image':str(fixture.root/'different-mask.png')}
        changed_body=changed if origin=='generator' else {'model':server.API_MODEL,'request':changed}
        assert client.post('/v1/jobs',json=changed_body,headers=headers).status_code==409
        claimed=fixture.store.claim();assert claimed['id']==identifier
        result=server.execute_job(claimed)
        frozen=fixture.store.get(identifier)['request']
        assert frozen['mask_image']==str(mask) and frozen['preserve']=='Face and background'
        for image in result['images']:
            assert image['task']=='edit' and image['edit_scope']=='mask'
            assert image['mask_image']==str(mask)
            with Image.open(image['file_path']) as pixels:
                assert pixels.getpixel((0,0))==(255,0,0,255)
                assert pixels.getpixel((75,75))==(0,128,0,255)
        # Simulate a crash after artifacts/checkpoints, before success was recorded.
        fixture.store.recover();fixture.fake.calls.clear()
        restored=fixture.store.claim();recovered=server.execute_job(restored)
        assert recovered==result and fixture.fake.calls==[]
        assert fixture.store.get(identifier)['request']==frozen
        fixture.store.finish(identifier,result=recovered)
        queue_result=fixture.queue_http.get('/v1/jobs/'+identifier,params={'model':server.API_MODEL}).json()
        generator_result=fixture.client.get('/v1/jobs/'+identifier).json()
        assert queue_result['status']==generator_result['status']=='succeeded'
        assert queue_result['result']['images'][0]['mask_image']==str(mask)
        assert generator_result['result']['data'][0]['task']=='edit'
    # Queue admission stays available without reading source/mask files.
    missing={**settings,'mask_image':str(fixture.root/'missing.png')}
    assert fixture.client.post('/v1/jobs',json=missing).status_code==400
    accepted=fixture.queue_http.post('/v1/jobs',json={'model':server.API_MODEL,'request':missing})
    assert accepted.status_code==202,accepted.text
finally:fixture.tearDown()
'''
        with tempfile.TemporaryDirectory() as temporary:
            result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,
                env={**os.environ,'IMAGE_MODEL':'qwen-image-2.1-turbo','DATA_DIR':temporary},
                capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)

    def test_real_http_contracts_generation_editing_masks_and_refinement_without_gpu(self):
        script=r'''
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image,ImageDraw
import server,prompt_expander
from tests.test_api import APITest
from photo_workflows import EDIT_SCOPE
fixture=APITest();fixture.setUp()
try:
    source=fixture.root/'source.png';Image.new('RGB',(256,256),'red').save(source)
    mask=fixture.root/'mask.png';pixels=Image.new('L',(256,256));ImageDraw.Draw(pixels).rectangle((50,50,100,100),fill=255);pixels.save(mask)
    base={**fixture.payload,'reference_image':str(source),'prompt':'Change the jacket to green.','preserve':'face and background'}
    response=fixture.client.post('/v1/images/edits',json=base)
    assert response.status_code==200,response.text
    item=response.json()['data'][0];assert item['task']=='edit'
    assert EDIT_SCOPE in fixture.fake.calls[-1]['prompt']
    assert 'face and background' in fixture.fake.calls[-1]['prompt']
    assert fixture.fake.calls[-1]['image_strength'] is None
    metadata=json.loads(Path(item['metadata_path']).read_text());assert metadata['task']=='edit' and metadata['edit_scope']=='whole_photo'
    assert fixture.client.post('/v1/images/edits',json=fixture.payload).status_code==422
    assert fixture.client.post('/v1/jobs',json={**base,'task':'edit','expand_prompt':True,'vary_prompt':True,'n':2}).status_code==422
    assert fixture.client.post('/v1/jobs',json={**base,'task':'generate'}).status_code==422
    assert fixture.client.post('/v1/images/edits',json={**base,'image_strength':.6}).status_code==422
    # Both editing families still generate from text without a reference.
    response=fixture.post(prompt='A landscape photograph')
    assert response.status_code==200,response.text
    assert fixture.fake.calls[-1]['image_path'] is None
    assert fixture.fake.calls[-1]['prompt']=='A landscape photograph'
    if server.API_MODEL=='qwen-image-2.1-turbo':
        response=fixture.client.post('/v1/images/edits',json={**base,'mask_image':str(mask)})
        assert response.status_code==200,response.text
        with Image.open(response.json()['data'][0]['file_path']) as image:
            assert image.getpixel((0,0))==(255,0,0,255)
            assert image.getpixel((75,75))==(0,128,0,255)
        assert fixture.fake.calls[-1]['mask_path']
        assert not Path(fixture.fake.calls[-1]['mask_path']).exists()
    else:
        assert fixture.client.post('/v1/images/edits',json={**base,'mask_image':str(mask)}).status_code==422
    def refine(prompt,**kwargs):
        assert kwargs['task']=='edit' and kwargs['preserve']=='face and background'
        return dict(original_prompt=prompt,expanded_prompt=prompt+'\nKeep everything else unchanged.',task='edit',model='fake',revision='pinned',mode='expand',policy_version=1,seconds=.1)
    with patch.object(server.app.state,'llm_enabled',True),patch.object(prompt_expander,'expand',side_effect=refine) as refiner:
        fixture.worker.llm_enabled=lambda:True
        response=fixture.client.post('/v1/images/edits',json={**base,'expand_prompt':True})
        assert response.status_code==200,response.text
        assert response.json()['data'][0]['prompt_expansion']['task']=='edit'
        metadata=json.loads(Path(response.json()['data'][0]['metadata_path']).read_text())
        assert metadata[server.API_MODEL]['prompt_expansion_guidance']['id'].endswith('-edit-v1')
        refiner.assert_called_once()
    schema=fixture.client.get('/openapi.json').json()
    assert '/v1/images/edits' in schema['paths']
    assert 'reference_image' in schema['components']['schemas']['EditRequest']['required']
    print('OK '+server.API_MODEL)
finally:fixture.tearDown()
'''
        with tempfile.TemporaryDirectory() as temporary:
            for identifier in ('flux2-klein-4b','flux2-klein-9b','qwen-image-2.1-turbo'):
                with self.subTest(model=identifier):
                    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,env={**os.environ,'IMAGE_MODEL':identifier,'DATA_DIR':temporary},capture_output=True,text=True,timeout=45)
                    self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
