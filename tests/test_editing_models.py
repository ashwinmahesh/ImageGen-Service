"""Editing model provisioning, routing and native invocation without GPU inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import download
import launch
import mflux_engine
from model_assets import validate_snapshot
from model_profiles import PROFILES
from prompt_policy import public_guidance
import queue_api

ROOT=Path(__file__).resolve().parents[1]
EDITORS=('flux2-klein-4b','flux2-klein-9b','qwen-image-2.1-turbo')


def snapshot(folder, profile, *, completed=True):
    for relative in profile['required_files']:
        path=folder/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        if relative.endswith('.safetensors.index.json'):
            tensors={'layer.weight':'0.safetensors'}
            if path.parent.name=='text_encoder' and profile['base_model'].startswith('qwen'):
                tensors['visual.patch_embed.proj.weight']='0.safetensors'
            path.write_text(json.dumps({'metadata':{'quantization_level':str(profile['quantization'])},'weight_map':tensors}))
            (path.parent/'0.safetensors').write_bytes(b'fake')
        else:
            path.write_text('{}')
    if completed:
        (folder/'download.json').write_text(json.dumps({'repository':profile['repository'],'revision':profile['revision']}))


class EditingAssetsTest(unittest.TestCase):
    def test_menu_requires_complete_pinned_shards_and_vision_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary)/'project'; project.mkdir()
            store=Path(temporary)/'store'; store.mkdir()
            with patch.object(launch,'ROOT',project),patch.dict(os.environ,{'DATA_DIR':str(store)}):
                for identifier in EDITORS:
                    snapshot(store/identifier/'weights',PROFILES[identifier])
                self.assertEqual(launch.available_models(),list(EDITORS))
                weights=store/'qwen-image-2.1-turbo/weights'
                shard=weights/'text_encoder/0.safetensors'
                shard.unlink()
                self.assertNotIn('qwen-image-2.1-turbo',launch.available_models())
                shard.write_bytes(b'fake')
                index=weights/'text_encoder/model.safetensors.index.json'
                data=json.loads(index.read_text()); data['weight_map'].pop('visual.patch_embed.proj.weight')
                index.write_text(json.dumps(data))
                with self.assertRaisesRegex(SystemExit,'vision encoder'):
                    launch.check_assets('qwen-image-2.1-turbo')
                weights=store/'flux2-klein-9b/weights'
                (weights/'download.json').write_text('{"repository":"wrong","revision":"wrong"}')
                self.assertNotIn('flux2-klein-9b',launch.available_models())

    def test_downloader_validates_before_recording_completion(self):
        for identifier in EDITORS:
            with self.subTest(model=identifier),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary)/'weights'; profile=PROFILES[identifier]
                def fetch(**kwargs):
                    self.assertEqual(kwargs['revision'],profile['revision'])
                    self.assertEqual(kwargs['repo_id'],profile['repository'])
                    self.assertIn('processor/*',kwargs['allow_patterns'])
                    self.assertIn('model_index.json',kwargs['allow_patterns'])
                    snapshot(folder,profile,completed=False)
                with patch.object(download,'MODEL_DIR',folder),patch.object(download,'PROFILE',profile), \
                     patch.object(download,'MODEL_ID',profile['repository']),patch.object(download,'MODEL_REVISION',profile['revision']), \
                     patch('huggingface_hub.snapshot_download',side_effect=fetch),contextlib.redirect_stdout(io.StringIO()):
                    download.main()
                validate_snapshot(folder,profile,completed=True)
                (folder/'download.json').unlink()
                (folder/'transformer/0.safetensors').unlink()
                with patch.object(download,'MODEL_DIR',folder),patch.object(download,'PROFILE',profile), \
                     patch('huggingface_hub.snapshot_download'):
                    with self.assertRaisesRegex(ValueError,'missing shard'):
                        download.main()
                self.assertFalse((folder/'download.json').exists())


class EditingAdapterTest(unittest.TestCase):
    def test_text_and_edit_commands_use_local_checkpoints_and_retain_leases(self):
        for identifier in EDITORS:
            for reference in (None,'reference.png'):
                with self.subTest(model=identifier,reference=reference),tempfile.TemporaryDirectory() as temporary:
                    root=Path(temporary); profile=PROFILES[identifier]
                    selection={'file_path':'/tmp/adapter.safetensors','strength':.7,'trigger_prompt':'Illustration'}
                    def run(command,**kwargs):
                        module=profile['cli_module'] if reference else profile.get('text_cli_module',profile['cli_module'])
                        self.assertEqual(command[:3],[sys.executable,'-m',module])
                        self.assertEqual(command[command.index('--model')+1],str(root/'weights'))
                        self.assertEqual(command[command.index('--base-model')+1],identifier)
                        self.assertIn('--low-ram',command)
                        self.assertNotIn('--quantize',command)
                        self.assertEqual('--vae-tiling' in command,bool(profile.get('vae_tiling')))
                        self.assertEqual('--image-paths' in command,bool(reference))
                        self.assertEqual(command[command.index('--lora')+1:command.index('--lora')+3],['/tmp/adapter.safetensors','0.7'])
                        self.assertEqual(Path(command[command.index('--prompt-file')+1]).read_text(),'Illustration, Change the jacket to green.')
                        self.assertEqual(kwargs['pass_fds'],(91,92))
                        self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
                        self.assertEqual(kwargs['env']['TRANSFORMERS_OFFLINE'],'1')
                        Image.new('RGBA',(320,256),'green').save(command[command.index('--output')+1])
                        return subprocess.CompletedProcess(command,0)
                    with patch.object(mflux_engine,'ROOT',root),patch.object(mflux_engine,'MODEL_DIR',root/'weights'), \
                         patch.object(mflux_engine,'PROFILE',profile),patch.object(mflux_engine.subprocess,'run',side_effect=run):
                        settings=dict(prompt='Change the jacket to green.',width=320,height=256,seed=7,
                                      num_inference_steps=profile['steps'],image_path=reference,lora=selection,inherited_fds=(91,92))
                        path=root/'output.png'
                        mflux_engine.MfluxResult(settings).save(path)
                        with Image.open(path) as image:self.assertEqual(image.size,(320,256))
                        with self.assertRaises(FileExistsError):mflux_engine.MfluxResult(settings).save(path)

    def test_failure_or_wrong_dimensions_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            settings=dict(prompt='A fox',width=256,height=256,seed=7,num_inference_steps=4)
            def wrong(command,**kwargs):
                Image.new('RGB',(32,32),'red').save(command[command.index('--output')+1])
                return subprocess.CompletedProcess(command,0)
            with patch.object(mflux_engine,'ROOT',root),patch.object(mflux_engine,'PROFILE',PROFILES['flux2-klein-4b']):
                for runner,message in [(lambda *a,**k:subprocess.CompletedProcess(a[0],1),'exited 1'),(wrong,'Unexpected output size')]:
                    with patch.object(mflux_engine.subprocess,'run',side_effect=runner):
                        with self.assertRaisesRegex(RuntimeError,message):mflux_engine.MfluxResult(settings).save(root/'out.png')
                    self.assertFalse((root/'out.png').exists())
                with self.assertRaisesRegex(ValueError,'unsupported'):
                    mflux_engine.MfluxResult({**settings,'image_strength':.6}).save(root/'out.png')


class EditingContractTest(unittest.TestCase):
    def test_offline_queue_discovers_and_routes_all_editing_models(self):
        with tempfile.TemporaryDirectory() as temporary,TestClient(queue_api.create_app(Path(temporary))) as client:
            from tests.model_fixtures import setup_model
            for identifier in EDITORS:setup_model(Path(temporary),identifier)
            discovered={item['id']:item for item in client.get('/v1/models').json()['data']}
            for identifier in EDITORS:
                self.assertEqual(discovered[identifier]['reference_mode'],'edit')
                self.assertIn('license',discovered[identifier][identifier])
                response=client.post('/v1/jobs',json={'model':identifier,'request':{'prompt':'Change the jacket to green.'}})
                self.assertEqual(response.status_code,202,response.text)
                self.assertEqual(client.get('/v1/jobs?model='+identifier).json()['data'][0]['id'],response.json()['id'])
                self.assertIn(identifier,public_guidance(identifier)['id'])

    def test_actual_server_schemas_and_edit_geometry_without_gpu(self):
        script='''
import json
from pathlib import Path
from PIL import Image
import server
from tests.test_api import FakeModel
root=Path(server.ROOT); root.mkdir(parents=True)
reference=root/'source.png'; Image.new('RGB',(100,73),'red').save(reference)
server.model=FakeModel()
request=server.GenerateRequest(prompt='Change the jacket to green.',reference_image=str(reference),size='320x320',seed=1)
normalized=server.normalized_request(request)
assert 'image_strength' not in normalized
request=server.GenerateRequest.model_validate(normalized)
try: server.GenerateRequest(prompt='Edit',reference_image=str(reference),image_strength=.6)
except ValueError: pass
else: raise AssertionError('accepted ineffective strength')
path,metadata=server.generate(request)
assert server.model.calls[-1]['image_strength'] is None
assert server.model.calls[-1]['image_path']
assert Image.open(path).size==(320,234)
assert metadata[server.API_MODEL]['reference_mode']=='edit'
assert metadata['image_strength'] is None
cap=server.capabilities()
assert cap['reference_mode']=='edit' and not cap['image_strength_supported']
assert server.API_MODEL in server.GenerateRequest.model_json_schema()['properties']['model']['enum']
print(json.dumps({'model':server.API_MODEL,'steps':request.steps}))
'''
        with tempfile.TemporaryDirectory() as temporary:
            for identifier in EDITORS:
                with self.subTest(model=identifier):
                    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,
                        env={**os.environ,'DATA_DIR':temporary,'IMAGE_MODEL':identifier},capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(json.loads(result.stdout)['steps'],PROFILES[identifier]['steps'])


if __name__=='__main__':
    unittest.main()
