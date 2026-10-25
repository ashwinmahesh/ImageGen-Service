"""Model selection, isolated data, shared catalogs and native adapters without GPU work."""
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
from PIL import Image
from fastapi.testclient import TestClient
import model_profiles
import config
import server
import chroma_engine
import download


class SelectionTest(unittest.TestCase):
    def test_menu_explicit_and_noninteractive(self):
        with patch.dict(os.environ,{},clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model_profiles.select_model(interactive=False),'z-image-turbo')
            self.assertEqual(model_profiles.select_model(interactive=True,prompt=lambda _: '3'),'chroma1-flash')
            self.assertEqual(model_profiles.select_model(interactive=True,prompt=lambda _: ''),'z-image-turbo')
            self.assertEqual(model_profiles.select_model('chroma1-hd',prompt=lambda _: self.fail('prompted')),'chroma1-hd')
            for answer in ['4','bad']:
                with self.assertRaises(ValueError):
                    model_profiles.select_model(interactive=True,prompt=lambda _: answer)

    def test_each_startup_model_has_own_schema_and_only_accepts_itself(self):
        script='''import server,json
request=server.GenerateRequest(prompt='A fox')
assert request.model==server.API_MODEL
assert request.steps==server.PROFILE['steps']
assert server.GenerateRequest(prompt='A fox',steps=server.PROFILE['max_steps'])
for change in [{'steps':server.PROFILE['max_steps']+1},{'model':next(x for x in server.PROFILES if x!=server.API_MODEL)}]:
    try: server.GenerateRequest(prompt='A fox',**change)
    except ValueError: pass
    else: raise AssertionError(change)
print(json.dumps({'model':server.API_MODEL,'steps':request.steps,'root':str(server.ROOT)}))
'''
        with tempfile.TemporaryDirectory() as folder:
            for identifier,profile in model_profiles.PROFILES.items():
                env={**os.environ,'IMAGE_MODEL':identifier,'DATA_DIR':folder}
                result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parent,env=env,capture_output=True,text=True,check=True)
                state=json.loads(result.stdout)
                self.assertEqual(state['steps'],profile['steps'])
                self.assertEqual(state['root'],str(Path(folder).resolve()/identifier))


class CrossModelGalleryTest(unittest.TestCase):
    def test_pagination_search_tags_and_download_work_across_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots={name:Path(temporary)/name for name in model_profiles.PROFILES}
            ids=[]
            for number,(name,root) in enumerate(roots.items(),1):
                images=root/'images'; images.mkdir(parents=True)
                identifier=f'{number:032x}'; ids.append(identifier)
                photo=images/(identifier+'.png'); Image.new('RGB',(32,24),'green').save(photo)
                photo.with_suffix('.json').write_text(json.dumps({'prompt':'A fox '+name,'model':name,'created':number,'tags':['Forest']}))
            with patch.object(server,'ROOT',roots[config.API_MODEL]),patch.object(server,'IMAGE_DIR',roots[config.API_MODEL]/'images'),patch.object(server,'MODEL_ROOTS',roots):
                client=TestClient(server.app,base_url='http://127.0.0.1')
                listing=client.get('/v1/images?limit=2&q=fox&tag=forest').json()
                self.assertEqual(listing['total'],3)
                self.assertEqual(listing['next_offset'],2)
                second=client.get('/v1/images?limit=2&offset=2').json()
                self.assertEqual(len(second['data']),1)
                self.assertEqual(second['previous_offset'],0)
                for identifier in ids:
                    response=client.get('/v1/images/'+identifier)
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(client.get(response.json()['url']).status_code,200)
                    self.assertEqual(client.get(response.json()['thumbnail_url']).status_code,200)
                    changed=client.patch('/v1/images/'+identifier+'/tags',json={'tags':['Campaign']})
                    self.assertEqual(changed.status_code,200)
                self.assertEqual(client.get('/v1/images?tag=campaign').json()['total'],3)
                self.assertEqual(client.get('/v1/images?q=chroma1-flash').json()['total'],1)
                client.close()


class NativeAdapterTest(unittest.TestCase):
    def test_both_samplers_reference_strength_and_output_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            reference=root/'ref.png'; Image.new('RGB',(256,256),'red').save(reference)
            def run(command,**kwargs):
                target=Path(command[command.index('--output')+1])
                Image.new('RGB',(256,256),'green').save(target)
                self.assertEqual(command[command.index('--strength')+1],str(1-.6))
                self.assertEqual(command[command.index('--cfg-scale')+1],profile['sampling'][1])
                self.assertEqual(command[command.index('--sampling-method')+1],profile['applied']['sampler'])
                return subprocess.CompletedProcess(command,0)
            for identifier in ['chroma1-hd','chroma1-flash']:
                profile=model_profiles.PROFILES[identifier]
                with patch.object(chroma_engine,'ROOT',root),patch.object(chroma_engine,'PROFILE',profile),patch.object(chroma_engine.subprocess,'run',side_effect=run):
                    output=root/(identifier+'.png')
                    result=chroma_engine.ChromaResult(root/'sd-cli',dict(prompt='A fox',seed=1,width=256,height=256,num_inference_steps=profile['steps'],image_path=str(reference),image_strength=.6))
                    result.save(output)
                    self.assertEqual(Image.open(output).size,(256,256))

    def test_flash_setup_fetches_shared_assets_without_hd_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            profile=model_profiles.PROFILES['chroma1-flash']
            own=[(profile['repository'],profile['revision'],profile['filename'],profile['sha256'])]
            fetched=[]
            def fetch(asset,directory=None):
                fetched.append((asset,directory)); return {'file':asset[2]}
            with patch.object(download,'MODEL_DIR',root/'flash/weights'),patch.object(download,'SHARED_ROOT',root/'hd'),patch.object(download,'PROFILE',profile),patch.object(download,'ASSETS',own),patch.object(download,'download_runtime'),patch.object(download,'download_asset',side_effect=fetch):
                download.main()
            self.assertEqual(len(fetched),3)
            self.assertEqual({x[0][2] for x in fetched},{profile['filename'],*(x[2] for x in model_profiles.SHARED_ASSETS)})
            self.assertTrue((root/'flash/weights/download.json').is_file())

class StartupSafetyTest(unittest.TestCase):
    def test_explicit_model_is_forwarded_and_busy_cross_model_switch_is_refused(self):
        from test_serving_mode import LauncherModeTest
        helper=LauncherModeTest()
        execute,_=helper.run_launcher(['--model','chroma1-flash'])
        command=execute.call_args.args[1]
        self.assertEqual(command[command.index('--model')+1],'chroma1-flash')
        with self.assertRaisesRegex(SystemExit,'generating'):
            helper.run_launcher(['--model','chroma1-hd'],{'model':'chroma1-flash','busy':True,'loras_enabled':True})

    def test_missing_selected_assets_fail_before_stopping_a_running_service(self):
        import launch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(launch,'ROOT',root),patch.dict(os.environ,{'DATA_DIR':str(root)}),patch.object(launch.os,'kill') as kill:
                with self.assertRaisesRegex(SystemExit,'make setup MODEL=chroma1-flash'):
                    launch.check_assets('chroma1-flash')
                kill.assert_not_called()
