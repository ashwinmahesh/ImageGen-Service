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
import launch
import setup as model_setup


class SelectionTest(unittest.TestCase):
    def test_menu_explicit_and_noninteractive(self):
        with patch.dict(os.environ,{},clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model_profiles.select_model(interactive=False),'z-image-turbo')
            self.assertEqual(model_profiles.select_model(interactive=True,prompt=lambda _: '3'),'chroma1-flash')
            self.assertEqual(model_profiles.select_model(interactive=True,prompt=lambda _: ''),'z-image-turbo')
            self.assertEqual(model_profiles.select_model('chroma1-hd',prompt=lambda _: self.fail('prompted')),'chroma1-hd')
            for answer in [str(len(model_profiles.PROFILES)+1),'bad']:
                with self.assertRaises(ValueError):
                    model_profiles.select_model(interactive=True,prompt=lambda _: answer)

    def test_filtered_menu_numbers_and_ready_default(self):
        with patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(io.StringIO()) as output:
            ready=['chroma1-hd','chroma1-flash']
            self.assertEqual(model_profiles.select_model(choices=ready,interactive=True,prompt=lambda _: '2'),'chroma1-flash')
            self.assertEqual(model_profiles.select_model(choices=ready,interactive=True,prompt=lambda _: ''),'chroma1-hd')
            self.assertEqual(model_profiles.select_model(choices=['chroma1-flash'],interactive=False),'chroma1-flash')
            self.assertEqual(model_profiles.select_model(choices=list(model_profiles.PROFILES),interactive=False),'z-image-turbo')
            self.assertIn('1. Chroma1-HD',output.getvalue())
            self.assertIn('2. Chroma1-Flash',output.getvalue())
            self.assertIn('- Z-Image Turbo (z-image-turbo) — make setup MODEL=z-image-turbo',output.getvalue())
            self.assertNotIn('1. Z-Image Turbo',output.getvalue())
            with self.assertRaisesRegex(ValueError,'not set up'):
                model_profiles.select_model(choices=ready,interactive=True,prompt=lambda _: 'z-image-turbo')
            for options in [dict(choices=[]),dict(explicit='z-image-turbo',choices=ready)]:
                with self.assertRaisesRegex(ValueError,'make setup'):
                    model_profiles.select_model(interactive=False,**options)

    def test_empty_serving_menu_shows_setup_commands_without_prompting(self):
        with patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(ValueError,'No models are set up'):
                model_profiles.select_model(choices=[],interactive=True,prompt=lambda _:self.fail('prompted with no ready models'))
            for name in model_profiles.PROFILES:
                self.assertIn('make setup MODEL='+name,output.getvalue())
            self.assertNotIn('  1.',output.getvalue())

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
                result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,check=True)
                state=json.loads(result.stdout)
                self.assertEqual(state['steps'],profile['steps'])
                self.assertEqual(state['root'],str(Path(folder).resolve()/identifier))


class InstalledModelMenuTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.project=self.root/'project';self.project.mkdir()
        self.store=self.root/'store';self.store.mkdir()
        self.env=patch.dict(os.environ,{'DATA_DIR':str(self.store)},clear=True)
        self.env.start()
        self.location=patch.object(launch,'ROOT',self.project);self.location.start()

    def tearDown(self):
        self.location.stop();self.env.stop();self.temp.cleanup()

    def z_image(self):
        weights=self.store/'z-image-turbo/weights';weights.mkdir(parents=True)
        (weights/'download.json').write_text('{}')
        for name in ['transformer','text_encoder','tokenizer','vae']:(weights/name).mkdir()
        return weights

    def flash(self):
        weights=self.store/'chroma1-flash/weights';weights.mkdir(parents=True)
        (weights/'download.json').write_text('{}')
        (weights/model_profiles.PROFILES['chroma1-flash']['filename']).touch()
        runtime=self.project/'runtime';runtime.mkdir()
        (runtime/'sd-cli').touch()
        (runtime/'libstable-diffusion.dylib').touch()
        for asset in model_profiles.SHARED_ASSETS:
            path=self.store/'chroma1-hd/weights'/asset[2]
            path.parent.mkdir(parents=True,exist_ok=True);path.touch()

    def setup_menu(self, answer=''):
        select=lambda **kwargs:model_profiles.select_model(interactive=True,prompt=lambda _:answer,**kwargs)
        with patch.object(model_setup,'ROOT',self.project),patch.object(model_setup,'select_model',side_effect=select), \
             patch.object(model_setup.subprocess,'Popen') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            run.return_value.wait.return_value=0
            self.assertEqual(model_setup.main(),0)
        return output.getvalue(),run.call_args.kwargs['env']['IMAGE_MODEL']

    def test_setup_menu_labels_ready_assets_and_keeps_every_model_selectable(self):
        self.z_image();self.flash()
        output,selected=self.setup_menu('2')
        self.assertEqual(selected,'chroma1-hd')
        self.assertIn('1. Z-Image Turbo [SETUP COMPLETE] (z-image-turbo)',output)
        self.assertIn('2. Chroma1-HD (chroma1-hd)',output)
        self.assertIn('3. Chroma1-Flash [SETUP COMPLETE] (chroma1-flash)',output)
        self.assertEqual(output.count('[SETUP COMPLETE]'),2)
        for index,(identifier,profile) in enumerate(model_profiles.PROFILES.items(),1):
            self.assertIn(f"{index}. {profile['name']}",output)
        library=self.project/'runtime/libstable-diffusion.dylib';library.unlink()
        output,selected=self.setup_menu('3')
        self.assertEqual(selected,'chroma1-flash')
        self.assertEqual(output.count('[SETUP COMPLETE]'),1)
        library.touch()
        # Shared files are checked each time, rather than trusting download.json.
        (self.store/'chroma1-hd/weights'/model_profiles.SHARED_ASSETS[0][2]).unlink()
        output,selected=self.setup_menu()
        self.assertEqual(selected,'z-image-turbo')
        self.assertEqual(output.count('[SETUP COMPLETE]'),1)
        self.assertNotIn('Chroma1-Flash [SETUP COMPLETE]',output)

    def test_setup_menu_missing_mount_has_no_completion_labels_or_writes(self):
        with patch.dict(os.environ,{'DATA_DIR':'/Volumes/setup-label-test-unmounted/models'}), \
             patch('storage_paths.Path.is_mount',return_value=False),patch.object(Path,'mkdir') as mkdir:
            output,selected=self.setup_menu()
            self.assertEqual(selected,'z-image-turbo')
            self.assertNotIn('[SETUP COMPLETE]',output)
            mkdir.assert_not_called()

    def test_setup_skips_readiness_scan_for_explicit_and_noninteractive_selection(self):
        for environment in ({'MODEL':'chroma1-flash'}, {'NONINTERACTIVE':'1'}):
            with patch.dict(os.environ,environment,clear=True),patch.object(model_setup,'completed_models') as scan, \
                 patch.object(model_setup.subprocess,'Popen') as run,contextlib.redirect_stdout(io.StringIO()) as output:
                run.return_value.wait.return_value=0
                self.assertEqual(model_setup.main(),0)
                scan.assert_not_called()
                self.assertNotIn('Choose an image model:',output.getvalue())

    def test_discovery_checks_shared_assets_and_does_not_require_hd_checkpoint_for_flash(self):
        weights=self.z_image();self.flash()
        self.assertEqual(launch.available_models(),['z-image-turbo','chroma1-flash'])
        (weights/'download.json').unlink()
        self.assertEqual(launch.available_models(),['chroma1-flash'])
        shared=self.store/'chroma1-hd/weights'/model_profiles.SHARED_ASSETS[0][2]
        shared.unlink()
        self.assertEqual(launch.available_models(),[])

    def test_completion_marker_alone_or_wrong_asset_type_is_not_ready(self):
        weights=self.z_image()
        (weights/'vae').rmdir();(weights/'vae').touch()
        self.assertEqual(launch.available_models(),[])
        output,_=self.setup_menu()
        self.assertNotIn('[SETUP COMPLETE]',output)
        (weights/'vae').unlink();(weights/'vae').mkdir()
        (weights/'download.json').unlink();(weights/'download.json').mkdir()
        self.assertEqual(launch.available_models(),[])

    def test_no_setup_or_explicit_missing_model_leaves_existing_services_untouched(self):
        for args,message in [([], 'No models are set up'),(['--model','chroma1-flash'],'make setup MODEL=chroma1-flash')]:
            with patch.object(sys,'argv',['launch.py',*args]),patch.object(launch,'port_free') as port,patch.object(launch.os,'kill') as kill,patch.object(launch.os,'execv') as execute,contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:launch.main()
                if args:self.assertIn(message,str(stopped.exception))
                else:
                    self.assertEqual(stopped.exception.code,2)
                    self.assertIn(message,errors.getvalue())
                port.assert_not_called();kill.assert_not_called();execute.assert_not_called()

    def test_missing_storage_mount_fails_without_creating_a_fallback(self):
        with patch.dict(os.environ,{'DATA_DIR':'/Volumes/model-menu-test-unmounted/models'}),patch('storage_paths.Path.is_mount',return_value=False),patch.object(Path,'mkdir') as mkdir:
            with self.assertRaisesRegex(RuntimeError,'not mounted'):launch.available_models()
            mkdir.assert_not_called()


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
                self.assertEqual(listing['total'],len(roots))
                self.assertEqual(listing['next_offset'],2)
                second=client.get('/v1/images?limit=2&offset=2').json()
                self.assertEqual(len(second['data']),min(2,len(roots)-2))
                self.assertEqual(second['previous_offset'],0)
                for identifier in ids:
                    response=client.get('/v1/images/'+identifier)
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(client.get(response.json()['url']).status_code,200)
                    self.assertEqual(client.get(response.json()['thumbnail_url']).status_code,200)
                    changed=client.patch('/v1/images/'+identifier+'/tags',json={'tags':['Campaign']})
                    self.assertEqual(changed.status_code,200)
                self.assertEqual(client.get('/v1/images?tag=campaign').json()['total'],len(roots))
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
        from tests.test_serving_mode import LauncherModeTest
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
