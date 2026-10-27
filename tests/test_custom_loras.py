"""User LoRA setup/discovery and clean Git updates; no network or inference."""
from contextlib import redirect_stdout,redirect_stderr
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,Mock
import numpy as np
from safetensors.numpy import save_file
from fastapi.testclient import TestClient
import adapter_catalog
import lora_setup
import queue_api
from tests.model_fixtures import setup_model

PROJECT=Path(__file__).resolve().parents[1]


def weights(path,family='z-image-turbo',value=0):
    prefix='diffusion_model.layers.0.attention.to_k' if family=='z-image-turbo' else 'diffusion_model.double_blocks.0.img_attn.qkv'
    save_file({prefix+'.lora_A.weight':np.full((1,1),value,dtype=np.float32),prefix+'.lora_B.weight':np.zeros((1,1),np.float32)},str(path),metadata={'ss_base_model_version':'zimage' if family=='z-image-turbo' else 'chroma'})
    return path


class CustomLoRATest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.project=self.root/'project';self.project.mkdir();self.store=self.root/'SSD/models'
        self.environment=patch.dict(os.environ,{'DATA_DIR':str(self.store)},clear=True);self.environment.start()
        self.source=weights(self.root/'Ink Style.safetensors')
        self.public=[{'id':'realism','name':'Realism','filename':'realism.safetensors','models':['z-image-turbo'],'default_strength':.8,'source':'https://example.com/public','trigger_prompt':'','sha256':'a'*64}]
        (self.project/'lora-catalog.json').write_text(json.dumps(self.public))

    def tearDown(self):self.environment.stop();self.temp.cleanup()

    def install(self,**options):
        values={'identifier':'ink','name':'Ink style','trigger':'inkstyle','strength':.7};values.update(options)
        return lora_setup.register(self.project,'z-image-turbo',self.source,**values)

    def run_cli(self,args,**options):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):return lora_setup.main(args,project=self.project,**options)

    def test_local_install_is_separate_repeatable_checked_and_source_preserving(self):
        original=self.source.read_bytes();public=(self.project/'lora-catalog.json').read_bytes()
        entry,directory,changed=self.install();self.assertTrue(changed)
        self.assertEqual(directory,self.store/'z-image-turbo/loras');self.assertEqual(entry['id'],'custom-ink')
        self.assertEqual(self.source.read_bytes(),original);self.assertEqual((directory/entry['filename']).read_bytes(),original)
        self.assertEqual(entry['sha256'],hashlib.sha256(original).hexdigest())
        self.assertFalse((directory/'catalog.json').exists());self.assertEqual((self.project/'lora-catalog.json').read_bytes(),public)
        before=(directory/'custom.json').read_bytes();self.assertFalse(self.install()[2]);self.assertEqual((directory/'custom.json').read_bytes(),before)
        self.assertEqual([item['id'] for item in adapter_catalog.read_catalog(self.project,directory,'z-image-turbo')],['realism','custom-ink'])
        self.assertEqual([item['id'] for item in adapter_catalog.read_catalog(self.project,directory,'z-image-turbo',local=False)],['realism'])

    def test_custom_runtime_selection_and_legacy_import_do_not_duplicate_user_entries(self):
        import install_lora,lora_catalog
        entry,directory,_=self.install();before=(directory/'custom.json').read_bytes()
        with patch.object(lora_catalog,'ROOT',self.project),patch.object(lora_catalog,'LORA_DIR',directory),patch.object(install_lora,'LORA_DIR',directory):
            selected=lora_catalog.resolve_lora(entry['id'])
            self.assertEqual(selected['strength'],.7);self.assertEqual(selected['file_path'],str(directory/entry['filename']))
            self.assertEqual(lora_catalog.effective_prompt('A village',selected),'inkstyle, A village')
            self.assertEqual(lora_catalog.effective_prompt('inkstyle, A village',selected),'inkstyle, A village')
            self.assertEqual(lora_catalog.resolve_lora(entry['id'],1.2)['strength'],1.2)
            with redirect_stdout(io.StringIO()):install_lora.install(self.source,'realism')
            self.assertEqual([item['id'] for item in json.loads((directory/'catalog.json').read_text())],['realism'])
            self.assertEqual([item['id'] for item in lora_catalog.catalog()],['realism','custom-ink'])
        self.assertEqual((directory/'custom.json').read_bytes(),before)

    def test_reusing_id_never_replaces_different_weights_or_settings(self):
        first,directory,_=self.install();before=(directory/'custom.json').read_bytes();original=(directory/first['filename']).read_bytes()
        with self.assertRaisesRegex(ValueError,'different weights/settings'):self.install(trigger='different')
        weights(self.source,value=1)
        with self.assertRaisesRegex(ValueError,'different weights/settings'):self.install()
        self.assertEqual((directory/'custom.json').read_bytes(),before);self.assertEqual((directory/first['filename']).read_bytes(),original)
        self.assertFalse(list(directory.glob('.install-*')))

    def test_preserves_legacy_presets_and_uses_separate_custom_catalog(self):
        directory=self.store/'z-image-turbo/loras';directory.mkdir(parents=True)
        legacy=[{**self.public[0],'id':'legacy-style'}];raw=json.dumps(legacy).encode();(directory/'catalog.json').write_bytes(raw)
        self.install();self.assertEqual((directory/'catalog.json').read_bytes(),raw)
        self.assertEqual([item['id'] for item in adapter_catalog.read_catalog(self.project,directory,'z-image-turbo')],['legacy-style','custom-ink'])
        (directory/'custom.json').write_text('broken private catalog')
        self.assertEqual(adapter_catalog.read_catalog(self.project,directory,'z-image-turbo',local=False),self.public)

    def test_invalid_download_wrong_family_checksum_and_symlink_leave_catalog_unchanged(self):
        _,directory,_=self.install();before=(directory/'custom.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):self.install(identifier='wrong-digest',expected_sha256='0'*64)
        weights(self.source,'chroma1-hd')
        with self.assertRaisesRegex(ValueError,'model family'):self.install(identifier='wrong-family')
        self.source.write_text('<html>not weights</html>')
        with self.assertRaisesRegex(ValueError,'Invalid or incomplete'):self.install(identifier='invalid')
        self.assertEqual((directory/'custom.json').read_bytes(),before)
        entry=json.loads(before)[0];target=directory/entry['filename'];target.unlink();target.symlink_to(self.source)
        weights(self.source)
        with self.assertRaisesRegex(ValueError,'regular file'):self.install()
        self.assertEqual((directory/'custom.json').read_bytes(),before)

    def test_missing_existing_weight_can_be_repaired_without_mutating_source(self):
        entry,directory,_=self.install();(directory/entry['filename']).unlink()
        self.assertTrue(self.install()[2]);self.assertEqual((directory/entry['filename']).read_bytes(),self.source.read_bytes())

    def test_explicit_chroma_compatibility_shares_weights_and_keeps_model_filters(self):
        source=weights(self.root/'Chroma.safetensors','chroma1-hd')
        first,directory,_=lora_setup.register(self.project,'chroma1-hd',source,identifier='ink',name='Chroma Ink')
        second,other,_=lora_setup.register(self.project,'chroma1-flash',source,identifier='ink',name='Chroma Ink')
        self.assertEqual(directory,other);self.assertEqual(first['filename'],second['filename']);self.assertEqual(second['models'],['chroma1-hd','chroma1-flash'])
        self.assertEqual(len(list(directory.glob('*.safetensors'))),1)
        self.assertEqual(len(adapter_catalog.read_catalog(self.project,directory,'chroma1-flash')),1)
        self.assertEqual(adapter_catalog.read_catalog(self.project,directory,'qwen-image-2.1-turbo'),[])

    def test_input_validation_and_unvalidated_models_fail_before_writing(self):
        for options in [{'identifier':'../escape'},{'name':'Bad\nName'},{'strength':'nan'},{'strength':2.1},{'expected_sha256':'short'},{'provenance':{'id':'realism'}}]:
            with self.subTest(options=options),self.assertRaises(ValueError):self.install(**options)
        with self.assertRaisesRegex(ValueError,'Unsupported'):lora_setup.register(self.project,'qwen-image-2.1-turbo',self.source,identifier='ink',name='Ink')
        self.assertFalse(self.store.exists())
        for value in ['https://example.com/model.safetensors','https://huggingface.co/org/repo/blob/main/../bad.safetensors','https://huggingface.co/org/repo/blob/main/bad.safetensors?token=secret']:
            with self.subTest(value=value),self.assertRaises(ValueError):lora_setup.parse_source(value)

    def test_interactive_cancel_eof_and_interrupted_copy_are_graceful(self):
        for effect in [['q'],EOFError,KeyboardInterrupt]:
            with patch('sys.stdin.isatty',return_value=True),patch('builtins.input',side_effect=effect):self.assertEqual(self.run_cli([]),130)
        self.assertFalse(self.store.exists())
        with patch('sys.stdin.isatty',return_value=False),patch.object(lora_setup.shutil,'copyfile',side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_cli(['--source',str(self.source),'--id','ink']),130)
        directory=self.store/'z-image-turbo/loras';self.assertFalse((directory/'custom.json').exists());self.assertFalse(list(directory.glob('.install-*')))
        self.assertTrue(self.source.is_file())

    def test_noninteractive_cli_registers_without_prompts_and_does_not_import_inference(self):
        with patch('builtins.input',side_effect=AssertionError('Must not prompt')):
            self.assertEqual(self.run_cli(['--noninteractive','--source',str(self.source),'--id','ink','--trigger','inkstyle']),0)
        entry=json.loads((self.store/'z-image-turbo/loras/custom.json').read_text())[0];self.assertEqual(entry['id'],'custom-ink')
        result=subprocess.run([sys.executable,'-c',"import sys,lora_setup;assert 'mlx' not in sys.modules and 'torch' not in sys.modules"],cwd=PROJECT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_huggingface_file_is_pinned_cached_and_provenance_is_recorded(self):
        api=Mock();api.model_info.return_value=types.SimpleNamespace(sha='b'*40)
        with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',return_value=str(self.source)) as fetch:
            code=self.run_cli(['--noninteractive','--source','https://huggingface.co/creator/style/blob/main/subdir/weights.safetensors?download=true','--id','ink','--trigger','inkstyle'])
        self.assertEqual(code,0);api.model_info.assert_called_once_with('creator/style',revision='main',timeout=15)
        api.list_repo_files.assert_not_called();self.assertEqual(fetch.call_args.kwargs['revision'],'b'*40)
        self.assertTrue(Path(fetch.call_args.kwargs['local_dir']).is_relative_to(self.store/'z-image-turbo/loras'))
        entry=json.loads((self.store/'z-image-turbo/loras/custom.json').read_text())[0]
        self.assertEqual(entry['revision'],'b'*40);self.assertEqual(entry['remote_filename'],'subdir/weights.safetensors');self.assertIn('b'*40,entry['source'])
        self.assertTrue(self.source.is_file())  # Never move/mutate the Hub cache file.

    def test_ambiguous_hub_files_require_selection_and_network_errors_keep_registrations(self):
        _,directory,_=self.install();before=(directory/'custom.json').read_bytes()
        api=Mock();api.model_info.return_value=types.SimpleNamespace(sha='b'*40);api.list_repo_files.return_value=['one.safetensors','two.safetensors']
        with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download') as fetch:
            self.assertEqual(self.run_cli(['--noninteractive','--source','creator/style']),1);fetch.assert_not_called()
        api.model_info.side_effect=RuntimeError('No access')
        with patch('huggingface_hub.HfApi',return_value=api):self.assertEqual(self.run_cli(['--noninteractive','--source','creator/style']),1)
        self.assertEqual((directory/'custom.json').read_bytes(),before)

    def test_custom_adapter_is_discovered_by_both_apis_without_hashing_or_queue_sql(self):
        entry,directory,_=self.install();setup_model(self.store,'z-image-turbo')
        app=queue_api.create_app(self.project);client=TestClient(app,base_url='http://127.0.0.1')
        try:
            with patch.object(queue_api.JobStore,'initialize',side_effect=AssertionError('No SQLite for discovery')):
                result=client.get('/v1/loras?model=z-image-turbo');self.assertEqual(result.status_code,200,result.text)
            found=next(item for item in result.json()['data'] if item['id']==entry['id']);self.assertTrue(found['available']);self.assertEqual(found['trigger_prompt'],'inkstyle')
        finally:client.close()
        import server,lora_catalog
        with patch.object(server,'PROJECT_ROOT',self.project),patch.object(server,'API_MODEL','z-image-turbo'),patch.object(lora_catalog,'ROOT',self.project),patch.object(lora_catalog,'LORA_DIR',directory),patch.object(server,'ready_models',return_value=['z-image-turbo']),patch.object(server.app.state,'loras_enabled',True):
            client=TestClient(server.app,base_url='http://127.0.0.1')
            try:self.assertIn('custom-ink',[item['id'] for item in client.get('/v1/loras?model=z-image-turbo').json()['data']])
            finally:client.close()
        self.assertFalse((self.project/'state').exists())

    def test_concurrent_installers_do_not_lose_catalog_entries(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            installed=list(executor.map(lambda identifier:self.install(identifier=identifier),['one','two']))
        catalog=installed[0][1]/'custom.json'
        self.assertEqual({item['id'] for item in json.loads(catalog.read_text())},{'custom-one','custom-two'})

    def test_make_passes_paths_and_trigger_words_literally_without_command_execution(self):
        marker=self.root/'unexpected-command-execution'
        trigger=f'literal `touch {marker}` $(shell touch {marker})'
        command=['make','setup-lora','VENV='+str(Path(sys.executable).parent.parent),
                 'MODEL=z-image-turbo','LORA_SOURCE='+str(self.source),'LORA_NAME=Ink Style','LORA_ID=literal',
                 'LORA_TRIGGER='+trigger,'NONINTERACTIVE=1']
        result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse(marker.exists())
        entries=json.loads((self.store/'z-image-turbo/loras/custom.json').read_text())
        self.assertEqual(entries[0]['trigger_prompt'],trigger)

    def test_git_pull_updates_project_without_touching_local_or_external_adapters(self):
        def git(folder,*args):
            result=subprocess.run(['git',*args],cwd=folder,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);return result.stdout
        for external in (False,True):
            with self.subTest(external=external):
                upstream=self.root/f'upstream-{external}';upstream.mkdir();git(upstream,'init','-b','main')
                shutil.copy2(PROJECT/'.gitignore',upstream/'.gitignore');shutil.copy2(self.project/'lora-catalog.json',upstream/'lora-catalog.json');git(upstream,'add','.');git(upstream,'-c','user.name=Fixture','-c','user.email=fixture@example.com','commit','-m','Initial fixture')
                clone=self.root/f'clone-{external}';git(self.root,'clone',str(upstream),str(clone))
                store=self.root/f'external-{external}' if external else clone
                with patch.dict(os.environ,{'DATA_DIR':str(store)}):
                    entry,directory,_=lora_setup.register(clone,'z-image-turbo',self.source,identifier='ink',name='Ink')
                    before=(directory/'custom.json').read_bytes();payload=(directory/entry['filename']).read_bytes()
                    self.assertEqual(git(clone,'status','--porcelain'),'')
                    public=json.loads((upstream/'lora-catalog.json').read_text());public[0]['name']='Updated public preset';(upstream/'lora-catalog.json').write_text(json.dumps(public));git(upstream,'add','.');git(upstream,'-c','user.name=Fixture','-c','user.email=fixture@example.com','commit','-m','Upstream preset update')
                    git(clone,'pull','--ff-only');self.assertEqual(git(clone,'status','--porcelain'),'')
                    self.assertEqual((directory/'custom.json').read_bytes(),before);self.assertEqual((directory/entry['filename']).read_bytes(),payload)
                    entries=adapter_catalog.read_catalog(clone,directory,'z-image-turbo');self.assertEqual(entries[0]['name'],'Updated public preset');self.assertEqual(entries[-1]['id'],'custom-ink')
