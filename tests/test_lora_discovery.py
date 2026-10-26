"""Discover per-model adapters with no generator, weights load or SQLite work."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import adapter_catalog
import queue_api
from tests.model_fixtures import setup_model


class LoRADiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.project=Path(self.temp.name).resolve()
        self.environment=patch.dict(os.environ,{'DATA_DIR':str(self.project)});self.environment.start()
        for identifier in ('z-image-turbo','chroma1-flash','chroma1-hd'):setup_model(self.project,identifier)
        self.client=TestClient(queue_api.create_app(self.project))

    def tearDown(self):
        self.client.close();self.environment.stop();self.temp.cleanup()

    def entry(self,identifier,models,**settings):
        return {'id':identifier,'name':identifier.title(),'filename':identifier+'.safetensors',
                'models':models,'default_strength':.8,'trigger_prompt':'retro','source':'https://example.com/adapter',
                'sha256':'a'*64,**settings}

    def catalog(self,folder,entries,installed=()):
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'catalog.json').write_text(json.dumps(entries))
        for entry in entries:
            if entry['id'] in installed:(folder/entry['filename']).write_bytes(b'presence fixture; not real adapter weights')

    def test_default_model_filters_compatibility_and_reports_metadata_without_hashing(self):
        folder=self.project/'z-image-turbo'/'loras'
        self.catalog(folder,[self.entry('ink',['z-image-turbo']),self.entry('missing',['z-image-turbo']),
                             self.entry('chroma',['chroma1-flash'])],installed=['ink','chroma'])
        original_open=Path.open
        def metadata_only(path,*args,**kwargs):
            if path.suffix=='.safetensors':
                raise AssertionError('Discovery must not open weights')
            return original_open(path,*args,**kwargs)
        with patch.object(queue_api.JobStore,'initialize',side_effect=AssertionError('Discovery must not touch SQLite')),\
             patch.object(Path,'open',metadata_only):
            response=self.client.get('/v1/loras')
        self.assertEqual(response.status_code,200,response.text)
        body=response.json();self.assertEqual(body['model'],'z-image-turbo')
        self.assertEqual(body['directory'],str(folder))
        self.assertEqual([entry['id'] for entry in body['data']],['ink','missing'])
        self.assertEqual([entry['available'] for entry in body['data']],[True,False])
        self.assertEqual(body['data'][0]['default_strength'],.8)
        self.assertEqual(body['data'][0]['trigger_prompt'],'retro')
        self.assertNotIn('enabled',body)
        self.assertFalse((self.project/'state').exists())

    def test_chroma_uses_shared_directory_filters_each_model_and_reuses_external_store(self):
        with tempfile.TemporaryDirectory() as external:
            store=Path(external).resolve()
            self.environment.stop()
            for identifier in ('chroma1-hd','chroma1-flash'):setup_model(store,identifier)
            (self.project/'runtime').mkdir(exist_ok=True)
            for name in ('sd-cli','libstable-diffusion.dylib'):(self.project/'runtime'/name).write_bytes(b'fixture')
            (self.project/'storage.json').write_text(json.dumps({'data_dir':str(store)}))
            folder=store/'chroma1-hd'/'loras'
            self.catalog(folder,[self.entry('flash',['chroma1-flash']),self.entry('hd',['chroma1-hd']),
                                 self.entry('both',['chroma1-hd','chroma1-flash'])],installed=['flash','hd','both'])
            for model,expected in [('chroma1-flash',['flash','both']),('chroma1-hd',['hd','both'])]:
                response=self.client.get('/v1/loras',params={'model':model})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['directory'],str(folder))
                self.assertEqual([entry['id'] for entry in response.json()['data']],expected)
            self.assertFalse((self.project/'state').exists())

    def test_public_fallback_empty_models_invalid_model_and_live_catalog_refresh(self):
        public=self.entry('realism',['z-image-turbo'])
        (self.project/'lora-catalog.json').write_text(json.dumps([public]))
        result=self.client.get('/v1/loras').json()
        self.assertEqual(result['data'][0]['id'],'realism');self.assertFalse(result['data'][0]['available'])
        self.assertEqual(self.client.get('/v1/loras?model=chroma1-flash').json()['data'],[])
        self.assertEqual(self.client.get('/v1/loras?model=unknown').status_code,422)
        self.catalog(self.project/'z-image-turbo'/'loras',[self.entry('local',['z-image-turbo'])],installed=['local'])
        self.assertEqual(self.client.get('/v1/loras').json()['data'][0]['id'],'local')
        schema=self.client.get('/openapi.json').json()
        model=next(item for item in schema['paths']['/v1/loras']['get']['parameters'] if item['name']=='model')
        self.assertNotIn('default',model['schema'])

    def test_missing_volume_corrupt_catalog_and_escape_are_clear_errors(self):
        folder=self.project/'z-image-turbo'/'loras'
        with patch.object(adapter_catalog,'require_storage',side_effect=RuntimeError('Storage volume is not mounted')):
            response=self.client.get('/v1/loras')
            self.assertEqual(response.status_code,503);self.assertIn('not mounted',response.json()['detail'])
            self.assertEqual(self.client.post('/v1/jobs',json={'request':{'prompt':'Queue stays available'}}).status_code,202)
        folder.mkdir(parents=True)
        (folder/'catalog.json').write_text('{invalid')
        self.assertEqual(self.client.get('/v1/loras').status_code,503)
        self.catalog(folder,[self.entry('escape',['z-image-turbo'])])
        outside=self.project/'outside.safetensors';outside.write_bytes(b'keep outside')
        (folder/'escape.safetensors').symlink_to(outside)
        self.assertEqual(self.client.get('/v1/loras').status_code,503)
        self.assertEqual(outside.read_bytes(),b'keep outside')


if __name__=='__main__':unittest.main()
