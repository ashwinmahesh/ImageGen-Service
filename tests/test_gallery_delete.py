"""Confirmed gallery deletion: real file/SQLite boundaries, no user data or GPU."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import image_catalog
from job_queue import JobStore, fingerprint
import server


class GalleryDeleteTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.images=self.root/'images';self.images.mkdir()
        self.store=JobStore(self.root/'state/queue/jobs.sqlite3');self.store.initialize()
        self.patches=[patch.object(server,'PROJECT_ROOT',self.root),patch.object(server,'ROOT',self.root),patch.object(server,'IMAGE_DIR',self.images),patch.object(server,'job_store',self.store)]
        for item in self.patches:item.start()
        self.client=TestClient(server.app,base_url='http://127.0.0.1')

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()

    def save(self,identifier='a'*32,folder=None,metadata=None,index=True):
        folder=folder or self.images;folder.mkdir(parents=True,exist_ok=True);path=folder/(identifier+'.png')
        Image.new('RGB',(128,80),'steelblue').save(path)
        path.with_suffix('.json').write_text(json.dumps(metadata or {'prompt':'A lighthouse','tags':['coast']}))
        if index:image_catalog.write_index(self.root,identifier,path,['favorite'])
        return path

    def delete(self,identifier='a'*32):return self.client.delete('/v1/images/'+identifier)

    def enqueue(self,**payload):
        request={'model':server.API_MODEL,'prompt':'Make a blue sky',**payload};return self.store.enqueue(request,fingerprint(request))

    def test_deletes_image_sidecar_index_all_preview_versions_and_keeps_other_data(self):
        job=self.enqueue();self.store.claim();self.store.finish(job['id'],result={'file_path':'history retained'})
        path=self.save(metadata={'prompt':'A lighthouse','job_id':job['id']});other=self.save('b'*32)
        reference=self.root/'references'/(('a'*32)+'.png');reference.parent.mkdir();reference.write_bytes(b'original upload')
        detail=self.client.get('/v1/images/'+'a'*32).json();self.assertEqual(self.client.get(detail['thumbnail_url']).status_code,200)
        old=self.root/'cache/thumbnails'/('a'*32+'-older.webp');old.write_bytes(b'old cache')
        response=self.delete();self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json(),{'id':'a'*32,'deleted':True,'cleanup_pending':[]})
        self.assertEqual(response.headers['cache-control'],'no-store')
        for file in [path,path.with_suffix('.json'),self.root/'image-index'/('a'*32+'.json'),old]:self.assertFalse(file.exists(),file)
        self.assertEqual(list((self.root/'cache/thumbnails').glob('a'*32+'-*')),[])
        for url in ['/v1/images/'+'a'*32,detail['url'],detail['metadata_url'],detail['thumbnail_url']]:self.assertEqual(self.client.get(url).status_code,404,url)
        self.assertEqual(self.client.get('/v1/images').json()['total'],1);self.assertTrue(other.is_file());self.assertTrue(reference.is_file());self.assertEqual(self.store.get(job['id'])['status'],'succeeded')
        self.assertEqual(self.delete().status_code,404)

    def test_custom_destinations_and_unreadable_legacy_sidecars(self):
        for number,custom in enumerate([False,True]):
            identifier=f'{number:032x}';path=self.save(identifier,folder=self.root/'external-custom' if custom else None,index=custom)
            path.with_suffix('.json').write_text('{broken JSON')
            self.assertEqual(self.delete(identifier).status_code,200);self.assertFalse(path.exists());self.assertFalse(path.with_suffix('.json').exists())

    def test_invalid_ids_missing_images_and_symlinks_cannot_delete_other_files(self):
        path=self.save();source=self.save('b'*32,folder=self.root/'sources',index=False)
        for identifier in ['invalid','A'*32,'c'*32]:self.assertEqual(self.delete(identifier).status_code,404)
        path.unlink();path.symlink_to(source.with_name('a'*32+'.png'))
        target=source.with_name('a'*32+'.png');target.write_bytes(b'linked original')
        self.assertEqual(self.delete().status_code,409);self.assertEqual(target.read_bytes(),b'linked original');self.assertTrue(path.is_symlink())

    def test_nonregular_companions_fail_before_removing_the_png(self):
        for companion in ['sidecar','index','thumbnail']:
            with self.subTest(companion=companion):
                path=self.save();target=path.with_suffix('.json') if companion=='sidecar' else self.root/'image-index'/('a'*32+'.json') if companion=='index' else self.root/'cache/thumbnails'/('a'*32+'-old.webp')
                target.parent.mkdir(parents=True,exist_ok=True);target.unlink(missing_ok=True)
                original=self.root/(companion+'-original');original.write_text('{}');target.symlink_to(original)
                self.assertEqual(self.delete().status_code,409);self.assertTrue(path.is_file());self.assertEqual(original.read_text(),'{}');target.unlink()

    def test_pending_references_masks_and_own_batch_block_deletion_but_unrelated_work_does_not(self):
        for field in ['reference_image','mask_image','owner']:
            with self.subTest(field=field):
                path=self.save();job=self.enqueue(**({field:str(path)} if field!='owner' else {}))
                if field=='owner':path.with_suffix('.json').write_text(json.dumps({'job_id':job['id']}))
                self.assertEqual(self.delete().status_code,409);self.assertTrue(path.exists())
                claimed=self.store.claim();self.assertEqual(claimed['id'],job['id']);self.assertEqual(self.delete().status_code,409)
                self.store.finish(job['id'],result={});self.assertEqual(self.delete().status_code,200)
        path=self.save();self.enqueue(reference_image=str(self.root/'unrelated.png'));self.store.claim();self.assertEqual(self.delete().status_code,200)

    def test_reference_alias_and_sqlite_outage_are_handled_without_data_loss(self):
        path=self.save();alias=self.root/'alias.png';alias.symlink_to(path);job=self.enqueue(reference_image=str(alias))
        self.assertEqual(self.delete().status_code,409);self.store.cancel(job['id'])
        with patch.object(self.store,'guard_artifact_deletion',side_effect=__import__('sqlite3').OperationalError('locked')):
            response=self.delete();self.assertEqual(response.status_code,503);self.assertEqual(response.json()['code'],'queue_storage_unavailable')
        self.assertTrue(path.is_file())

    def test_staging_failure_restores_every_original(self):
        path=self.save();before=path.read_bytes();sidecar=path.with_suffix('.json');index=self.root/'image-index'/('a'*32+'.json');saved=index.read_bytes();rename=Path.rename
        def fail_sidecar(current,target):
            if current==sidecar:raise PermissionError('read only')
            return rename(current,target)
        with patch.object(Path,'rename',fail_sidecar):self.assertEqual(self.delete().status_code,503)
        self.assertEqual(path.read_bytes(),before);self.assertTrue(sidecar.is_file());self.assertEqual(index.read_bytes(),saved);self.assertFalse(list(self.root.rglob('.delete-*')))

    def test_cleanup_failure_is_reported_after_the_image_is_removed_from_the_gallery(self):
        path=self.save();unlink=Path.unlink
        def fail_sidecar(current,*args,**kwargs):
            if current.name.startswith('.delete-') and current.suffix=='.json' and current.parent==path.parent:raise PermissionError('cannot erase')
            return unlink(current,*args,**kwargs)
        with patch.object(Path,'unlink',fail_sidecar),self.assertLogs('image-model-server',level='WARNING'):
            result=self.delete();self.assertEqual(result.status_code,200,result.text)
        pending=result.json()['cleanup_pending'];self.assertEqual(len(pending),1);self.assertTrue(Path(pending[0]).is_file());self.assertFalse(path.exists());self.assertEqual(self.client.get('/v1/images').json()['total'],0)

    def test_duplicate_ids_do_not_reveal_or_delete_a_different_copy(self):
        original=self.save();other=self.save(folder=self.root/'custom');self.assertEqual(self.delete().status_code,409);self.assertTrue(original.is_file());self.assertTrue(other.is_file())
        # Cross-model duplicates receive the same protection.
        root2=self.root/'other-model';root2.mkdir();other2=root2/'images';other2.mkdir();Image.new('RGB',(16,16)).save(other2/('a'*32+'.png'))
        other.unlink();(self.root/'image-index'/('a'*32+'.json')).unlink()
        with patch.object(server,'MODEL_ROOTS',{server.API_MODEL:self.root,'chroma1-flash':root2}):self.assertEqual(self.delete().status_code,409)
        self.assertTrue(original.exists());self.assertTrue((other2/('a'*32+'.png')).exists())

    def test_concurrent_delete_is_once_only_and_stale_tag_updates_cannot_recreate_index(self):
        path=self.save()
        with ThreadPoolExecutor(max_workers=2) as executor:responses=list(executor.map(lambda _:self.delete(),range(2)))
        self.assertEqual(sorted(r.status_code for r in responses),[200,404])
        with self.assertRaises(FileNotFoundError):image_catalog.write_index(self.root,'a'*32,path,['stale'])
        self.assertFalse((self.root/'image-index'/('a'*32+'.json')).exists())

    def test_thumbnail_finishing_after_delete_cannot_republish_cache(self):
        self.save();save=Image.Image.save
        def deleting_save(image,file,*args,**kwargs):
            result=save(image,file,*args,**kwargs)
            if kwargs.get('format')=='WEBP':self.assertEqual(self.delete().status_code,200)
            return result
        with patch.object(Image.Image,'save',deleting_save):self.assertEqual(self.client.get('/thumbnails/'+'a'*32+'.webp').status_code,404)
        self.assertFalse(list((self.root/'cache/thumbnails').glob('*')))

    def test_delete_is_in_generator_schema_with_no_path_body(self):
        route=self.client.get('/openapi.json').json()['paths']['/v1/images/{identifier}']['delete'];self.assertNotIn('requestBody',route);self.assertIn('200',route['responses'])


if __name__=='__main__':unittest.main()
