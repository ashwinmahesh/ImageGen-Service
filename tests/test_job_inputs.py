"""Job input previews: a job's own source photo and mask, from any saved path, without inference."""
import unittest
from PIL import Image
from tests import test_api


class JobInputTest(unittest.TestCase):
    def setUp(self):
        self.fixture=test_api.APITest();self.fixture.setUp()
        self.fixture.store.pause(True)  # Keep the jobs waiting; previews only read their saved request.
        self.client=self.fixture.client;self.folder=self.fixture.root/'elsewhere'/'portraits';self.folder.mkdir(parents=True)

    def tearDown(self):
        self.fixture.tearDown()

    def add(self,**request):
        return self.fixture.store.enqueue({'prompt':'Change the jacket','model':'qwen-image-2.1-turbo','task':'edit',**request},str(request))

    def test_serves_any_saved_photo_and_mask_named_by_the_job(self):
        photo=self.folder/'tobin-3.jpg';Image.new('RGB',(40,30),'teal').save(photo)
        mask=self.folder/'mask.png';Image.new('L',(40,30),255).save(mask)
        job=self.add(reference_image=str(photo),mask_image=str(mask))
        response=self.client.get(f"/v1/jobs/{job['id']}/inputs/reference_image")
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual((response.headers['content-type'],response.content),('image/jpeg',photo.read_bytes()))
        response=self.client.get(f"/v1/jobs/{job['id']}/inputs/mask_image")
        self.assertEqual((response.status_code,response.headers['content-type']),(200,'image/png'))

    def test_missing_unreadable_and_absent_inputs_are_clear_errors(self):
        text=self.folder/'notes.png';text.write_text('not an image')
        unreadable=self.add(reference_image=str(text))
        self.assertEqual(self.client.get(f"/v1/jobs/{unreadable['id']}/inputs/reference_image").status_code,415)
        self.assertEqual(self.client.get(f"/v1/jobs/{unreadable['id']}/inputs/mask_image").status_code,404)
        gone=self.add(reference_image=str(self.folder/'deleted.png'))
        self.assertEqual(self.client.get(f"/v1/jobs/{gone['id']}/inputs/reference_image").status_code,404)
        self.assertEqual(self.client.get(f"/v1/jobs/{'0'*32}/inputs/reference_image").status_code,404)
        self.assertEqual(self.client.get(f"/v1/jobs/{gone['id']}/inputs/prompt").status_code,422)


if __name__=='__main__':
    unittest.main()
