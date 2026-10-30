"""Job progress: runtime step parsing, whole-job percentages and both APIs, without inference."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import job_progress
from job_progress import ProgressTracker, last_step, runtime_steps
from job_logs import candidate_identifier, runtime_folder
from job_queue import JobStore
import queue_api
import server
from tests.model_fixtures import setup_model


class StepParsingTest(unittest.TestCase):
    def test_reads_the_newest_counter_from_each_runtime_format(self):
        mflux=' 50%|█████     | 4/8 [01:32<01:29, 22.30s/it]\r 62%|██████▎   | 5/8 [01:53<01:05, 21.87s/it]'
        self.assertEqual(last_step(mflux),(5,8,21.87))
        fast=' 75%|███████▌  | 3/4 [00:02<00:00,  2.00it/s]'
        self.assertEqual(last_step(fast),(3,4,.5))
        native='  |=====>      | 7/8 -   1.73s/it - ETA 00:01\x1b[K\n[I] saving'
        self.assertEqual(last_step(native),(7,8,1.73))

    def test_ignores_unstarted_bars_weight_loading_and_unrelated_text(self):
        self.assertIsNone(last_step('  0%|          | 0/8 [00:00<?, ?it/s]'))
        self.assertIsNone(last_step('Loading checkpoint shards:  67%|██████▋   | 2/3 [00:01<00:00,  1.50it/s]'))
        self.assertIsNone(last_step('Wrote 3/4 files'))
        sampling=' 25%|██▌       | 1/4 [00:05<00:15,  5.00s/it]\nLoading checkpoint shards: 100%|██| 3/3 [00:01<00:00, 2.0it/s]'
        self.assertEqual(last_step(sampling),(1,4,5.0))


class Worker:
    def __init__(self,job=None,stage='idle'):self.job,self.stage=job,stage


class Store:
    def __init__(self):self.saved=[]
    def save_live_progress(self,identifier,attempt,progress):
        self.saved.append((identifier,attempt,progress));return True


def job(n=1,attempts=1,model='qwen-image-2.1-turbo'):
    return {'id':'a'*32,'attempts':attempts,'started':time.time(),'request':{'model':model,'n':n}}


class TrackerTest(unittest.TestCase):
    def tracker(self,worker,steps):
        self.now=[100.0]
        return ProgressTracker(Store(),worker,Path('/unused'),steps=lambda project,job,index:steps.get(index),clock=lambda:self.now[0])

    def test_single_image_phases_percent_and_estimate(self):
        steps={};worker=Worker(job(),'loading_model');tracker=self.tracker(worker,steps)
        self.assertEqual(tracker.poll()['phase'],'loading_model');self.assertEqual(tracker.current['percent'],0)
        worker.stage='generating'
        self.assertEqual(tracker.poll()['phase'],'loading_model')
        steps[0]=(4,8,20.0)
        progress=tracker.poll()
        self.assertEqual((progress['phase'],progress['percent'],progress['step'],progress['total_steps']),('sampling',47,4,8))
        self.assertEqual((progress['image'],progress['total_images'],progress['estimated_remaining_seconds']),(1,1,80))
        steps[0]=(8,8,20.0)
        progress=tracker.poll()
        self.assertEqual((progress['phase'],progress['percent'],progress['estimated_remaining_seconds']),('finishing',95,0))

    def test_batch_counts_each_image_equally_and_never_moves_backwards(self):
        steps={0:(4,4,2.0)};worker=Worker(job(n=4),'generating_option_1_of_4');tracker=self.tracker(worker,steps)
        self.assertEqual(tracker.poll()['percent'],23)
        self.now[0]=130.0;worker.stage='generating_option_2_of_4'
        progress=tracker.poll()  # Option 2 is still loading its runtime.
        self.assertEqual((progress['phase'],progress['image'],progress['percent']),('loading_model',2,25))
        steps[1]=(2,4,2.0)
        progress=tracker.poll()
        self.assertEqual(progress['percent'],36)
        # Two remaining images at the measured 30s per image, plus this image's two steps.
        self.assertEqual(progress['estimated_remaining_seconds'],64)
        worker.stage='expanding_prompt'
        self.assertEqual(tracker.poll()['percent'],36)

    def test_running_jobs_stay_below_100_and_new_attempts_restart(self):
        steps={0:(8,8,1.0)};worker=Worker(job(n=1),'generating');tracker=self.tracker(worker,steps)
        self.assertLess(tracker.poll()['percent'],100)
        worker.job=job(attempts=2);worker.stage='waiting_for_accelerator';steps.clear()
        self.assertEqual(tracker.poll()['percent'],0)
        self.assertEqual(tracker.store.saved[-1][1],2)
        worker.job=None
        self.assertIsNone(tracker.poll())

    def test_unchanged_progress_is_not_rewritten(self):
        steps={0:(2,8,3.0)};worker=Worker(job(),'generating');tracker=self.tracker(worker,steps)
        tracker.poll();tracker.poll()
        self.assertEqual(len(tracker.store.saved),1)
        steps[0]=(3,8,3.0);tracker.poll()
        self.assertEqual(len(tracker.store.saved),2)

    def test_tracker_errors_never_stop_polling(self):
        worker=Worker(job(),'generating')
        tracker=ProgressTracker(Store(),worker,Path('/unused'),interval=.01,steps=lambda *args:1/0)
        tracker.start();time.sleep(.05);tracker.stop()
        self.assertFalse(tracker.thread.is_alive())


class RuntimeLogTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.storage=patch.dict(os.environ,{'DATA_DIR':str(self.root/'data')});self.storage.start()

    def tearDown(self):
        self.storage.stop();self.temp.cleanup()

    def test_native_runtime_reads_this_attempts_option_log_only(self):
        identifier='b'*32;batch={'id':identifier,'attempts':2,'started':time.time(),'request':{'model':'flux2-klein-4b','n':3}}
        logs=self.root/'data'/'flux2-klein-4b'/'logs';logs.mkdir(parents=True)
        path=logs/(candidate_identifier(identifier,1,3)+'.log')
        self.assertIsNone(runtime_steps(self.root,batch,1))
        path.write_text(' 50%|█████     | 2/4 [00:08<00:08,  4.00s/it]')
        self.assertEqual(runtime_steps(self.root,batch,1),(2,4,4.0))
        old=batch['started']-60;os.utime(path,(old,old))  # Left by an earlier attempt.
        self.assertIsNone(runtime_steps(self.root,batch,1))

    def test_resident_runtime_counts_steps_after_the_latest_stage(self):
        identifier='c'*32;resident={'id':identifier,'attempts':1,'started':time.time(),'request':{'model':'z-image-turbo','n':2}}
        folder=runtime_folder(self.root,identifier);folder.mkdir(parents=True)
        log=folder/'attempt-1.log'
        log.write_text('[t] Stage: generating_option_1_of_2\n100%|██████████| 9/9 [00:16<00:00,  1.34s/it]\n[t] Stage: generating_option_2_of_2\n')
        self.assertIsNone(runtime_steps(self.root,resident,1))
        with log.open('a') as stream:stream.write(' 33%|███▎      | 3/9 [00:04<00:08,  1.40s/it]')
        self.assertEqual(runtime_steps(self.root,resident,1),(3,9,1.4))


class StoredProgressTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        for identifier in ('z-image-turbo','chroma1-flash'):setup_model(self.root,identifier)
        self.client=TestClient(queue_api.create_app(self.root))
        self.store=JobStore(self.root/'state'/'queue'/'jobs.sqlite3')

    def tearDown(self):
        self.client.close();self.temp.cleanup()

    def test_both_apis_report_the_saved_running_percentage(self):
        admitted=self.client.post('/v1/jobs',json={'model':'z-image-turbo','request':{'prompt':'A castle','n':2}}).json()
        self.assertIsNone(admitted['progress'])
        claimed=self.store.claim()
        self.assertEqual(self.store.get(claimed['id'])['progress']['phase'],'starting')
        saved={'percent':36,'phase':'sampling','image':2,'total_images':2,'step':2,'total_steps':4,'estimated_remaining_seconds':64,'updated':1.0}
        self.assertFalse(self.store.save_live_progress(claimed['id'],claimed['attempts']+1,saved))
        self.assertTrue(self.store.save_live_progress(claimed['id'],claimed['attempts'],saved))
        self.assertEqual(self.client.get('/v1/jobs/'+claimed['id']).json()['progress'],saved)
        self.assertEqual(self.client.get('/v1/jobs?status=pending').json()['data'][0]['progress'],saved)
        generator=server.JobResponse.model_validate(server.job_response(self.store.get(claimed['id']),None,ready=['z-image-turbo']))
        self.assertEqual(generator.progress.model_dump(),saved)
        self.store.requeue(claimed['id'])
        self.assertIsNone(self.store.get(claimed['id'])['progress'])
        again=self.store.claim()
        self.assertEqual(self.store.get(again['id'])['progress']['percent'],0)
        self.store.finish(again['id'],result={'created':1})
        done=self.client.get('/v1/jobs/'+again['id']).json()['progress']
        self.assertEqual((done['percent'],done['phase'],done['image'],done['total_images']),(100,'done',2,2))

    def test_failed_jobs_and_stale_attempts_have_no_progress(self):
        self.client.post('/v1/jobs',json={'model':'z-image-turbo','request':{'prompt':'A castle'}})
        claimed=self.store.claim()
        self.store.save_live_progress(claimed['id'],claimed['attempts'],{'percent':50,'phase':'sampling','total_images':1})
        self.store.finish(claimed['id'],error='Failed')
        self.assertIsNone(self.store.get(claimed['id'])['progress'])
        self.assertFalse(self.store.save_live_progress(claimed['id'],claimed['attempts'],{'percent':60}))


if __name__=='__main__':
    unittest.main()
