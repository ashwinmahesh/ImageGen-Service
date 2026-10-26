"""Base-only mode must prevent API adapters and propagate through the launcher."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import launch
import server
from tests import test_api


class ServingModeTest(unittest.TestCase):
    setUp = test_api.APITest.setUp
    tearDown = test_api.APITest.tearDown
    post = test_api.APITest.post

    def test_base_only_generation_and_hidden_catalog(self):
        with patch.object(server.app.state, 'loras_enabled', False), patch.object(server, 'list_loras', side_effect=AssertionError('Catalog must not be read')):
            self.assertFalse(self.client.get('/health').json()['loras_enabled'])
            self.assertEqual(self.client.get('/v1/loras').json(), {'object':'list','model':server.API_MODEL,'enabled':False,'data':[]})
            response=self.post()
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(self.fake.calls[-1]['lora'])
            response=self.post(lora=None)
            self.assertEqual(response.status_code, 200, response.text)
            with open(response.json()['data'][0]['metadata_path']) as stream:
                meta=json.load(stream)
            self.assertFalse(meta['loras_enabled'])
            self.assertIsNone(meta['lora'])
            self.assertEqual(meta['effective_prompt'], self.payload['prompt'])

    def test_adapter_requests_rejected_before_inference(self):
        with patch.object(server.app.state, 'loras_enabled', False):
            response=self.post(lora='realism',lora_strength=.8)
            self.assertEqual(response.status_code, 422)
            self.assertIn('LoRAs are disabled', response.json()['detail'])
            self.assertEqual(self.fake.calls, [])

    def test_default_mode_remains_enabled(self):
        self.assertTrue(self.client.get('/health').json()['loras_enabled'])
        self.assertTrue(self.client.get('/v1/loras').json()['enabled'])


class LauncherModeTest(unittest.TestCase):
    def run_launcher(self, args, state=None, *, stub_enhancer=True):
        patches=[patch.object(sys,'argv',['launch.py',*args]), patch.object(launch.os,'chdir'),
                 patch.object(launch.os,'execv'),patch.object(launch.os,'kill'),patch.object(launch,'check_assets'),patch.object(launch,'ensure_queue')]
        if state is not None:
            state={'service':'imagegen-generator','runtime_model_switching':True,'project_dir':str(launch.ROOT),**state}
            patches.append(patch.object(launch,'initialize_shared_queue'))
        if state is None:
            patches.append(patch.object(launch,'port_free',return_value=True))
        else:
            patches.extend([patch.object(launch,'port_free',side_effect=[False,True]),
                patch.object(launch.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(state).encode())),
                patch.object(launch.subprocess,'check_output',side_effect=['1234\n',f'n{launch.ROOT}\n','python server.py'])])
        if stub_enhancer:
            patches.append(patch.object(launch,'check_enhancer_assets'))
        with contextlib.ExitStack() as stack:
            mocks=[stack.enter_context(item) for item in patches]
            with contextlib.redirect_stdout(io.StringIO()):
                launch.main()
            return mocks[2],mocks[3]

    def test_flag_forwarded_to_server(self):
        execute,_=self.run_launcher(['--no-loras','--port','8010'])
        command=execute.call_args.args[1]
        self.assertIn('--no-loras',command)
        self.assertIn('8010',command)
        execute,_=self.run_launcher([])
        self.assertNotIn('--no-loras',execute.call_args.args[1])
        self.assertEqual(execute.call_args.args[1][execute.call_args.args[1].index('--port')+1], '5732')

    def test_same_model_switches_mode_only_while_idle(self):
        execute,kill=self.run_launcher(['--no-loras'], {'model':launch.API_MODEL,'busy':False,'loras_enabled':True})
        kill.assert_called_once()
        self.assertIn('--no-loras',execute.call_args.args[1])
        execute,kill=self.run_launcher([], {'model':launch.API_MODEL,'busy':False,'loras_enabled':False})
        kill.assert_called_once()
        self.assertNotIn('--no-loras',execute.call_args.args[1])

    def test_mode_switch_accepts_custom_checkout_folder(self):
        with patch.object(launch, 'ROOT', Path('/tmp/custom-checkout-name')):
            execute,kill=self.run_launcher(['--no-loras'], {'model':launch.API_MODEL,'busy':False,'loras_enabled':True})
            kill.assert_called_once()
            self.assertIn('/tmp/custom-checkout-name/server.py',execute.call_args.args[1])

    def test_busy_mode_switch_is_refused_and_same_mode_reused(self):
        with self.assertRaisesRegex(SystemExit,'processing'):
            self.run_launcher(['--no-loras'], {'model':launch.API_MODEL,'busy':True,'loras_enabled':True})
        execute,kill=self.run_launcher(['--no-loras'], {'model':launch.API_MODEL,'busy':True,'loras_enabled':False,'prompt_expansion':{'enabled':True}})
        execute.assert_not_called()
        kill.assert_not_called()


if __name__=='__main__':
    unittest.main()
