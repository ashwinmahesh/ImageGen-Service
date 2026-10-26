"""Default-on prompt capability and explicit opt-out, without starting inference."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch
import launch
import server
import test_serving_mode


class LLMStartup(unittest.TestCase):
    run_launcher=test_serving_mode.LauncherModeTest.run_launcher

    def test_interactive_enter_yes_no_invalid_and_eof(self):
        for answer,expected in [('',True),('Y',True),('yes',True),('n',False),(' NO ',False)]:
            prompt=Mock(return_value=answer)
            self.assertIs(launch.select_llm_mode(None,interactive=True,prompt=prompt),expected)
            self.assertIn('[Y/n]',prompt.call_args.args[0])
        with contextlib.redirect_stdout(io.StringIO()):
            prompt=Mock(side_effect=['maybe','n'])
            self.assertFalse(launch.select_llm_mode(None,interactive=True,prompt=prompt))
        self.assertEqual(prompt.call_count,2)
        with self.assertRaisesRegex(SystemExit,'cancelled'):
            launch.select_llm_mode(None,interactive=True,prompt=Mock(side_effect=EOFError))

    def test_noninteractive_and_explicit_choices_do_not_prompt(self):
        prompt=Mock(side_effect=AssertionError('Must not prompt'))
        self.assertTrue(launch.select_llm_mode(None,interactive=False,prompt=prompt))
        for explicit in (True,False):
            self.assertIs(launch.select_llm_mode(explicit,interactive=True,prompt=prompt),explicit)
        with patch.object(launch.sys.stdin,'isatty',return_value=True),patch.object(launch.sys.stdout,'isatty',return_value=True),patch.dict(os.environ,{'NONINTERACTIVE':'1'}):
            self.assertTrue(launch.select_llm_mode(None,prompt=prompt))

    def test_default_on_checks_assets_and_never_silently_falls_back(self):
        with patch.object(launch,'check_enhancer_assets') as assets:
            execute,_=self.run_launcher([],stub_enhancer=False)
            assets.assert_called_once_with(launch.ROOT)
            self.assertIn('--with-llm',execute.call_args.args[1])
        with patch.object(launch,'check_enhancer_assets',side_effect=RuntimeError('Run make download-llm')),patch.object(launch,'check_assets'),patch.object(launch.sys,'argv',['launch.py','--model','z-image-turbo']),patch.dict(os.environ,{'NONINTERACTIVE':'1'}),patch.object(launch,'port_free') as port,patch.object(launch.os,'kill') as kill:
            with self.assertRaisesRegex(SystemExit,'NO_LLM=1'):
                launch.main()
            port.assert_not_called()
            kill.assert_not_called()

    def test_explicit_off_skips_prompt_assets_and_is_forwarded(self):
        with patch.object(launch,'check_enhancer_assets',side_effect=AssertionError('Off mode must not require LLM weights')):
            execute,_=self.run_launcher(['--no-llm'],stub_enhancer=False)
        command=execute.call_args.args[1]
        self.assertIn('--no-llm',command);self.assertNotIn('--with-llm',command)
        state={'model':launch.API_MODEL,'busy':True,'loras_enabled':True,'prompt_expansion':{'enabled':False}}
        execute,kill=self.run_launcher(['--no-llm'],state)
        execute.assert_not_called();kill.assert_not_called()
        with self.assertRaisesRegex(SystemExit,'generating'):self.run_launcher([],state)

    def test_interactive_off_reaches_server_and_conflicting_cli_flags_reject(self):
        with patch.object(launch,'select_llm_mode',return_value=False):
            execute,_=self.run_launcher([])
        self.assertIn('--no-llm',execute.call_args.args[1])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            self.run_launcher(['--with-llm','--no-llm'])
        self.assertEqual(error.exception.code,2)

    def test_direct_server_default_and_explicit_opt_out(self):
        self.assertTrue(server.parse_server_args([]).with_llm)
        self.assertFalse(server.parse_server_args(['--no-llm']).with_llm)
        self.assertTrue(server.parse_server_args(['--with-llm']).with_llm)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            server.parse_server_args(['--with-llm','--no-llm'])

    def test_make_passes_explicit_modes_and_rejects_conflicts(self):
        environment={key:value for key,value in os.environ.items() if key not in {'WITH_LLM','NO_LLM','MAKEFLAGS','MFLAGS'}}
        root=Path(__file__).resolve().parent
        def dry(*options):
            return subprocess.run(['make','-n','serve','MODEL=z-image-turbo',*options],cwd=root,env=environment,capture_output=True,text=True)
        default=dry();self.assertEqual(default.returncode,0,default.stderr)
        self.assertNotIn('--no-llm',default.stdout)
        for options,flag in [(('NO_LLM=1',),'--no-llm'),(('WITH_LLM=0',),'--no-llm'),(('WITH_LLM=1',),'--with-llm')]:
            result=dry(*options);self.assertEqual(result.returncode,0,result.stderr);self.assertIn(flag,result.stdout)
        for options in [('NO_LLM=1','WITH_LLM=1'),('WITH_LLM=invalid',)]:
            self.assertNotEqual(dry(*options).returncode,0)


if __name__=='__main__':unittest.main()
