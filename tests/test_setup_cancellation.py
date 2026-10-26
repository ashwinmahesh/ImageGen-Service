"""Exercise real Make/terminal cancellation with disposable, non-GPU setup work."""
import contextlib
import errno
import io
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import setup as model_setup
from model_profiles import select_model

ROOT=Path(__file__).resolve().parents[1]


class SetupCancellationTest(unittest.TestCase):
    def test_cancel_and_eof_before_selection_do_not_start_setup(self):
        for error in (KeyboardInterrupt,EOFError):
            with self.subTest(error=error),patch.object(model_setup,'select_model',side_effect=error), \
                 patch.object(model_setup.subprocess,'Popen') as start,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(model_setup.main(),130)
                start.assert_not_called()
                self.assertIn('Setup cancelled.',output.getvalue())
                self.assertIn('unchanged model weights need no redownload',output.getvalue())

    def test_quit_answers_cancel_without_changing_default_or_model_ids(self):
        with patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(io.StringIO()):
            for answer in ('q','Q','quit','cancel'):
                with self.assertRaises(KeyboardInterrupt):
                    select_model(interactive=True,allow_cancel=True,prompt=lambda _:answer)
            self.assertEqual(select_model(interactive=True,allow_cancel=True,prompt=lambda _:''),'z-image-turbo')
            self.assertEqual(select_model(interactive=True,allow_cancel=True,prompt=lambda _:'chroma1-flash'),'chroma1-flash')

    def test_setup_child_ignoring_termination_is_killed_and_reaped(self):
        process=unittest.mock.Mock(pid=12345)
        process.wait.side_effect=[subprocess.TimeoutExpired('setup',5),-signal.SIGKILL]
        with patch.object(model_setup.os,'killpg') as stop:
            model_setup.stop_setup(process)
        self.assertEqual(stop.call_args_list,[unittest.mock.call(12345,signal.SIGTERM),unittest.mock.call(12345,signal.SIGKILL)])
        self.assertEqual(process.wait.call_count,2)

    def test_interrupt_during_spawn_still_stops_the_created_group(self):
        process=unittest.mock.Mock()
        def spawn(*args,**kwargs):
            signal.raise_signal(signal.SIGINT)
            return process
        with patch.object(model_setup,'select_model',return_value='z-image-turbo'), \
             patch.object(model_setup.subprocess,'Popen',side_effect=spawn),patch.object(model_setup,'stop_setup') as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model_setup.main(),130)
            stop.assert_called_once_with(process)


@unittest.skipUnless(os.name=='posix','Terminal/process-group checks require POSIX')
class SetupTerminalTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        for name in ['Makefile','bootstrap_python.sh','setup.py','model_profiles.py','model_assets.py','storage_paths.py']:
            shutil.copy2(ROOT/name,self.root/name)
        binaries=self.root/'bin';binaries.mkdir()
        # Only nested setup work is replaced. The outer make setup and bootstrap
        # are real, including terminal input and the model menu.
        stage=binaries/'make'
        stage.write_text(f'''#!{sys.executable}
import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path.cwd()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
def terminate(*_):
    child.wait(timeout=3)
    raise SystemExit(143)
signal.signal(signal.SIGTERM,terminate)
(root/'partial-download').write_text('resumable bytes')
(root/'children.json').write_text(json.dumps([os.getpid(),child.pid]))
print('DISPOSABLE STORAGE PROMPT:',flush=True)
answer=input()
print('INPUT RECEIVED: '+answer,flush=True)
time.sleep(60)
''')
        stage.chmod(0o755)
        self.environment={key:value for key,value in os.environ.items() if key not in ('MODEL','IMAGE_MODEL','NONINTERACTIVE','DATA_DIR','VENV')}
        self.environment.update(PATH=str(binaries)+os.pathsep+os.environ.get('PATH',''),DATA_DIR=str(self.root/'models'))
        self.process=None;self.master=None;self.slave=None;self.output=''

    def tearDown(self):
        children=self.root/'children.json'
        if children.exists():
            for pid in json.loads(children.read_text()):
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if self.master is not None:os.close(self.master)
        if self.slave is not None:os.close(self.slave)
        self.temporary.cleanup()

    def launch(self):
        self.master,self.slave=pty.openpty()
        # Keep the slave open until drained: Darwin can discard unread terminal
        # output on the last slave close, hiding a fast cancellation message.
        self.process=subprocess.Popen([shutil.which('make'),'setup','BOOTSTRAP_PYTHON='+sys.executable],cwd=self.root,
            env=self.environment,stdin=self.slave,stdout=self.slave,stderr=self.slave,start_new_session=True)
        self.read_until('Model [1 / Z-Image Turbo]:')

    def read_until(self,expected=None):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            ready,_,_=select.select([self.master],[],[],.1)
            if ready:
                try:chunk=os.read(self.master,65536)
                except OSError as error:
                    if error.errno==errno.EIO:break
                    raise
                if not chunk:break
                self.output+=chunk.decode(errors='replace')
            if expected and expected in self.output:return
            if expected is None and self.process.poll() is not None and not ready:break
        if expected:self.assertIn(expected,self.output)

    def assert_cancelled(self):
        self.process.wait(timeout=8);self.read_until()
        self.assertNotEqual(self.process.returncode,0)
        self.assertIn('Setup cancelled.',self.output)
        self.assertNotIn('Traceback',self.output)
        self.assertFalse((self.root/'models').exists())

    def test_q_cancels_actual_make_setup_menu(self):
        self.launch();os.write(self.master,b'q\n');self.assert_cancelled()
        self.assertFalse((self.root/'children.json').exists())

    def test_ctrl_c_at_actual_make_setup_menu(self):
        self.launch();os.killpg(self.process.pid,signal.SIGINT);self.assert_cancelled()
        self.assertFalse((self.root/'children.json').exists())

    def test_ctrl_c_after_selection_stops_descendants_and_keeps_partial_download(self):
        self.launch();os.write(self.master,b'\n')
        self.read_until('DISPOSABLE STORAGE PROMPT:')
        # The isolated child still reads the same terminal; stdin must survive.
        os.write(self.master,b'keep-existing-store\n');self.read_until('INPUT RECEIVED: keep-existing-store')
        children=json.loads((self.root/'children.json').read_text())
        self.assertNotEqual(os.getpgid(children[0]),self.process.pid)
        os.killpg(self.process.pid,signal.SIGINT);self.assert_cancelled()
        self.assertEqual((self.root/'partial-download').read_text(),'resumable bytes')
        for pid in children:
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':
    unittest.main()
