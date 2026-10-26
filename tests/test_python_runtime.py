"""Python selection and setup boundaries with fake downloads; no models or GPU."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import python_runtime as runtime
import runtime_setup

PROJECT = Path(__file__).resolve().parents[1]


def info(minor=14, **overrides):
    value = dict(version=[3,minor,9],implementation='cpython',system='Darwin',machine='arm64',
                 free_threaded=False,releaselevel='final',path='/installed/python',base='/installed/python',
                 prefix='/installed',base_prefix='/installed')
    value.update(overrides)
    return value


class PythonRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.target=self.root/'.venv-test'
        self.enterContext(patch.object(runtime.platform,'system',return_value='Darwin'))
        self.enterContext(patch.object(runtime.platform,'machine',return_value='arm64'))
        self.enterContext(redirect_stderr(io.StringIO()))

    def virtual(self,**values):
        return info(**{'prefix':str(self.target),**values})

    def test_full_lock_python_bounds_and_unsupported_builds(self):
        for minor in range(9,16):
            self.assertEqual(runtime.compatible(info(minor)),minor in {12,13,14})
        self.assertFalse(runtime.compatible(info(version=[3,14,1])))
        for change in [{'implementation':'pypy'},{'system':'Linux'},{'machine':'x86_64'},
                       {'free_threaded':True},{'releaselevel':'candidate'}]:
            self.assertFalse(runtime.compatible(info(**change)))
        self.assertFalse(runtime.compatible(None))

    def test_existing_environment_is_reused_without_search_creation_or_download(self):
        self.target.mkdir();(self.target/'keep').write_text('preserve')
        with patch.object(runtime,'probe_python',return_value=self.virtual()), \
             patch.object(runtime,'installed_pythons') as candidates, \
             patch.object(runtime,'managed_python') as download, patch.object(runtime.subprocess,'run') as run:
            self.assertEqual(runtime.ensure_environment(self.root,self.target),self.target/'bin/python')
            candidates.assert_not_called();download.assert_not_called();run.assert_not_called()
        self.assertEqual((self.target/'keep').read_text(),'preserve')

    def test_broken_incompatible_and_nonvirtual_targets_are_preserved(self):
        self.target.mkdir();(self.target/'keep').write_text('preserve')
        for detected in [None,self.virtual(version=[3,11,9]),info(),self.virtual(prefix=str(self.root/'other'))]:
            with patch.object(runtime,'probe_python',return_value=detected), \
                 patch.object(runtime,'managed_python') as download,patch.object(runtime.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'preserved'):
                    runtime.ensure_environment(self.root,self.target)
                download.assert_not_called();run.assert_not_called()
        self.assertEqual((self.target/'keep').read_text(),'preserve')

    def test_broken_environment_symlink_is_not_replaced(self):
        self.target.symlink_to(self.root/'missing')
        with patch.object(runtime,'probe_python',return_value=None),patch.object(runtime,'managed_python') as download:
            with self.assertRaisesRegex(RuntimeError,'preserved'):runtime.ensure_environment(self.root,self.target)
            download.assert_not_called()
        self.assertTrue(self.target.is_symlink())

    def test_installed_python_is_selected_after_skipping_incompatible_default(self):
        def detect(path):
            if str(path)==str(self.target/'bin/python'):return self.virtual(version=[3,12,9])
            return info(15,base='/default') if str(path)=='/default' else info(12)
        with patch.object(runtime,'installed_pythons',return_value=['/default','/compatible']), \
             patch.object(runtime,'probe_python',side_effect=detect),patch.object(runtime,'managed_python') as download, \
             patch.object(runtime.subprocess,'run') as run:
            runtime.ensure_environment(self.root,self.target)
            run.assert_called_once_with(['/installed/python','-m','venv','--copies',str(self.target)],check=True)
            download.assert_not_called()

    def test_valid_explicit_interpreter_is_honored_without_search(self):
        with patch.object(runtime,'probe_python',side_effect=[info(12),self.virtual(version=[3,12,9])]), \
             patch.object(runtime,'installed_pythons') as candidates,patch.object(runtime,'managed_python') as download, \
             patch.object(runtime.subprocess,'run'):
            runtime.ensure_environment(self.root,self.target,'/chosen/python')
            candidates.assert_not_called();download.assert_not_called()

    def test_bad_explicit_override_never_falls_back(self):
        for detected in [None,info(11),info(version=[3,14,1])]:
            with patch.object(runtime,'probe_python',return_value=detected),patch.object(runtime,'managed_python') as download:
                with self.assertRaisesRegex(RuntimeError,'Explicit interpreter'):
                    runtime.ensure_environment(self.root,self.target,'/wrong/python')
                download.assert_not_called()

    def test_different_explicit_base_does_not_rebuild_existing_environment(self):
        self.target.mkdir()
        with patch.object(runtime,'probe_python',side_effect=[info(base='/chosen/python'),self.virtual()]), \
             patch.object(runtime.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'different Python'):
                runtime.ensure_environment(self.root,self.target,'/chosen/python')
            run.assert_not_called()

    def test_only_missing_compatible_python_triggers_project_download(self):
        with patch.object(runtime,'installed_pythons',return_value=[]), \
             patch.object(runtime,'managed_python',return_value='/downloaded/python') as download, \
             patch.object(runtime,'probe_python',side_effect=[info(),self.virtual()]),patch.object(runtime.subprocess,'run'):
            runtime.ensure_environment(self.root,self.target)
            download.assert_called_once_with(self.root)

    def test_failed_or_invalid_download_never_creates_environment(self):
        with patch.object(runtime,'installed_pythons',return_value=[]), \
             patch.object(runtime,'managed_python',side_effect=subprocess.CalledProcessError(1,'uv')):
            with self.assertRaises(subprocess.CalledProcessError):runtime.ensure_environment(self.root,self.target)
        with patch.object(runtime,'installed_pythons',return_value=[]), \
             patch.object(runtime,'managed_python',return_value='/invalid/python'), \
             patch.object(runtime,'probe_python',return_value=info(15)),patch.object(runtime.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'Downloaded interpreter failed'):runtime.ensure_environment(self.root,self.target)
            run.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_new_invalid_environment_is_preserved_for_inspection(self):
        def create(*args,**kwargs):
            self.target.mkdir();(self.target/'keep').write_text('inspect')
        with patch.object(runtime,'installed_pythons',return_value=['/installed/python']), \
             patch.object(runtime,'probe_python',side_effect=[info(),None]),patch.object(runtime.subprocess,'run',side_effect=create):
            with self.assertRaisesRegex(RuntimeError,'preserved'):runtime.ensure_environment(self.root,self.target)
        self.assertEqual((self.target/'keep').read_text(),'inspect')

    def test_managed_python_uses_only_project_directories_and_no_path_edits(self):
        managed=str(self.root/'.python/cpython/bin/python3')
        with patch.object(runtime.subprocess,'check_output',side_effect=['/project/uv\n',managed+'\n']), \
             patch.object(runtime.subprocess,'run') as run:
            self.assertEqual(runtime.managed_python(self.root),managed)
        env=run.call_args.kwargs['env']
        for key,folder in [('UV_PYTHON_INSTALL_DIR','.python'),('UV_CACHE_DIR','.bootstrap/cache'),('UV_PYTHON_BIN_DIR','.bootstrap/bin')]:
            self.assertEqual(env[key],str(self.root/folder))
        self.assertEqual(env['UV_PYTHON_INSTALL_BIN'],'0');self.assertEqual(env['UV_NO_MODIFY_PATH'],'1')

    def test_external_managed_directories_and_returned_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            for name in ['.python','.bootstrap']:
                link=self.root/name;link.symlink_to(outside)
                with patch.object(runtime.subprocess,'check_output') as execute:
                    with self.assertRaisesRegex(RuntimeError,'inside the serving project'):runtime.managed_python(self.root)
                    execute.assert_not_called()
                link.unlink()
        with patch.object(runtime.subprocess,'check_output',side_effect=['/project/uv\n','/external/python\n']), \
             patch.object(runtime.subprocess,'run'):
            with self.assertRaisesRegex(RuntimeError,'inside the serving project'):runtime.managed_python(self.root)

    def test_setup_installs_only_into_selected_environment_and_checks_pins(self):
        python=self.target/'bin/python'
        with patch.object(runtime_setup,'PROJECT_ROOT',self.root),patch.dict(os.environ,{'VENV':str(self.target),'BOOTSTRAP_PYTHON':'/chosen/python'}), \
             patch.object(runtime_setup,'ensure_environment',return_value=python) as ensure, \
             patch.object(runtime_setup.subprocess,'run') as run:
            runtime_setup.main()
            ensure.assert_called_once_with(self.root,self.target,'/chosen/python')
            self.assertTrue(all(call.args[0][0]==str(python) for call in run.call_args_list))
            self.assertIn('--no-deps',run.call_args_list[0].args[0])
            self.assertEqual(run.call_args_list[1].args[0],[str(python),'-m','pip','check'])

    def test_actual_probe_is_standard_library_only(self):
        actual=runtime.probe_python(sys.executable)
        self.assertEqual(actual['version'],list(sys.version_info[:3]))
        self.assertEqual(actual['implementation'],'cpython')
        self.assertTrue(Path(actual['base']).is_file())


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        shutil.copyfile(PROJECT/'bootstrap_python.sh',self.root/'bootstrap_python.sh')
        self.commands=self.root/'commands';self.commands.mkdir()
        for name in ['dirname','sh']:(self.commands/name).symlink_to(shutil.which(name))
        self.env={**os.environ,'PATH':str(self.commands),'BOOTSTRAP_PYTHON':'','VENV':''}

    def environment(self):
        python=self.root/'.venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
        return python

    def launch(self,mode,*args,**kwargs):
        return subprocess.run(['/bin/sh',str(self.root/'bootstrap_python.sh'),mode,*args],env=self.env,capture_output=True,text=True,**kwargs)

    def test_offline_launcher_reuses_project_without_versioned_global_command(self):
        python=self.environment();result=self.launch('run','-c','import sys;print(sys.executable)')
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(Path(result.stdout.strip()).resolve(),python.resolve())
        self.assertFalse((self.root/'.bootstrap').exists())

    def test_offline_mode_without_python_never_downloads(self):
        result=self.launch('run','-c','pass')
        self.assertNotEqual(result.returncode,0);self.assertIn('make setup',result.stderr)
        self.assertFalse((self.root/'.bootstrap').exists())

    def test_python_free_setup_bootstraps_locally_without_consuming_menu_input(self):
        managed=self.root/'.python/cpython/bin/python3';uv=self.root/'fake-uv'
        uv.write_text('#!/bin/sh\nif [ "$3" = install ]; then\n'
                      f'/bin/mkdir -p "{managed.parent}"\n/bin/ln -s "{sys.executable}" "{managed}"\n'
                      'printf "%s\\n" "$UV_PYTHON_INSTALL_DIR|$UV_PYTHON_BIN_DIR|$UV_CACHE_DIR|$UV_PYTHON_INSTALL_BIN|$UV_NO_MODIFY_PATH|$4" > "'+str(self.root/'uv-env')+'"\n'
                      'else\nprintf "%s\\n" "'+str(managed)+'"\nfi\n');uv.chmod(0o755)
        (self.root/'bootstrap_uv.sh').write_text('#!/bin/sh\nprintf "%s\\n" "'+str(uv)+'"\n')
        result=self.launch('setup','-c','print(input())',input='menu input survives\n')
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),'menu input survives')
        self.assertEqual((self.root/'uv-env').read_text().strip(),f'{self.root}/.python|{self.root}/.bootstrap/bin|{self.root}/.bootstrap/cache|0|1|3.14.9')

    def test_broken_target_with_no_python_is_preserved_without_downloading(self):
        target=self.root/'.venv';target.mkdir();(target/'keep').write_text('preserve')
        result=self.launch('setup','-c','pass')
        self.assertNotEqual(result.returncode,0);self.assertIn('preserved',result.stderr)
        self.assertFalse((self.root/'.bootstrap').exists());self.assertEqual((target/'keep').read_text(),'preserve')

    def test_explicit_interpreter_with_spaces_and_invalid_override(self):
        self.environment();python=self.root/'python with spaces';python.symlink_to(sys.executable)
        self.env['BOOTSTRAP_PYTHON']=str(python)
        result=self.launch('run','-c','print("chosen")');self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'chosen')
        self.env['BOOTSTRAP_PYTHON']='/missing/python'
        result=self.launch('setup','-c','pass');self.assertNotEqual(result.returncode,0)
        self.assertIn('BOOTSTRAP_PYTHON',result.stderr)

    def test_uv_checksum_mismatch_does_not_install_binary(self):
        shutil.copyfile(PROJECT/'bootstrap_uv.sh',self.root/'bootstrap_uv.sh')
        curl=self.commands/'curl';curl.write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\nshift\nprintf invalid > "$1"\n');curl.chmod(0o755)
        result=subprocess.run(['/bin/sh',str(self.root/'bootstrap_uv.sh')],env={**self.env,'PATH':str(self.commands)+':/usr/bin:/bin'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('checksum',result.stderr)
        self.assertEqual(list((self.root/'.bootstrap').iterdir()),[])

    def test_make_serve_and_download_use_existing_environment_offline(self):
        self.environment();shutil.copyfile(PROJECT/'Makefile',self.root/'Makefile');self.env.pop('VENV')
        self.env['BOOTSTRAP_PYTHON']='/missing/setup-only-python'
        (self.root/'bootstrap_python.sh').unlink()
        for name in ['launch.py','storage_paths.py','download.py']:
            (self.root/name).write_text('import json,sys;print(json.dumps([sys.argv[0],*sys.argv[1:]]))')
        make=shutil.which('make',path=os.defpath)
        for action in ['serve','download']:
            result=subprocess.run([make,action,'MODEL=z-image-turbo','NO_LLM=1'],cwd=self.root,env=self.env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((self.root/'.bootstrap').exists())
            self.assertNotIn('pip install',result.stdout)


if __name__=='__main__':unittest.main()
