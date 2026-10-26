"""Choose one model, then provision it through the same Makefile as scripts."""
import os
import signal
import subprocess
import sys
from pathlib import Path
from model_profiles import select_model
from model_assets import available_models

ROOT=Path(__file__).resolve().parent
CANCEL_MESSAGE='Setup cancelled. Existing files and partial downloads were kept; rerun make setup to continue.'


def stop_setup(process):
    """Stop only our isolated setup group, including pip/download descendants."""
    previous=signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # A make/shell parent may exit before a download grandchild. Stop any
        # survivors in our group even when the direct child already exited.
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        signal.signal(signal.SIGINT,previous)


def completed_models():
    try:
        return available_models(ROOT)
    except (OSError,RuntimeError,ValueError):
        # Unreadable or unavailable storage cannot establish completion. Let the
        # normal storage setup step explain or reconfigure it after selection.
        return ()


def main():
    process=None
    try:
        print('Setup is safe to rerun: unchanged model weights need no redownload; missing or changed pinned files are updated in place. Dependencies and shared assets are still checked.',flush=True)
        print('Press Ctrl+C at any time to cancel setup, or choose q at the model menu.',flush=True)
        selected=select_model(setup_complete=completed_models,allow_cancel=True)
        print(f'Setting up {selected}',flush=True)
        env={**os.environ,'IMAGE_MODEL':selected}
        # The parent receives terminal Ctrl+C and stops this group itself, so
        # nested Python/download tools do not each emit interrupt tracebacks.
        # Defer interrupts across spawn until we own its process handle.
        pending=[]
        previous=signal.signal(signal.SIGINT,lambda *_:pending.append(True))
        try:
            process=subprocess.Popen(['make','setup-selected'],cwd=ROOT,env=env,start_new_session=True)
        finally:
            signal.signal(signal.SIGINT,previous)
        if pending:
            raise KeyboardInterrupt
        return process.wait()
    except (KeyboardInterrupt,EOFError):
        if process is not None:
            stop_setup(process)
        print('\n'+CANCEL_MESSAGE,flush=True)
        return 130
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__=='__main__':
    sys.exit(main())
