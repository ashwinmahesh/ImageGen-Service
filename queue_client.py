"""On-demand queue API health. Generator operations access shared SQLite directly."""
import http.client
import json
from pathlib import Path
import urllib.error
import urllib.request
from queue_api import DEFAULT_QUEUE_PORT


class QueueClient:
    def __init__(self,port=DEFAULT_QUEUE_PORT,*,transport=None,project=None):
        self.url=f'http://127.0.0.1:{port}'
        self.transport=transport
        self.project=str(Path(project).resolve()) if project is not None else None
        self.recovery_command='make queue'+(f' QUEUE_PORT={port}' if port!=DEFAULT_QUEUE_PORT else '')

    def status(self):
        """Bounded availability check; never restarts services or gates local work."""
        state={'url':self.url,'available':False,'recovery_command':self.recovery_command,
               'error':f'Standalone queue API unavailable. Run {self.recovery_command} for agent requests. This page and already accepted jobs can continue.'}
        try:
            if self.transport is not None:  # In-process HTTP transport for tests.
                response=self.transport.get('/health')
                status,raw=response.status_code,response.content
            else:
                with urllib.request.urlopen(self.url+'/health',timeout=1) as response:
                    status,raw=response.status,response.read(65536)
            health=json.loads(raw)
            if status!=200 or not isinstance(health,dict) or health.get('service')!='imagegen-queue' or health.get('status')!='ready':
                return state
            if self.project is not None and health.get('project_dir')!=self.project:
                return state
        except urllib.error.HTTPError as error:
            error.close()
            return state
        except (OSError,http.client.HTTPException,ValueError,TypeError):
            return state
        return {**state,'available':True,'error':None}
