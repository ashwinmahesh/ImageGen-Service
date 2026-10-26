"""Small localhost HTTP client for generator-facing queue CRUD."""
import json
import urllib.error
import urllib.parse
import urllib.request
from fastapi import HTTPException
from queue_api import DEFAULT_QUEUE_PORT


class QueueClient:
    def __init__(self,port=DEFAULT_QUEUE_PORT,*,transport=None):
        self.url=f'http://127.0.0.1:{port}'
        self.transport=transport

    def call(self,method,path,*,model=None,payload=None,key=None,query=None):
        params={**(query or {}),**({'model':model} if model else {})}
        if params:
            path+='?'+urllib.parse.urlencode({k:v for k,v in params.items() if v is not None})
        headers={'Content-Type':'application/json'}
        if key: headers['Idempotency-Key']=key
        if self.transport is not None:  # In-process HTTP transport for integration tests.
            response=self.transport.request(method,path,json=payload,headers=headers)
            if response.status_code>=400:
                raise HTTPException(response.status_code,response.json().get('detail','Queue request failed'))
            return response.json()
        request=urllib.request.Request(self.url+path,method=method,headers=headers,
                                      data=json.dumps(payload,allow_nan=False).encode() if payload is not None else None)
        try:
            with urllib.request.urlopen(request,timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try: detail=json.load(error).get('detail','Queue request failed')
            except (ValueError,OSError): detail='Queue request failed'
            raise HTTPException(error.code,detail) from error
        except (OSError,ValueError) as error:
            raise HTTPException(503,'Queue API is unavailable. Run make queue or restart with make serve; accepted jobs remain in SQLite.') from error
