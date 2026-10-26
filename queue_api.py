"""Always-available SQLite CRUD API. No model assets, inference, worker or UI."""
import argparse
from functools import lru_cache
from pathlib import Path
import re
import sqlite3
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from job_queue import JobStore, QueueConflict, fingerprint
from model_profiles import PROFILES, DEFAULT_MODEL, model_details
from photo_workflows import supported_tasks
from queue_storage import queue_database
from adapter_catalog import model_loras

PROJECT_ROOT=Path(__file__).resolve().parent
DEFAULT_QUEUE_PORT=55302
Model=Literal[tuple(PROFILES)]
Status=Literal['pending','queued','running','succeeded','failed','cancelled']


class QueueJob(BaseModel):
    id:str
    model:Model
    status:Literal['queued','running','succeeded','failed','cancelled']
    sequence:int
    queue_position:int|None=Field(default=None,description="One-based position among all waiting jobs; null once running or finished.")
    request:dict=Field(description='Original queued settings, then frozen effective defaults/seed once consumed.')
    generate_immediately:bool
    created:float
    started:float|None=None
    finished:float|None=None
    attempts:int
    retry_of:str|None=None
    error:str|None=None
    error_status:int|None=None
    prompt_expansion:dict|None=None
    result:dict|None=Field(default=None,description='Saved generator metadata/checkpoint. May contain partial batch images before success. Use status to distinguish complete success. Contains server-local paths; the generator provides download URLs.')


class QueuePage(BaseModel):
    model:Model
    data:list[QueueJob]
    total:int
    limit:int
    offset:int
    has_more:bool
    next_offset:int|None=None
    has_previous:bool
    previous_offset:int|None=None


class LoRAEntry(BaseModel):
    model_config=ConfigDict(extra='allow')
    id:str=Field(description='Adapter ID to pass as request.lora. Use null or omit lora for the base model.')
    name:str
    filename:str
    models:list[str]
    available:bool=Field(description='Compatible catalog entry has a local weights file and recorded checksum. Presence check only; inference verifies the checksum before use.')
    default_strength:float|None=None
    trigger_prompt:str=''
    source:str|None=None
    sha256:str|None=None


class LoRAList(BaseModel):
    object:Literal['list']='list'
    model:Model
    directory:str=Field(description='Resolved adapter directory on the server. Chroma HD and Flash share the HD folder.')
    data:list[LoRAEntry]


def create_app(project=PROJECT_ROOT):
    project=Path(project).resolve()
    app=FastAPI(title='ImageGen Queue API',version='3.0.0',description=(
        'Durable model-scoped queue CRUD, available without the generator or model weights. '
        'An omitted model selects z-image-turbo on every queue operation. Requests are validated by the selected '
        'generator when consumed, not by this lightweight API. Model/adapter metadata can be discovered without inference. No inference or UI runs here. '
        'All SQLite state stays under the serving project. DELETE cancels waiting entries; '
        'running work, history and media are retained. Results contain server-local paths; '
        'image download URLs and the catalog are provided by the generator.'))
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

    @lru_cache(maxsize=len(PROFILES))
    def store(model):
        value=JobStore(queue_database(project,model))
        value.initialize()  # Never recovers running jobs; only the generator holding its lease may do that.
        return value
    app.state.store=store

    @app.exception_handler(QueueConflict)
    async def conflict(request,error):
        return JSONResponse({'detail':str(error)},status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def missing(request,error):
        return JSONResponse({'detail':'Unknown job ID in this model queue'},status_code=404)

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(RuntimeError)
    async def unavailable(request,error):
        return JSONResponse({'detail':'Project-local queue storage is unavailable; check its path and permissions.'},status_code=503)

    def find(model,identifier):
        if not re.fullmatch('[a-f0-9]{32}',identifier):
            raise HTTPException(404,'Unknown job ID')
        return store(model).get(identifier)

    def record(item):
        return {**item,'model':item['request']['model']}

    @app.get('/health')
    def health():
        return {'status':'ready','service':'imagegen-queue','project_dir':str(project),'default_model':DEFAULT_MODEL,'version':'3.0.0'}

    @app.get('/v1/models',summary='List supported queue model IDs')
    def models():
        return {'default_model':DEFAULT_MODEL,'data':[{'id':identifier,'name':profile['name'],
            'reference_mode':profile.get('reference_mode','variation'),'tasks':supported_tasks(profile),
            'mask_editing':bool(profile.get('mask_editing')),identifier:model_details(identifier)}
            for identifier,profile in PROFILES.items()]}

    @app.get('/v1/loras',response_model=LoRAList,response_model_exclude_none=True,
             summary='List compatible LoRAs for a model',description=(
                 'Defaults to z-image-turbo when model is omitted. Reads the local catalog for the selected model from its '
                 'adapter folder (or shipped public catalog if no override exists), filters compatibility and '
                 'reports local availability, default strength and trigger text. Use available:true entries '
                 'for request.lora. Does not load or hash model/adapter weights, start a generator or create '
                 'SQLite state. An unavailable configured media volume or unreadable catalog returns503; '
                 'queue CRUD remains available. No enabled flag is reported because generator startup '
                 'mode is independent; a base-only generator will leave adapter jobs waiting.'))
    def loras(model:Model=DEFAULT_MODEL):
        try:
            directory,entries=model_loras(project,model)
            return LoRAList(model=model,directory=str(directory),data=entries)
        except RuntimeError as error:
            raise HTTPException(503,str(error)) from error
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(503,'Cannot read the LoRA catalog for this model; check its metadata, paths and permissions.') from error

    @app.post('/v1/jobs',response_model=QueueJob,status_code=202,summary='Commit a request to its model queue',description=(
        'model (default z-image-turbo) routes the entry to state/queues/<model>/jobs.sqlite3. request contains '
        'the image settings, such as prompt, size, seed, n and expand_prompt. For photo edits, include '
        'task:"edit", reference_image and optional preserve inside request. Qwen selected-area edits also '
        'accept request.mask_image: an absolute server-local mask path matching the oriented source photo '
        '(white edits, black preserves at output resolution). Klein supports whole-photo edits only. '
        'The queue persists these fields without reading media; model/media validation happens when consumed. Missing generation '
        'defaults and a random seed are frozen by the generator before rendering. '
        'generate_immediately:true inside request gives next-free-slot priority, without interrupting '
        'active work; default false joins the end of the queue. Idempotency-Key is scoped to the model: matching '
        'settings return the same ID, different settings return 409. 202 confirms the SQLite commit.'))
    def enqueue(payload:Submission,idempotency_key:str|None=Header(default=None,max_length=128,pattern=r'^[A-Za-z0-9._:-]+$')):
        data=payload.data()
        return record(store(payload.model).enqueue(data,fingerprint(data),key=idempotency_key,adapter_sha=payload.adapter_sha256))

    @app.get('/v1/jobs',response_model=QueuePage,summary='List one model queue with pagination')
    def listing(model:Model=DEFAULT_MODEL,limit:int=Query(24,ge=1,le=100),offset:int=Query(0,ge=0),status:Status|None=None):
        items,total=store(model).listing(limit,offset,status)
        following=offset+len(items)
        return {'model':model,'data':[record(item) for item in items],'total':total,'limit':limit,'offset':offset,
                'has_more':following<total,'next_offset':following if following<total else None,
                'has_previous':offset>0,'previous_offset':max(0,offset-limit) if offset else None}

    @app.get('/v1/jobs/{identifier}',response_model=QueueJob,summary='Read a job and saved result from its model queue')
    def get(identifier:str,model:Model=DEFAULT_MODEL):
        return record(find(model,identifier))

    @app.patch('/v1/jobs/{identifier}',response_model=QueueJob,summary='Replace settings of a never-started waiting job',description=(
        'Send the complete replacement request. Only queued entries with attempts=0 can be changed; '
        'running, interrupted and finished entries return 409. Model cannot be changed. Existing '
        'idempotency keys remain associated with the updated settings.'))
    def update(identifier:str,payload:Replacement,model:Model=DEFAULT_MODEL):
        find(model,identifier)
        data=with_model(payload.request,model)
        return record(store(model).update(identifier,data))

    @app.post('/v1/jobs/{identifier}/move',response_model=QueueJob,summary='Move a waiting job to a saved queue position',description=(
        'position is one-based among all waiting jobs, independent of pagination and priority. '
        'Only queued jobs can move (409 otherwise); an out-of-date position beyond the queue returns 409. '
        'Does not change request settings, immediate status or saved checkpoints.'))
    def move(identifier:str,payload:MoveJob,model:Model=DEFAULT_MODEL):
        find(model,identifier)
        return record(store(model).move(identifier,payload.position))

    @app.post('/v1/jobs/{identifier}/run-now',response_model=QueueJob,summary='Move a waiting job to the next free slot',description=(
        'Moves the same job to position 1 and gives it immediate scheduling priority, including while paused. '
        'Never interrupts active inference or starts a generator. Only queued jobs are eligible (409 otherwise). '
        'Preserves request, seed, idempotency comparison and saved progress.'))
    def run_now(identifier:str,model:Model=DEFAULT_MODEL):
        find(model,identifier)
        return record(store(model).run_now(identifier))

    @app.delete('/v1/jobs/{identifier}',response_model=QueueJob,summary='Cancel a waiting job; preserve its record')
    def cancel(identifier:str,model:Model=DEFAULT_MODEL):
        find(model,identifier)
        return record(store(model).cancel(identifier))

    @app.delete('/v1/jobs',summary='Cancel all waiting entries for one model',description='Running and completed entries and their media are preserved.')
    def clear(model:Model=DEFAULT_MODEL):
        return {'cancelled':store(model).clear()}

    @app.post('/v1/jobs/{identifier}/retry',response_model=QueueJob,status_code=202,summary='Create a fresh ordinary job from a failed or cancelled entry')
    def retry(identifier:str,model:Model=DEFAULT_MODEL):
        item=find(model,identifier)
        if item['status'] not in ('failed','cancelled'):
            raise QueueConflict('Only failed or cancelled jobs can be retried')
        data={**item['request'],'generate_immediately':False}
        return record(store(model).enqueue(data,fingerprint(data),retry_of=identifier))

    @app.get('/v1/queue',summary='Read persisted counts and pause state for one model')
    def summary(model:Model=DEFAULT_MODEL):
        return {'model':model,**store(model).summary()}

    @app.patch('/v1/queue',summary='Pause or resume a model queue',description='Persisted across process restarts. Immediate-priority entries can still run while ordinary queued work is paused.')
    def pause(payload:Pause,model:Model=DEFAULT_MODEL):
        return {'model':model,**store(model).pause(payload.paused)}

    return app


def with_model(request,model):
    if 'model' in request and request['model']!=model:
        raise HTTPException(422,'request.model must match the queue model')
    data={**request,'model':model}
    immediate=data.setdefault('generate_immediately',False)
    if type(immediate) is not bool:
        raise HTTPException(422,'generate_immediately must be a boolean')
    try:
        fingerprint(data)  # Reject non-finite values before a SQLite transaction.
    except (ValueError,TypeError):
        raise HTTPException(422,'request must contain finite JSON values') from None
    return data


class Replacement(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request:dict=Field(description='Complete image request. Model defaults and validation are applied by the generator when consumed. Photo edits use task:"edit", reference_image, optional preserve, and Qwen-only mask_image (white edits, black preserves). Paths must remain accessible on the generator Mac until completion and recovery.',examples=[
        {'prompt':'A lantern in a misty forest','size':'1024x1024','n':1,'expand_prompt':True},
        {'task':'edit','prompt':'Change the jacket to green.','reference_image':'/absolute/path/photo.png',
         'mask_image':'/absolute/path/mask.png','preserve':'Face, pose and background','size':'1024x1024'}])


class Submission(Replacement):
    model:Model=Field(default=DEFAULT_MODEL,description='Queue/model ID; omitted uses z-image-turbo regardless of the active generator. May be enqueued before its weights are set up.')
    adapter_sha256:str|None=Field(default=None,pattern=r'^[a-f0-9]{64}$',description='Optional adapter checksum pinned by the generator admission API. Omit for normal queue submissions.')

    def data(self):
        return with_model(self.request,self.model)


class MoveJob(BaseModel):
    model_config=ConfigDict(extra='forbid')
    position:int=Field(ge=1,strict=True,description='One-based destination among all waiting jobs, not the current page. Read queue_position from the list; 1 is next.')


class Pause(BaseModel):
    model_config=ConfigDict(extra='forbid')
    paused:bool=Field(strict=True)


app=create_app()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=DEFAULT_QUEUE_PORT)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
