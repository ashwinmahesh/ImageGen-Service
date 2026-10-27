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
from queue_storage import initialize_shared_queue,queue_key
from model_catalog import default_model,require_ready,configure_schema
from model_assets import available_models
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
    blocked_reason:str|None=None
    error:str|None=None
    error_status:int|None=None
    prompt_expansion:dict|None=None
    result:dict|None=Field(default=None,description='Saved generator metadata/checkpoint. May contain partial batch images before success. Use status to distinguish complete success. Contains server-local paths; the generator provides download URLs.')


class QueuePage(BaseModel):
    model:Model|None=None
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
    app=FastAPI(title='ImageGen Queue API',version='4.0.0',description=(
        'One durable cross-model queue, available without the generator. '
        'Omitted submission model uses the shared default; omitted list filter returns all models. Requests are validated by the selected '
        'generator when consumed; setup readiness is checked at admission. Model/adapter metadata can be discovered without inference. No inference or UI runs here. '
        'All SQLite state stays under the serving project. DELETE cancels waiting entries; '
        'running work, history and media are retained. Results contain server-local paths; '
        'image download URLs and the catalog are provided by the generator.'))
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

    @lru_cache(maxsize=1)
    def store():
        return initialize_shared_queue(project)
    app.state.store=store

    def ready(model=None):
        selected=model or default_model(project)
        if selected is None:raise HTTPException(422,'No models are set up. Run make setup first.')
        try:return require_ready(project,selected)
        except ValueError as error:raise HTTPException(422,str(error)) from None

    @app.exception_handler(QueueConflict)
    async def conflict(request,error):
        return JSONResponse({'detail':str(error)},status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def missing(request,error):
        return JSONResponse({'detail':'Unknown job ID'},status_code=404)

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(RuntimeError)
    async def unavailable(request,error):
        return JSONResponse({'detail':'Project-local queue storage is unavailable; check its path and permissions.'},status_code=503)

    def find(identifier):
        if not re.fullmatch('[a-f0-9]{32}',identifier):
            raise HTTPException(404,'Unknown job ID')
        item=store().get(identifier)
        return item

    def record(item,ready=None):
        item={**item,'model':item['request']['model']}
        if item['status']=='queued' and item['model'] not in (available_models(project) if ready is None else ready):
            item['blocked_reason']='Model not set up or storage unavailable; run make setup MODEL='+item['model']
        return item

    @app.get('/health')
    def health():
        return {'status':'ready','service':'imagegen-queue','project_dir':str(project),'default_model':default_model(project),'version':'4.0.0','queue_scope':'shared'}

    @app.get('/v1/models',summary='List setup model IDs')
    def models():
        return {'default_model':default_model(project),'data':[{'id':identifier,'name':profile['name'],
            'reference_mode':profile.get('reference_mode','variation'),'tasks':supported_tasks(profile),
            'mask_editing':bool(profile.get('mask_editing')),identifier:model_details(identifier)}
            for identifier in available_models(project) for profile in [PROFILES[identifier]]]}

    @app.get('/v1/loras',response_model=LoRAList,response_model_exclude_none=True,
             summary='List compatible LoRAs for a model',description=(
                 'Uses the shared default when model is omitted. Reads the local catalog for the selected model from its '
                 'adapter folder (or shipped public catalog if no override exists), filters compatibility and '
                 'reports local availability, default strength and trigger text. Use available:true entries '
                 'for request.lora. Does not load or hash model/adapter weights, start a generator or create '
                 'SQLite state. An unavailable configured media volume or unreadable catalog returns503; '
                 'queue CRUD remains available. No enabled flag is reported because generator startup '
                 'mode is independent; a base-only generator will leave adapter jobs waiting.'))
    def loras(model:Model|None=None):
        model=ready(model)
        try:
            directory,entries=model_loras(project,model)
            return LoRAList(model=model,directory=str(directory),data=entries)
        except RuntimeError as error:
            raise HTTPException(503,str(error)) from error
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(503,'Cannot read the LoRA catalog for this model; check its metadata, paths and permissions.') from error

    @app.post('/v1/jobs',response_model=QueueJob,status_code=202,summary='Commit a request to the shared queue',description=(
        'model (default: shared saved choice) labels a job in state/queue/jobs.sqlite3. Only setup models are accepted. request contains '
        'the image settings, such as prompt, size, seed, n and expand_prompt. With a reference, omitted size or size:auto detects source resolution (2048px long-edge cap) during consumption. For photo edits, include '
        'task:"edit", reference_image and optional preserve inside request. Qwen selected-area edits also '
        'accept request.mask_image: an absolute server-local mask path matching the oriented source photo '
        '(white edits, black preserves at output resolution). Klein supports whole-photo edits only. '
        'The queue persists these fields without reading media; model/media validation happens when consumed. Missing generation '
        'defaults and a random seed are frozen by the generator before rendering. '
        'generate_immediately:true inside request gives next-free-slot priority, without interrupting '
        'active work; default false joins the end of the queue. Idempotency-Key is scoped to the model: matching '
        'settings return the same ID, different settings return 409. 202 confirms the SQLite commit.'))
    def enqueue(payload:Submission,idempotency_key:str|None=Header(default=None,max_length=128,pattern=r'^[A-Za-z0-9._:-]+$')):
        selected=ready(payload.model)
        data=with_model(payload.request,selected)
        return record(store().enqueue(data,fingerprint(data),key=queue_key(selected,idempotency_key),adapter_sha=payload.adapter_sha256))

    @app.get('/v1/jobs',response_model=QueuePage,summary='List the shared queue with optional model filter')
    def listing(model:Model|None=None,limit:int=Query(24,ge=1,le=100),offset:int=Query(0,ge=0),status:Status|None=None):
        items,total=store().listing(limit,offset,status,model=model)
        following=offset+len(items);ready=available_models(project)
        return {'model':model,'data':[record(item,ready=ready) for item in items],'total':total,'limit':limit,'offset':offset,
                'has_more':following<total,'next_offset':following if following<total else None,
                'has_previous':offset>0,'previous_offset':max(0,offset-limit) if offset else None}

    @app.get('/v1/jobs/{identifier}',response_model=QueueJob,summary='Read a job and saved result by ID')
    def get(identifier:str):
        return record(find(identifier))

    @app.patch('/v1/jobs/{identifier}',response_model=QueueJob,summary='Replace settings of a never-started waiting job',description=(
        'Send the complete replacement request. Only queued entries with attempts=0 can be changed; '
        'running, interrupted and finished entries return 409. Model cannot be changed. Existing '
        'idempotency keys remain associated with the updated settings.'))
    def update(identifier:str,payload:Replacement):
        item=find(identifier)
        selected=ready(item['request']['model'])
        data=with_model(payload.request,selected)
        return record(store().update(identifier,data))

    @app.post('/v1/jobs/{identifier}/move',response_model=QueueJob,summary='Move a waiting job to a saved queue position',description=(
        'position is one-based among all waiting jobs, independent of pagination and priority. '
        'Only queued jobs can move (409 otherwise); an out-of-date position beyond the queue returns 409. '
        'Does not change request settings, immediate status or saved checkpoints.'))
    def move(identifier:str,payload:MoveJob):
        find(identifier)
        return record(store().move(identifier,payload.position))

    @app.post('/v1/jobs/{identifier}/run-now',response_model=QueueJob,summary='Move a waiting job to the next free slot',description=(
        'Moves the same job to position 1 and gives it immediate scheduling priority, including while paused. '
        'Never interrupts active inference or starts a generator. Only queued jobs are eligible (409 otherwise). '
        'Preserves request, seed, idempotency comparison and saved progress.'))
    def run_now(identifier:str):
        find(identifier)
        return record(store().run_now(identifier))

    @app.delete('/v1/jobs/{identifier}',response_model=QueueJob,summary='Cancel a waiting job; preserve its record')
    def cancel(identifier:str):
        find(identifier)
        return record(store().cancel(identifier))

    @app.delete('/v1/jobs',summary='Cancel waiting entries across models, or an optional model filter',description='Running and completed entries and their media are preserved.')
    def clear(model:Model|None=None):
        return {'cancelled':store().clear(model=model)}

    @app.post('/v1/jobs/{identifier}/retry',response_model=QueueJob,status_code=202,summary='Create a fresh ordinary job from a failed or cancelled entry')
    def retry(identifier:str):
        item=find(identifier)
        if item['status'] not in ('failed','cancelled'):
            raise QueueConflict('Only failed or cancelled jobs can be retried')
        ready(item['request']['model'])
        data={**item['request'],'generate_immediately':False}
        return record(store().enqueue(data,fingerprint(data),retry_of=identifier))

    @app.get('/v1/queue',summary='Read global counts and pause state')
    def summary():
        return {'queue_scope':'shared',**store().summary()}

    @app.patch('/v1/queue',summary='Pause or resume the shared queue',description='Persisted across process restarts. Immediate-priority entries can still run while ordinary queued work is paused.')
    def pause(payload:Pause):
        return {'queue_scope':'shared',**store().pause(payload.paused)}

    configure_schema(app,project,['Submission'])
    return app


def with_model(request,model):
    if 'model' in request and request['model']!=model:
        raise HTTPException(422,'request.model must match the requested model')
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
         'mask_image':'/absolute/path/mask.png','preserve':'Face, pose and background','size':'auto'}])


class Submission(Replacement):
    model:Model|None=Field(default=None,description='Requested setup model. Omitted uses the shared saved default. Unready models are rejected; GET /v1/models lists selectable models.')
    adapter_sha256:str|None=Field(default=None,pattern=r'^[a-f0-9]{64}$',description='Optional adapter checksum pinned by the generator admission API. Omit for normal queue submissions.')



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
