"""Read-only observability routes, separate from queue state and admission."""
import math
from pathlib import Path
import sqlite3
import time
from typing import Literal
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from model_profiles import PROFILES
from observability import WINDOWS


class MetricSummary(BaseModel):
    mean:float|None
    peak:float|None
    count:int


class ResourceSummary(BaseModel):
    sample_count:int
    first_sample:float|None
    last_sample:float|None
    metrics:dict[str,MetricSummary]


class ResourcePoint(BaseModel):
    timestamp:float
    first_sample:float
    last_sample:float
    count:int
    models:list[str]
    sessions:list[str]
    stages:list[str]
    values:dict[str,float|None]=Field(description='Sample means in each time bucket. CPU/GPU values are percentages; memory is bytes. Unavailable values are null, never synthetic zero.')
    peaks:dict[str,float|None]=Field(description='Maximum sampled values in this bucket; these are sampled peaks, not guaranteed instantaneous peaks.')


class ResourceSeries(BaseModel):
    start:float
    end:float
    bucket_seconds:float
    points:list[ResourcePoint]
    summary:ResourceSummary


class ObservedRun(BaseModel):
    session:str
    job_id:str
    attempt:int
    model:str
    started:float
    finished:float|None
    status:Literal['running','succeeded','failed','interrupted']
    summary:ResourceSummary


class ObservedJobs(BaseModel):
    data:list[ObservedRun]
    total:int
    limit:int
    offset:int
    has_more:bool


class ObservedJob(ResourceSeries):
    job_id:str
    runs:list[ObservedRun]


def create_router(get_monitor,project):
    router=APIRouter(tags=['Observability'])
    project=Path(project)

    def monitor():
        value=get_monitor()
        if value is None:
            raise HTTPException(503,detail={'code':'observability_unavailable','message':'Resource sampling has not started.'},headers={'Retry-After':'2'})
        return value

    def checked_model(model):
        if model is not None and model not in PROFILES:raise HTTPException(422,'Unknown model; choose '+', '.join(PROFILES))
        return model

    def read(action):
        try:return action()
        except FileNotFoundError as error:raise HTTPException(404,str(error)) from None
        except (sqlite3.Error,RuntimeError,ValueError):
            raise HTTPException(503,detail={'code':'observability_unavailable','message':'Resource history is unavailable; check the project state directory and disk space. Generation and queue operations are independent.'},headers={'Retry-After':'2'}) from None

    @router.get('/observability',include_in_schema=False)
    def page():
        return FileResponse(project/'ui/observability.html',media_type='text/html',headers={'Cache-Control':'no-store'})

    @router.get('/assets/observability.js',include_in_schema=False)
    def script():
        return FileResponse(project/'ui/assets/observability.js',media_type='text/javascript',headers={'Cache-Control':'no-store'})

    @router.get('/assets/observability.css',include_in_schema=False)
    def styles():
        return FileResponse(project/'ui/assets/observability.css',media_type='text/css',headers={'Cache-Control':'no-store'})

    @router.get('/v1/observability',summary='Resource sampling status and metric definitions')
    def status():
        value=monitor()
        return {**value.status(),'models':[{'id':key,'name':profile['name']} for key,profile in PROFILES.items()],
                'windows':list(WINDOWS),'scopes':{
                    'service_cpu_percent':'Generator and child processes, normalized to total logical CPU capacity (0–100%). Independent queue API excluded.',
                    'service_rss_bytes':'Summed resident memory of the generator and its child processes; shared pages can be counted more than once.',
                    'system_cpu_percent':'Whole-machine CPU utilization.',
                    'system_memory_used_bytes':'Whole-machine total RAM minus available RAM.',
                    'gpu_utilization_percent':'Whole-device GPU utilization from Apple IORegistry, not per-process attribution.',
                    'gpu_memory_bytes':'Whole-device GPU driver In use system memory, sharing unified RAM; do not add this to RSS/system RAM.'}}

    @router.get('/v1/observability/metrics',response_model=ResourceSeries,summary='CPU, memory and GPU history',
                description='Across all job models/sessions by default. Optional model filter. Samples are kept for eight days and bucketed into bounded means and sampled peaks. Null means unavailable; missing buckets mean no samples, including stopped generators. No backfill or implicit model load.')
    def metrics(window:Literal['5m','30m','1h','6h','24h','3d','7d']='5m',model:str|None=None,
                end:float|None=Query(default=None,gt=0),max_points:int=Query(600,ge=30,le=1200)):
        checked_model(model);end=time.time() if end is None else end
        if not math.isfinite(end) or end>time.time()+5:raise HTTPException(422,'end must be a finite current or past Unix timestamp')
        return read(lambda:monitor().store.series(end-WINDOWS[window],end,model=model,max_points=max_points))

    @router.get('/v1/observability/jobs',response_model=ObservedJobs,summary='List measured job attempts',
                description='Newest measured attempt first; includes model, job ID, session, attempt, execution status and sampled means/peaks. Does not include waiting jobs or prompts/media. Completed summaries persist after raw samples expire.')
    def jobs(model:str|None=None,limit:int=Query(20,ge=1,le=100),offset:int=Query(0,ge=0)):
        checked_model(model)
        return read(lambda:monitor().store.jobs(model=model,limit=limit,offset=offset))

    @router.get('/v1/observability/jobs/{identifier}',response_model=ObservedJob,summary='Resource history for one image job',
                description='The default full window covers measured execution attempts. Only execution is sampled; waiting/restart periods between attempts appear as gaps. Short windows end at job completion, or now while running. Includes all recorded attempts/sessions. Raw points expire after eight days; each completed run retains its summary. 404 means no measured execution yet, including older/unstarted jobs.')
    def job(identifier:str,model:str|None=None,window:Literal['full','1m','5m','15m']='full',max_points:int=Query(600,ge=30,le=1200)):
        checked_model(model)
        if len(identifier)!=32 or any(char not in '0123456789abcdef' for char in identifier):raise HTTPException(404,'Unknown job ID')
        return read(lambda:monitor().store.job(identifier,model=model,window=window,max_points=max_points))

    return router
