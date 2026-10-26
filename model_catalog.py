"""Setup-only discovery and a shared default, without inference imports."""
from contextlib import closing
import json
import sqlite3
from model_assets import available_models,check_model_assets
from model_profiles import DEFAULT_MODEL,PROFILES
from queue_storage import queue_database


def default_model(project,ready=None):
    ready=available_models(project) if ready is None else ready
    saved=None
    path=queue_database(project)
    if path.is_file():
        try:
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.25)) as db:
                row=db.execute("SELECT value FROM settings WHERE key='default_model'").fetchone()
                saved=json.loads(row[0]) if row else None
        except (sqlite3.Error,ValueError):pass
    return saved if saved in ready else DEFAULT_MODEL if DEFAULT_MODEL in ready else next(iter(ready),None)


def require_ready(project,model):
    if model not in PROFILES:raise ValueError('Unknown model')
    try:check_model_assets(project,model)
    except (SystemExit,RuntimeError,OSError,ValueError) as error:
        raise ValueError(f'{PROFILES[model]["name"]} is not set up or its storage is unavailable. Run make setup MODEL={model}. {error}') from None
    return model


def configure_schema(app,project,request_names,default_provider=None):
    """Swagger choices use current setup readiness, just like admission/UI."""
    from copy import deepcopy
    from fastapi.openapi.utils import get_openapi
    def schema():
        if app.openapi_schema is None:
            app.openapi_schema=get_openapi(title=app.title,version=app.version,description=app.description,routes=app.routes)
        result=deepcopy(app.openapi_schema);ready=available_models(project)
        selected=default_provider() if default_provider else default_model(project,ready)
        def model_choices(node):
            if 'enum' in node:node['enum']=ready
            for variant in node.get('anyOf',[]):model_choices(variant)
        for name in request_names:
            field=result.get('components',{}).get('schemas',{}).get(name,{}).get('properties',{}).get('model')
            if field is not None:
                model_choices(field)
                if selected in ready:
                    field['default']=selected
                    properties=result['components']['schemas'][name]['properties']
                    if 'steps' in properties:properties['steps']['default']=PROFILES[selected]['steps']
                    if 'output_dir' in properties:
                        from storage_paths import data_directory
                        properties['output_dir']['default']=str(data_directory(project)/selected/'images')
                else:field.pop('default',None)
        for path in result['paths'].values():
            for route in path.values():
                if not isinstance(route,dict):continue
                for parameter in route.get('parameters',[]):
                    if parameter.get('name')=='model':model_choices(parameter['schema'])
        return result
    app.openapi=schema
