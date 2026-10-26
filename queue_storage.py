"""One project-local durable queue shared by all setup models."""
from pathlib import Path


def queue_database(project):
    project=Path(project).resolve()
    path=project/'state'/'queue'/'jobs.sqlite3'
    if not path.resolve().is_relative_to(project):
        raise RuntimeError('Queue state must stay inside the serving project; remove the external state symlink')
    return path


def queue_key(model,key):
    return f'{model}:{key}' if key is not None else None


def initialize_shared_queue(project):
    from job_queue import JobStore
    store=JobStore(queue_database(project));store.initialize()
    return store
