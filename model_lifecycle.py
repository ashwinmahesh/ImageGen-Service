"""Nonfatal boundary timers; no inference or storage dependency in runtime children."""
from contextlib import contextmanager
import logging
import time
import uuid

_sink=lambda event:None


def configure(sink):
    global _sink
    _sink=sink


class Timing:
    def __init__(self,action,scope,reason,*,emit=None,clock=time.time,monotonic=time.monotonic):
        self.emit=emit or _sink;self.clock=clock;self.monotonic=monotonic;self.done=False
        self.started=clock();self.tick=monotonic()
        self.event={'id':uuid.uuid4().hex,'action':action,'scope':scope,'reason':reason,'started':self.started}
        self.report(phase='started')

    def report(self,**fields):
        try:self.emit({**self.event,**fields})
        except Exception:logging.getLogger(__name__).warning('Lifecycle telemetry unavailable; runtime continues',exc_info=True)

    def finish(self,status='succeeded'):
        if self.done:return
        self.done=True
        seconds=max(0,self.monotonic()-self.tick)
        self.report(phase='finished',status=status,finished=self.clock(),duration_seconds=seconds)


@contextmanager
def operation(action,scope,reason):
    timing=Timing(action,scope,reason)
    try:yield
    except BaseException:
        timing.finish('failed');raise
    else:timing.finish()
