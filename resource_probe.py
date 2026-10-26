"""Bounded, read-only CPU/RSS and Apple GPU counters; never load an ML runtime."""
import importlib
import os
from pathlib import Path
import platform
import plistlib
import subprocess
import time

FIELDS=('service_cpu_percent','system_cpu_percent','service_rss_bytes',
        'system_memory_used_bytes','system_memory_total_bytes','gpu_utilization_percent','gpu_memory_bytes')


def gpu_counters():
    if platform.system()!='Darwin':
        return None,None
    result=subprocess.run(['/usr/sbin/ioreg','-r','-c','IOAccelerator','-d','1','-a'],
                          capture_output=True,timeout=1,check=True)
    if len(result.stdout)>2*1024**2:
        raise ValueError('GPU counter response too large')
    values=[]
    def visit(node):
        if isinstance(node,list):
            for child in node:visit(child)
        elif isinstance(node,dict):
            stats=node.get('PerformanceStatistics')
            if isinstance(stats,dict):values.append(stats)
            visit(node.get('IORegistryEntryChildren',[]))
    visit(plistlib.loads(result.stdout))
    utilization=[item['Device Utilization %'] for item in values
                 if type(item.get('Device Utilization %')) in (float,int) and 0<=item['Device Utilization %']<=100]
    memory=[item['In use system memory'] for item in values
            if type(item.get('In use system memory')) in (float,int) and item['In use system memory']>=0]
    return (max(utilization) if utilization else None,sum(memory) if memory else None)


class ResourceProbe:
    def __init__(self,project,*,pid=None,clock=time.monotonic,wall=time.time,gpu=gpu_counters):
        self.project=Path(project);self.pid=pid or os.getpid()
        self.clock,self.wall,self.gpu=clock,wall,gpu
        self.previous=None;self.gpu_retry=0
        self.status={'cpu_memory':'starting','gpu':'starting','process_tree_partial':False}

    def collect(self):
        values=dict.fromkeys(FIELDS)
        now=self.clock();wall=self.wall();partial=False;count=0
        try:
            psutil=importlib.import_module('psutil')
            memory=psutil.virtual_memory()
            values.update(system_memory_total_bytes=memory.total,system_memory_used_bytes=memory.total-memory.available)
            system=psutil.cpu_times()._asdict()
            total=sum(value for name,value in system.items() if name not in ('guest','guest_nice'))
            idle=system.get('idle',0)+system.get('iowait',0)
            parent=psutil.Process(self.pid)
            processes=[parent,*parent.children(recursive=True)]
            current={};rss=0
            for process in processes:
                try:
                    with process.oneshot():
                        # A detached queue API can remain a child of the launcher.
                        # It is independent of generation and excluded here.
                        if process.pid!=self.pid and str(self.project/'queue_api.py') in process.cmdline():
                            continue
                        key=(process.pid,process.create_time())
                        cpu=process.cpu_times()
                        current[key]=cpu.user+cpu.system
                        rss+=process.memory_info().rss;count+=1
                except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
                    partial=True
            values['service_rss_bytes']=rss if count else None
            if self.previous:
                previous_time,previous_wall,previous_total,previous_idle,previous=self.previous
                elapsed=now-previous_time;delta=total-previous_total
                if delta>0:
                    values['system_cpu_percent']=max(0,min(100,100*(delta-(idle-previous_idle))/delta))
                if elapsed>0 and count:
                    seconds=sum(max(0,value-previous[key]) if key in previous else
                                (value if key[1]>=previous_wall else 0) for key,value in current.items())
                    values['service_cpu_percent']=max(0,min(100,100*seconds/elapsed/(psutil.cpu_count() or 1)))
            self.previous=(now,wall,total,idle,current)
            self.status['cpu_memory']='available'
        except ImportError:
            self.status['cpu_memory']='Unavailable: run make setup-env and restart to install psutil.'
        except Exception as error:
            self.status['cpu_memory']='Unavailable: '+type(error).__name__
        self.status['process_tree_partial']=partial
        if now>=self.gpu_retry:
            try:
                values['gpu_utilization_percent'],values['gpu_memory_bytes']=self.gpu()
                self.status['gpu']='available' if values['gpu_utilization_percent'] is not None else 'GPU counters unavailable on this system.'
                if values['gpu_utilization_percent'] is None:self.gpu_retry=now+30
            except (OSError,ValueError,subprocess.SubprocessError,plistlib.InvalidFileException):
                self.status['gpu']='GPU counters unavailable; no administrator access is requested.'
                self.gpu_retry=now+30
        values.update(process_count=count,process_tree_partial=partial)
        return values
