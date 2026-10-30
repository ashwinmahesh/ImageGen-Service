(() => {
  'use strict';
  const $=id=>document.getElementById(id),modelNames=new Map();
  let callbacks={},statusHoldUntil=0;
  const QUEUE_PAGE_SIZE=20;
  const CANCEL_RUNNING_PROMPT='Stop this job now? The image being rendered is discarded; finished options in a batch are kept.';
  function setStatus(text,state){$('status-text').textContent=text;$('status').dataset.state=state;}
  function remainingLabel(seconds){
    if(seconds==null)return '';
    return seconds<60 ? ` · about ${Math.max(5,Math.round(seconds/5)*5)} sec left` : ` · about ${Math.round(seconds/60)} min left`;
  }
  function progressLabel(progress){
    if(!progress)return '';
    if(progress.phase==='sampling')return `${progress.percent}%${remainingLabel(progress.estimated_remaining_seconds)}`;
    if(progress.phase==='finishing')return `${progress.percent}% · saving the image`;
    if(progress.phase==='loading_model')return `${progress.percent}% · loading the model`;
    return `${progress.percent}%`;
  }
  function setProgressBar(bar,progress){
    const percent=Math.max(0,Math.min(100,progress?.percent ?? 0));
    bar.firstElementChild.style.width=percent+'%';
    bar.setAttribute('aria-valuenow',String(percent));bar.setAttribute('aria-valuetext',`${percent} percent`);
  }
  function progressBar(progress,label){
    const bar=document.createElement('div');bar.className='job-progress';bar.setAttribute('role','progressbar');
    bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax','100');bar.setAttribute('aria-label',label);
    bar.append(document.createElement('span'));setProgressBar(bar,progress);return bar;
  }
  function updateHealth(data){
    const name=modelNames.get(data.active_model)||data.active_model||'queued image',paused=!!data.queue?.paused;
    const percent=data.progress && data.progress.phase!=='done' ? ` ${data.progress.percent}%` : '';
    // Keep a failed pause/resume message visible briefly instead of replacing it on the next poll.
    if (Date.now()>=statusHoldUntil) setStatus(data.busy ? data.queue?.stage==='waiting_for_accelerator' ? 'Waiting for GPU…' : `${data.queue?.stage==='loading_model' ? 'Loading' : 'Processing'} ${name}…${percent}` : data.status==='ready' ? paused ? 'Queue paused' : 'Ready on your Mac' : data.status==='error' ? 'Server unhealthy' : 'Starting…',data.busy ? 'busy' : data.status==='ready' ? paused ? 'paused' : 'ready' : data.status==='error' ? 'offline' : 'checking');
    callbacks.onHealth?.(data);
  }
  function queueError(message){$('queue-error').textContent=message;$('queue-error').hidden=false;}
  async function inspectJob(job){
    $('queue-dialog').close();
    if(callbacks.onInspect){await callbacks.onInspect(job);return;}
    location.assign('/?'+new URLSearchParams({job:job.id}));
  }
    const queueIconPaths={
 queue:'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
 monitor:'M4 4h16v12H4zM8 21h8M12 16v5',
 play:'m8 5 11 7-11 7Z',
 remove:'M3 6h18M9 6V4h6v2M5 6l1 14h12l1-14M10 10v6M14 10v6',
 clear:'M3 6h18M3 12h9M3 18h9m5-6 5 6m0-6-5 6',
 close:'m6 6 12 12M18 6 6 18',
 stop:'M7 7h10v10H7z',
 previous:'m14 6-6 6 6 6',next:'m10 6 6 6-6 6',
 resources:'M4 3v18h17M8 16v-5M13 16V7M18 16v-8',
 retry:'M3 10a9 9 0 1 1 1 8M3 4v6h6',
 grip:'M8 5h.01M16 5h.01M8 12h.01M16 12h.01M8 19h.01M16 19h.01'
};
function queueIcon(name){
 const icon=document.createElementNS('http://www.w3.org/2000/svg','svg');
 for(const[key,value]of Object.entries({viewBox:'0 0 24 24',fill:'none',stroke:'currentColor','stroke-width':name==='grip'?'3':'1.6','stroke-linecap':'round','stroke-linejoin':'round','aria-hidden':'true',focusable:'false',class:'queue-icon'}))icon.setAttribute(key,value);
 const path=document.createElementNS(icon.namespaceURI,'path');path.setAttribute('d',queueIconPaths[name]);icon.append(path);return icon;
}
function queueLabel(control,icon,label){
      const text=document.createElement('span');text.textContent=label;control.replaceChildren(queueIcon(icon),text);return control;
    }
    function queueControl(tag,label,icon,cls='secondary'){
      const control=document.createElement(tag);control.className=cls;return queueLabel(control,icon,label);
    }
    for(const[id,icon,label]of [['clear-queue','clear','Clear waiting'],['queue-previous','previous','Previous'],['queue-next','next','Next']])queueLabel($(id),icon,label);
    $('close-queue').replaceChildren(queueIcon('close'));
    $('open-queue').prepend(queueIcon('queue'));
    let queueOffset = 0, queueNext = null, queuePrevious = null, queuePaused = false, pollingQueue = false, queueListVersion = null, queueDrag = null, queueMutating = false;
    // The header and Queue modal switches both mirror the persisted pause flag from polling.
    // Each pause/resume bumps the revision so a poll requested before it cannot undo the change.
    let queuePauseRevision = 0;
    const queueSwitches=() => [$('queue-processing'),$('queue-processing-dialog')];
    function showQueuePaused(paused) {
      queuePaused=paused;
      for (const toggle of queueSwitches()) { toggle.checked=!paused; toggle.disabled=false; }
    }
    async function setQueuePaused(paused) {
      const toggles=queueSwitches(); toggles.forEach(toggle => { toggle.disabled=true; }); $('queue-switch-alert').textContent='';
      if (queueMutating) { showQueuePaused(queuePaused); return; }
      queueMutating=true; queuePauseRevision++; $('queue-error').hidden=true;
      try { const state=await requestJSON('/v1/queue',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused})}); showQueuePaused(!!state.paused); }
      catch(error) {
        const message=`Could not ${paused ? 'pause' : 'resume'} the queue: ${error.message}`;
        showQueuePaused(queuePaused); queueError(message); $('queue-switch-alert').textContent=message;
        setStatus(paused ? 'Pause failed' : 'Resume failed','offline'); statusHoldUntil=Date.now()+6000;
      }
      finally { toggles.forEach(toggle => { toggle.disabled=false; }); queueMutating=false; queuePauseRevision++; await refreshQueue(); }
    }
    async function requestJSON(url, options = {}) {
      const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(10000),...options});
      const data=await response.json();
      if (!response.ok) {const error=new Error(typeof data.detail === 'string' ? data.detail : 'Queue request failed.');error.status=response.status;throw error;}
      return data;
    }
    async function queueAction(url, options) {
      if (queueMutating) return;
      queueMutating=true; $('queue-error').hidden=true;
      try { return await requestJSON(url,options); }
      catch(error) { queueError(error.message); }
      finally { queueMutating=false; await refreshQueue(); }
    }
    function moveQueueJob(identifier,position) {
      return queueAction('/v1/jobs/'+identifier+'/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position})});
    }
    function clearQueueDrag() {
      const drag=queueDrag; queueDrag=null;
      drag?.ghost?.remove();
      $('queue-list').querySelectorAll('.queue-item').forEach(row => row.classList.remove('dragging'));
      if (drag?.active) queueListVersion=null;  // Rows were moved locally; the next poll redraws the saved order.
      return drag;
    }
    // Dragging lifts a copy of the row that follows the pointer; the real row stays as the gap and moves between rows.
    function liftQueueRow(row,event) {
      const bounds=row.getBoundingClientRect(),ghost=row.cloneNode(true);
      ghost.className='queue-item queue-ghost'; ghost.removeAttribute('data-position'); ghost.removeAttribute('data-job-id'); ghost.setAttribute('aria-hidden','true'); ghost.inert=true;
      Object.assign(ghost.style,{left:bounds.left+'px',top:bounds.top+'px',width:bounds.width+'px',height:bounds.height+'px'});
      $('queue-dialog').append(ghost); row.classList.add('dragging');
      return {ghost,offset:event.clientY-bounds.top};
    }
    function moveQueueGap(list,row,before) {
      // Animate the rows that shift around the gap from where they were (FLIP).
      const rows=[...list.children].filter(item => item!==row),first=new Map(rows.map(item => [item,item.getBoundingClientRect().top]));
      list.insertBefore(row,before);
      if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      for (const item of rows) {
        const shift=first.get(item)-item.getBoundingClientRect().top;
        if (shift) item.animate([{transform:`translateY(${shift}px)`},{transform:'none'}],{duration:150,easing:'ease-out'});
      }
    }
    function attachQueueDrag(handle,row,job) {
      handle.addEventListener('keydown',event => {
        if (!['ArrowUp','ArrowDown'].includes(event.key)) return;
        event.preventDefault();
        const position=job.queue_position+(event.key==='ArrowUp' ? -1 : 1);
        if (position<1 || position>Number($('queue-list').dataset.waiting)) return;
        moveQueueJob(job.id,position).then(() => {
          $('queue-list').querySelector(`[data-job-id="${job.id}"] .queue-drag`)?.focus();
        });
      });
      handle.addEventListener('pointerdown',event => {
        if (event.button!==0 || queueMutating) return;
        queueDrag={id:job.id,position:job.queue_position,startY:event.clientY,active:false};
        handle.setPointerCapture(event.pointerId);
      });
      handle.addEventListener('pointermove',event => {
        if (!queueDrag || queueDrag.id!==job.id) return;
        if (!queueDrag.active) {
          if (Math.abs(event.clientY-queueDrag.startY)<5) return;
          Object.assign(queueDrag,{active:true},liftQueueRow(row,event));
        }
        queueDrag.ghost.style.top=(event.clientY-queueDrag.offset)+'px';
        const list=$('queue-list'),waiting=[...list.querySelectorAll('[data-position]')];
        const others=waiting.filter(item => item!==row);
        if (!others.length) return;
        // Place the gap by the lifted row's centre, so releasing where the drag began restores the original order.
        const center=event.clientY-queueDrag.offset+row.offsetHeight/2;
        const next=others.find(item => { const bounds=item.getBoundingClientRect(); return center<=bounds.top+bounds.height/2; });
        const before=next || others.at(-1).nextSibling;
        if (before!==row && row.nextSibling!==before) moveQueueGap(list,row,before);
        // Waiting rows on a page hold consecutive global positions, so the gap's index gives the new position.
        const first=Math.min(...waiting.map(item => Number(item.dataset.position)));
        queueDrag.position=first+[...list.querySelectorAll('[data-position]')].indexOf(row);
        const bounds=list.getBoundingClientRect();
        if (event.clientY<bounds.top+35) list.scrollBy(0,-18);
        else if (event.clientY>bounds.bottom-35) list.scrollBy(0,18);
      });
      handle.addEventListener('pointerup',() => {
        const drag=clearQueueDrag();
        if (drag?.active && drag.position!==job.queue_position) moveQueueJob(job.id,drag.position);
        else refreshQueue();
      });
      handle.addEventListener('pointercancel',() => { clearQueueDrag(); refreshQueue(); });
      handle.addEventListener('lostpointercapture',() => { if (queueDrag) { clearQueueDrag(); refreshQueue(); } });
    }
    window.addEventListener('blur',clearQueueDrag);
    function renderQueueJob(job) {
      const row=document.createElement('div'); row.className='queue-item';
      const copy=document.createElement('div'); copy.className='queue-copy';
      const title=document.createElement('strong'); title.textContent=job.request.prompt.slice(0,180);
      const detail=document.createElement('small'); detail.textContent=`${modelNames.get(job.model) || job.model} · ${job.status}${job.status==='queued' ? ' · Position '+job.queue_position : ''}${job.generate_immediately ? ' · Immediate priority' : ''} · ${job.request.size==='auto' || (!job.request.size && job.request.reference_image) ? 'Auto resolution' : job.request.size || '1024x1024'} · ${job.completed_images || 0}/${job.total_images || 1} images · Attempt ${job.attempts}`;
      copy.append(title,detail);
      if (job.blocked_reason || job.error) { const error=document.createElement('div'); error.className='queue-help queue-error'; error.textContent=job.blocked_reason || job.error; copy.append(error); }
      if (job.status==='running') { const progress=document.createElement('div'); progress.className='queue-progress'; const label=document.createElement('small'); label.textContent=job.cancel_requested ? 'Stopping…' : progressLabel(job.progress) || 'Starting'; progress.append(progressBar(job.progress,'Job progress'),label); copy.append(progress); }
      row.dataset.jobId=job.id;
      if (job.status==='queued') {
        row.dataset.position=job.queue_position;
        const handle=document.createElement('button'); handle.type='button'; handle.className='queue-drag'; handle.append(queueIcon('grip'));
        handle.setAttribute('aria-label',`Move waiting job ${job.queue_position}: ${job.request.prompt.slice(0,80)}`);
        handle.setAttribute('aria-describedby','queue-drag-help');
        attachQueueDrag(handle,row,job); row.append(handle);
      } else {
        const spacer=document.createElement('span'); spacer.className='queue-drag-spacer'; spacer.setAttribute('aria-hidden','true'); row.append(spacer);
      }
      row.append(copy);
      const actions=document.createElement('div'); actions.className='queue-actions';
      // Each job opens in its own tab with its saved form, photo and preview; native link gestures keep working.
      const inspect=queueControl('a',job.status==='succeeded' ? 'Open image' : 'Monitor',job.status==='succeeded' ? 'play' : 'monitor','secondary queue-monitor');
      inspect.href='/?'+new URLSearchParams({job:job.id}); inspect.target='_blank'; inspect.rel='noopener'; inspect.title='Open this job in a new tab';
      actions.append(inspect);
      if (job.status==='running') {
        const stop=queueControl('button',job.cancel_requested ? 'Stopping…' : 'Cancel','stop','secondary queue-remove'); stop.type='button'; stop.disabled=job.cancel_requested;
        stop.title='Stop this job now. Finished options in a batch are kept.';
        stop.addEventListener('click',() => { if (confirm(CANCEL_RUNNING_PROMPT)) queueAction('/v1/jobs/'+job.id,{method:'DELETE'}); }); actions.append(stop);
      }
      if (job.status==='queued') {
        const run=queueControl('button','Run Now','play','secondary queue-run-now'); run.type='button';
        run.title='Run at the next free slot, even while paused; the active batch finishes first.';
        run.addEventListener('click',() => queueAction('/v1/jobs/'+job.id+'/run-now',{method:'POST'}).then(result => { if (result) { $('queue-dialog').close(); if(callbacks.onRunNow)callbacks.onRunNow(job);else inspectJob(job); } })); actions.append(run);
        const remove=queueControl('button','Remove','remove','secondary queue-remove'); remove.type='button';
        remove.addEventListener('click',() => queueAction('/v1/jobs/'+job.id,{method:'DELETE'})); actions.append(remove);
      } else if (['failed','cancelled'].includes(job.status)) {
        const retry=queueControl('button','Retry','retry'); retry.type='button';
        retry.addEventListener('click',() => queueAction('/v1/jobs/'+job.id+'/retry',{method:'POST'})); actions.append(retry);
      }
      if (job.attempts>0) { const metrics=queueControl('a','Metrics','resources','secondary queue-metrics'); metrics.href='/observability?'+new URLSearchParams({job:job.id,model:job.model}); metrics.title='View resource measurements for this job'; actions.append(metrics); }
      row.append(actions); return row;
    }
    async function refreshQueue() {
      if (pollingQueue || queueDrag || queueMutating) return;
      pollingQueue=true; const pauseRevision=queuePauseRevision;
      try {
        const params=new URLSearchParams({limit:String(QUEUE_PAGE_SIZE),offset:String(queueOffset)});
        if ($('queue-filter').value) params.set('status',$('queue-filter').value);
        const [list,health]=await Promise.all([requestJSON('/v1/jobs?'+params),requestJSON('/health').catch(error=>{setStatus('Server offline','offline');throw error;})]);
        updateHealth(health);
        const summary=health.queue || {};
        if (!queueMutating && pauseRevision===queuePauseRevision) showQueuePaused(!!summary.paused);
        const count=(summary.running || 0)+(summary.queued || 0); $('queue-badge').textContent=String(count); $('queue-badge').hidden=!count;
        $('queue-summary').textContent=`${summary.running || 0} active · ${summary.queued || 0} waiting${queuePaused ? ' · Queue paused' : ''}`;
        $('clear-queue').disabled=!summary.queued;
        $('queue-notice').textContent=summary.error || summary.blocked_reason || (!health.queue_api?.available && health.queue_api?.error) || (summary.stage==='waiting_for_accelerator' ? 'Waiting for the shared GPU; no job is lost.' : '');
        $('queue-notice').hidden=!$('queue-notice').textContent;
        if (queueOffset && !list.data.length) { queueOffset=Math.max(0,queueOffset-QUEUE_PAGE_SIZE); return; }
        if (queueDrag || queueMutating) return;
        $('queue-list').dataset.waiting=String(summary.queued || 0);
        const version=JSON.stringify(list.data);
        if (version!==queueListVersion) {
          $('queue-list').replaceChildren(...list.data.map(renderQueueJob)); queueListVersion=version;
          if (!list.total) { const empty=document.createElement('p'); empty.className='queue-help'; empty.textContent='No jobs in this view.'; $('queue-list').append(empty); }
        }
        queueNext=list.next_offset ?? null; queuePrevious=list.previous_offset ?? null;
        $('queue-next').disabled=queueNext===null; $('queue-previous').disabled=queuePrevious===null;
        $('queue-page').textContent=`Page ${Math.floor(queueOffset/QUEUE_PAGE_SIZE)+1} of ${Math.max(1,Math.ceil(list.total/QUEUE_PAGE_SIZE))} · ${list.total} jobs`;
        await callbacks.onRefresh?.(summary,health);
      } catch(error) { $('queue-summary').textContent='Queue unavailable · retrying'; $('queue-badge').textContent='!'; $('queue-badge').hidden=false; queueSwitches().forEach(toggle => { toggle.disabled=true; }); }
      finally { pollingQueue=false; }
    }
    $('open-queue').addEventListener('click',() => { $('queue-error').hidden=true; $('queue-dialog').showModal(); refreshQueue(); });
    $('close-queue').addEventListener('click',() => $('queue-dialog').close());
    $('queue-dialog').addEventListener('close',clearQueueDrag);
    $('queue-dialog').addEventListener('click',event => { if (event.target!==$('queue-dialog')) return; const bounds=event.target.getBoundingClientRect(); if (event.clientX<bounds.left || event.clientX>bounds.right || event.clientY<bounds.top || event.clientY>bounds.bottom) event.target.close(); });
    for (const toggle of queueSwitches()) toggle.addEventListener('change',event => setQueuePaused(!event.target.checked));
    $('clear-queue').addEventListener('click',() => queueAction('/v1/jobs',{method:'DELETE'}));
    $('queue-filter').addEventListener('change',() => { queueOffset=0; refreshQueue(); });
    $('queue-next').addEventListener('click',() => { if (queueNext!==null) { queueOffset=queueNext; refreshQueue(); } });
    $('queue-previous').addEventListener('click',() => { if (queuePrevious!==null) { queueOffset=queuePrevious; refreshQueue(); } });

  async function loadNames(){try{const catalog=await requestJSON('/v1/models');for(const model of catalog.data)modelNames.set(model.id,model.name);queueListVersion=null;}catch{/* Health still works when model discovery is temporarily unavailable. */}}
  window.ImageGenShell={configure(options){callbacks={...callbacks,...options};},refresh:refreshQueue,requestJSON,updateHealth,setStatus,progressLabel,setProgressBar,cancelRunningPrompt:CANCEL_RUNNING_PROMPT,resetOffset(){queueOffset=0;}};
  document.addEventListener('DOMContentLoaded',()=>{loadNames().then(refreshQueue);setInterval(refreshQueue,2000);setInterval(loadNames,15000);},{once:true});
})();
