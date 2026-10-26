(() => {
  'use strict';
  const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
  const windows=[['5m','5 minutes'],['30m','30 minutes'],['1h','1 hour'],['6h','6 hours'],['24h','24 hours'],['3d','3 days'],['7d','7 days']];
  const jobWindows=[['full','Entire job'],['1m','Last minute'],['5m','Last 5 minutes'],['15m','Last 15 minutes']];
  let params=new URLSearchParams(location.search),job=params.get('job'),jobModel=params.get('model')||'',offset=0,total=0;
  let overviewWindow='5m',jobWindow='full',loadedModels=false,requestNumber=0,controller=null,lastData=null,refreshing=false;
  const name=id=>modelNames.get(id)||id,modelNames=new Map();
  const percent=value=>value==null?'—':value.toFixed(1)+'%';
  const gib=value=>value==null?'—':(value/1024**3).toFixed(value<1024**3?2:1)+' GiB';
  const time=value=>new Date(value*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
  function el(tag,text,cls){const node=document.createElement(tag);if(text!=null)node.textContent=text;if(cls)node.className=cls;return node;}
  function svg(tag,attributes,text){const node=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attributes||{}))node.setAttribute(key,value);if(text!=null)node.textContent=text;return node;}
  function configureWindow(){
    $('time-window').replaceChildren(...(job?jobWindows:windows).map(([value,label])=>{const option=el('option',label);option.value=value;return option;}));
    $('time-window').value=job?jobWindow:overviewWindow;$('overview').hidden=!job;
    $('model-filter').disabled=!!job;
    if(job&&jobModel&&loadedModels)$('model-filter').value=jobModel;
    $('selection').textContent=job?`Job ${job}${jobModel?' · '+name(jobModel):''}`:'All measured activity';
  }
  async function json(url,signal){
    const response=await fetch(url,{signal,cache:'no-store'});let body;try{body=await response.json();}catch{throw new Error('The server returned an unreadable response.');}
    if(!response.ok){const error=new Error(typeof body.detail==='string'?body.detail:body.detail?.message||'Could not load resource history.');error.status=response.status;throw error;}return body;
  }
  function drawChart(container,data,lines,unit){
    container.replaceChildren();const width=Math.max(280,container.clientWidth),height=240,left=63,right=18,top=17,bottom=37;
    const points=data.points,span=Math.max(.1,data.end-data.start),plotWidth=width-left-right,plotHeight=height-top-bottom;
    const root=svg('svg',{viewBox:`0 0 ${width} ${height}`,role:'img',tabindex:'0','aria-label':container.id.replace('-chart','')+' history; use left and right arrow keys to inspect samples'});
    root.append(svg('desc',{},'Solid lines show sample means, dashed lines sampled peaks. Gaps indicate missing measurements.'));
    const max=unit==='percent'?100:Math.max(.25,...points.flatMap(point=>lines.map(line=>(point.peaks[line.key]||0)/1024**3)))*1.12;
    const value=number=>unit==='percent'?number:number/1024**3;
    const x=t=>left+(t-data.start)/span*plotWidth,y=v=>top+plotHeight*(1-value(v)/max);
    for(let i=0;i<=4;i++){
      const level=max*i/4,position=top+plotHeight*(1-i/4);
      root.append(svg('line',{x1:left,x2:width-right,y1:position,y2:position,class:'grid'}));
      root.append(svg('text',{x:left-10,y:position+4,'text-anchor':'end'},unit==='percent'?Math.round(level)+'%':level.toFixed(max<2?2:1)+' GiB'));
    }
    const ticks=width<420?2:width<650?3:4;
    for(let i=0;i<=ticks;i++){
      const stamp=data.start+span*i/ticks,date=new Date(stamp*1000);
      const label=span>86400?date.toLocaleDateString(undefined,{month:'short',day:'numeric'}):date.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',...(span<600?{second:'2-digit'}:{})});
      root.append(svg('text',{x:left+plotWidth*i/ticks,y:height-10,'text-anchor':i===0?'start':i===ticks?'end':'middle'},label));
    }
    const signature=point=>[...point.sessions].sort().join(',')+'/'+point.models.join(',');
    for(const line of lines){
      for(const field of ['peaks','values']){
        let path='',previous=null;
        for(const point of points){
          const number=point[field][line.key];
          if(number==null){previous=null;continue;}
          const join=previous&&signature(previous)===signature(point)&&point.first_sample-previous.last_sample<=Math.max(12,data.bucket_seconds*2.5);
          path+=(join?'L':'M')+x(point.timestamp).toFixed(2)+','+y(number).toFixed(2)+' ';
          if(!join)root.append(svg('circle',{cx:x(point.timestamp),cy:y(number),r:1.8,fill:line.color,opacity:field==='peaks'?'.4':'1'}));
          previous=point;
        }
        root.append(svg('path',{d:path,stroke:line.color,class:field==='values'?'line':'max-line'}));
      }
    }
    if(!points.length||!points.some(point=>lines.some(line=>point.values[line.key]!=null))){
      root.append(svg('text',{x:left+plotWidth/2,y:top+plotHeight/2,'text-anchor':'middle'},points.length?'Counter unavailable for this window':'No samples in this window'));
    }
    const tooltip=el('div',null,'chart-tooltip');tooltip.hidden=true;
    const cursor=svg('line',{x1:0,x2:0,y1:top,y2:height-bottom,stroke:'#dddddd','stroke-dasharray':'2 4',visibility:'hidden'});root.append(cursor);
    let selected=0;
    const show=index=>{
      if(!points.length)return;selected=Math.max(0,Math.min(points.length-1,index));const point=points[selected];
      const labels=[time(point.timestamp),point.models.map(name).join(' · '),point.stages.join(' / ')];
      for(const line of lines){const format=unit==='percent'?percent:gib;labels.push(`${line.label}: ${format(point.values[line.key])} · peak ${format(point.peaks[line.key])}`);}
      tooltip.textContent=labels.join('\n');tooltip.hidden=false;cursor.setAttribute('x1',x(point.timestamp));cursor.setAttribute('x2',x(point.timestamp));cursor.setAttribute('visibility','visible');
    };
    const hide=()=>{tooltip.hidden=true;cursor.setAttribute('visibility','hidden');};
    root.addEventListener('pointermove',event=>{const bounds=root.getBoundingClientRect(),at=data.start+((event.clientX-bounds.left)/bounds.width*width-left)/plotWidth*span;let nearest=0;for(let i=1;i<points.length;i++)if(Math.abs(points[i].timestamp-at)<Math.abs(points[nearest].timestamp-at))nearest=i;show(nearest);});
    root.addEventListener('pointerleave',hide);root.addEventListener('blur',hide);
    root.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Escape'].includes(event.key)){event.preventDefault();if(event.key==='Escape')hide();else show(selected+(event.key==='ArrowRight'?1:-1));}});
    container.append(root,tooltip);
  }
  function combined(runs){
    const summary={metrics:{}};for(const key of ['service_cpu_percent','service_rss_bytes','gpu_utilization_percent']){
      const metrics=runs.map(run=>run.summary.metrics[key]).filter(Boolean),count=metrics.reduce((sum,item)=>sum+item.count,0);
      summary.metrics[key]={mean:count?metrics.reduce((sum,item)=>sum+(item.mean||0)*item.count,0)/count:null,peak:metrics.some(item=>item.peak!=null)?Math.max(...metrics.filter(item=>item.peak!=null).map(item=>item.peak)):null};
    }return summary;
  }
  function render(data){
    lastData=data;const summary=job&&jobWindow==='full'?combined(data.runs||[]):data.summary;
    $('cpu-value').textContent=percent(summary.metrics.service_cpu_percent?.mean);$('memory-value').textContent=gib(summary.metrics.service_rss_bytes?.peak);$('gpu-value').textContent=percent(summary.metrics.gpu_utilization_percent?.peak);
    drawChart($('cpu-chart'),data,[{key:'service_cpu_percent',label:'Service',color:'#8ebeff'},{key:'system_cpu_percent',label:'System',color:'#aaaaaa'}],'percent');
    drawChart($('gpu-chart'),data,[{key:'gpu_utilization_percent',label:'GPU busy',color:'#67d7e8'}],'percent');
    drawChart($('memory-chart'),data,[{key:'service_rss_bytes',label:'Service RSS',color:'#8ebeff'},{key:'system_memory_used_bytes',label:'System RAM',color:'#aaaaaa'},{key:'gpu_memory_bytes',label:'GPU driver',color:'#67d7e8'}],'memory');
    $('coverage').textContent=`${data.summary.sample_count.toLocaleString()} retained samples · ${data.points.length} plotted buckets · ${time(data.start)} — ${time(data.end)}. ${job?'Execution samples only; gaps can include waiting between attempts.':'Stopped periods are gaps; hover or focus a graph and use ← / → for models and values.'} Raw samples expire after 8 days; completed attempt summaries remain.`;
  }
  function renderJobs(result){
    total=result.total;$('jobs').replaceChildren();$('no-jobs').hidden=!!result.data.length;$('job-count').textContent=total+' measured attempt'+(total===1?'':'s');
    for(const run of result.data){const row=el('tr');
      row.append(el('td',time(run.started)),el('td',name(run.model)));
      const identifier=el('td');identifier.append(el('span',run.job_id.slice(0,10)+'…','job-id'),el('span','Attempt '+run.attempt,'attempt'));identifier.title=run.job_id;row.append(identifier);
      const status=el('td');status.append(el('span',run.status,'badge '+run.status));row.append(status);
      const metrics=run.summary.metrics;row.append(el('td',percent(metrics.service_cpu_percent?.mean)),el('td',gib(metrics.service_rss_bytes?.peak)),el('td',percent(metrics.gpu_utilization_percent?.peak)));
      const action=el('td'),link=el('a','View job','button');link.href='/observability?'+new URLSearchParams({job:run.job_id,model:run.model});
      link.addEventListener('click',event=>{if(event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;event.preventDefault();job=run.job_id;jobModel=run.model;offset=0;history.pushState(null,'',link.href);configureWindow();refresh();window.scrollTo({top:0,behavior:'instant'});});action.append(link);row.append(action);$('jobs').append(row);
    }
    $('previous').disabled=offset===0;$('next').disabled=!result.has_more;$('page-label').textContent=`Page ${Math.floor(offset/20)+1} of ${Math.max(1,Math.ceil(total/20))}`;
  }
  let logsJob=null;
  function configureLogs(){
    $('job-logs').hidden=!job;
    if(logsJob!==job){logsJob=job;$('log-sources').replaceChildren();$('logs-note').textContent='';$('logs-error').hidden=true;$('logs-status').textContent='Loading…';}
  }
  function renderLogs(data){
    $('logs-status').textContent=data.available?'Available · '+data.status:'No logs yet';
    $('logs-note').textContent=data.note || '';
    $('logs-error').hidden=!data.error;if(data.error)$('logs-error').textContent='Job failed: '+data.error;
    const existing=new Map([...$('log-sources').children].map(node=>[node.dataset.path,node]));
    const retained=new Set();
    for(const source of data.sources){
      let details=existing.get(source.file_path);retained.add(source.file_path);
      if(!details){details=el('details');details.dataset.path=source.file_path;details.open=true;details.append(el('summary'),el('p','','log-meta'),el('pre'));$('log-sources').append(details);}
      details.querySelector('summary').textContent=source.label+(source.truncated?' · tail truncated':'');
      details.querySelector('.log-meta').textContent=source.file_path+' · '+source.bytes_read.toLocaleString()+' of '+source.size_bytes.toLocaleString()+' bytes';
      const pre=details.querySelector('pre');if(pre.textContent!==source.text){const bottom=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-8;pre.textContent=source.text || 'Log file is empty.';if(bottom)pre.scrollTop=pre.scrollHeight;}
    }
    for(const [path,node] of existing)if(!retained.has(path))node.remove();
  }
  async function refreshLogs(current,number){
    if(!job)return;
    const query=new URLSearchParams();if(jobModel)query.set('model',jobModel);
    try{
      const data=await json('/v1/observability/jobs/'+encodeURIComponent(job)+'/logs?'+query,current.signal);
      if(number===requestNumber)renderLogs(data);
    }catch(error){
      if(number!==requestNumber)return;
      $('logs-status').textContent='Logs unavailable';$('logs-error').hidden=false;
      $('logs-error').textContent=error.name==='AbortError'?'Log request timed out; live updates will retry.':error.message;
    }
  }
  async function refresh(){
    if(controller)controller.abort();refreshing=true;controller=new AbortController();const current=controller,number=++requestNumber,timer=setTimeout(()=>current.abort(),12000);
    configureLogs();const logsRequest=refreshLogs(current,number);
    const selectedModel=job?jobModel:$('model-filter').value,query=new URLSearchParams({window:job?jobWindow:overviewWindow});if(selectedModel)query.set('model',selectedModel);
    const jobsQuery=new URLSearchParams({limit:'20',offset:String(offset)});if(selectedModel)jobsQuery.set('model',selectedModel);
    try{
      const [status,data,result]=await Promise.all([json('/v1/observability',current.signal),json((job?'/v1/observability/jobs/'+encodeURIComponent(job):'/v1/observability/metrics')+'?'+query,current.signal).catch(error=>{if(job&&error.status===404){const end=Date.now()/1000;return {start:end-60,end,bucket_seconds:1,points:[],runs:[],summary:{sample_count:0,metrics:{}},unmeasured:true};}throw error;}),json('/v1/observability/jobs?'+jobsQuery,current.signal)]);
      if(number!==requestNumber)return;
      if(!loadedModels){for(const model of status.models){modelNames.set(model.id,model.name);const option=el('option',model.name);option.value=model.id;$('model-filter').append(option);}loadedModels=true;if(job&&jobModel)$('model-filter').value=jobModel;}
      $('error').hidden=!data.unmeasured;if(data.unmeasured)$('error').textContent='No measurements for this job yet. It may still be waiting, or it ran before observability was added. Live updates will retry.';$('connection').textContent=status.error?'Sampling needs attention':status.recording?'Recording · '+name(status.model):'History · sampler stopped';$('connection').classList.toggle('offline',!!status.error||!status.recording);
      const messages=['GPU values cover the whole device; GPU memory and process RSS overlap in unified RAM.'];if(status.error)messages.push(status.error);for(const value of [status.sources.cpu_memory,status.sources.gpu])if(value!=='available')messages.push(value);if(status.sources.process_tree_partial)messages.push('Some child-process counters could not be read; service totals may be partial.');$('sources').textContent=messages.join(' ');
      $('selection').textContent=job?`Job ${job} · ${selectedModel?name(selectedModel):data.runs.map(run=>name(run.model)).join(', ')}`:selectedModel?name(selectedModel)+' · all measured sessions':'All models · all measured sessions';
      render(data);renderJobs(result);
    }catch(error){if(number!==requestNumber)return;if(error.name==='AbortError')error=new Error('History request timed out. Check that the generator is running.');$('error').textContent=job&&error.status===404?'No measurements for this job yet. It may still be waiting, or it ran before observability was added. This page will retry while live updates are enabled.':error.message;$('error').hidden=false;$('connection').textContent='History unavailable';$('connection').classList.add('offline');}
    finally{await logsRequest;clearTimeout(timer);if(number===requestNumber)refreshing=false;}
  }
  $('time-window').addEventListener('change',()=>{if(job)jobWindow=$('time-window').value;else overviewWindow=$('time-window').value;refresh();});
  $('model-filter').addEventListener('change',()=>{offset=0;refresh();});$('refresh').addEventListener('click',refresh);
  $('overview').addEventListener('click',()=>{job=null;jobModel='';offset=0;history.pushState(null,'','/observability');$('model-filter').value='';configureWindow();refresh();});
  $('previous').addEventListener('click',()=>{offset=Math.max(0,offset-20);refresh();});$('next').addEventListener('click',()=>{if(offset+20<total){offset+=20;refresh();}});
  window.addEventListener('popstate',()=>{params=new URLSearchParams(location.search);job=params.get('job');jobModel=params.get('model')||'';offset=0;configureWindow();refresh();});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden&&$('live').checked)refresh();});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(lastData)render(lastData);},150);});
  setInterval(()=>{if(!document.hidden&&$('live').checked&&!refreshing)refresh();},5000);
  configureWindow();refresh();
})();
