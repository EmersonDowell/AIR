(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const state = {
    connected: false,
    health: null,
    model: null,
    models: null,
    runtime: null,
    eventMap: new Map(),
    metricsText: '',
    lastResponse: null,
    lastNativeMetrics: null,
    rawTab: 'runtime',
    playgroundMode: 'native',
    abortController: null,
    pollTimer: null,
    polling: false,
    retryMs: 800,
  };

  const eventCategories = {
    request_queued: 'lifecycle', request_admitted: 'lifecycle', request_complete: 'lifecycle', request_failed: 'error', request_cancelled: 'error', stream_delivery_failed: 'error',
    scheduler_start: 'scheduler', scheduler_stop: 'scheduler', scheduler_yield_for_admission: 'scheduler',
    plan_selected: 'planner', regime_admission_compatible: 'planner', regime_admission_shape_compatible: 'planner', regime_admission_wait: 'planner', regime_candidate_skipped: 'planner', regime_epoch_start: 'planner', regime_epoch_end: 'planner', regime_epoch_fallback_pinned: 'planner', regime_reoptimize_shadow: 'planner', regime_transition_blocked: 'planner', regime_transition_committed: 'planner', regime_transition_failed: 'error', regime_transition_skipped: 'planner',
    prefix_hit: 'memory', prefill_batch: 'batch', decode_batch: 'batch'
  };

  function val(v, fallback = '—') { return v === undefined || v === null || v === '' ? fallback : v; }
  function num(v) { return Number.isFinite(Number(v)) ? Number(v) : 0; }
  function fmtNum(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return '—';
    if (Math.abs(n) >= 1e9) return `${(n/1e9).toFixed(2)}B`;
    if (Math.abs(n) >= 1e6) return `${(n/1e6).toFixed(2)}M`;
    if (Math.abs(n) >= 1e3) return `${(n/1e3).toFixed(2)}K`;
    return new Intl.NumberFormat().format(n);
  }
  function fmtBytes(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return '—';
    if (n === 0) return '0 B';
    const units = ['B','KB','MB','GB','TB'];
    let i = 0, x = Math.abs(n);
    while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
    return `${x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)} ${units[i]}`;
  }
  function fmtMs(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return '—';
    return n >= 1000 ? `${(n/1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 1 : 0)} ms`;
  }
  function fmtBool(v) { return v === true ? 'Yes' : v === false ? 'No' : '—'; }
  function pct(part, whole) { const p = num(part), w = num(whole); return w > 0 ? Math.max(0, Math.min(100, (p / w) * 100)) : 0; }
  function setText(id, text) { const el = $(id); if (el) el.textContent = val(text); }
  function setWidth(id, percent) { const el = $(id); if (el) el.style.width = `${Math.max(0, Math.min(100, percent || 0))}%`; }
  function jsonText(value) { try { return JSON.stringify(value ?? {}, null, 2); } catch { return '{}'; } }

  async function api(path, options = {}) {
    const response = await fetch(path, { cache: 'no-store', ...options });
    if (!response.ok) {
      let detail = '';
      try { detail = await response.text(); } catch {}
      throw new Error(`${response.status} ${response.statusText}${detail ? ` · ${detail}` : ''}`);
    }
    return response;
  }

  async function fetchJson(path) { return (await api(path)).json(); }

  function connectionState(ok, label) {
    state.connected = ok;
    const pill = $('connectionPill');
    const dot = pill?.querySelector('.status-dot');
    if (dot) dot.className = `status-dot ${ok ? 'status-ok' : 'status-bad'}`;
    setText('connectionLabel', label || (ok ? 'Connected' : 'Disconnected'));
    setText('stripStatus', ok ? 'Operational' : 'Disconnected');
  }

  function renderRuntime() {
    const r = state.runtime || {};
    const planner = r.planner || {};
    const sched = r.scheduler || {};
    const caps = r.capabilities || {};

    setText('stripBackend', val(r.backend, state.health?.backend));
    setText('stripModel', val(state.model?.id));
    setText('stripStrategy', val(planner.strategy_id));
    setText('stripActive', fmtNum(r.active_requests));
    setText('stripQueued', fmtNum(r.queued_requests));
    setText('stripTps', Number.isFinite(Number(r.aggregate_generated_tokens_per_second)) ? Number(r.aggregate_generated_tokens_per_second).toFixed(1) : '—');

    setText('metricTps', Number.isFinite(Number(r.aggregate_generated_tokens_per_second)) ? Number(r.aggregate_generated_tokens_per_second).toFixed(1) : '—');
    setText('metricP50', fmtMs(r.p50_total_ms));
    setText('metricP95', fmtMs(r.p95_total_ms));
    setText('metricPrefix', fmtNum(r.total_prefix_reused_tokens));
    setText('metricCompleted', fmtNum(r.completed_requests));
    setText('metricFailed', fmtNum(num(r.failed_requests) + num(r.stream_delivery_failures)));

    setText('strategySelected', planner.strategy_id);
    setText('plannerMode', planner.mode);
    setText('plannerObjective', planner.objective);
    setText('plannerTransition', fmtMs(planner.estimated_transition_ms));
    setText('plannerBreakEven', planner.estimated_break_even_tokens == null ? '—' : `${fmtNum(planner.estimated_break_even_tokens)} tok`);
    setText('plannerReason', planner.decision_reason || 'AIR has not reported a planner decision reason.');
    renderCandidateRows('candidateRows', planner.candidates || [], planner.strategy_id, false);

    setText('flowQueued', fmtNum(r.queued_requests));
    setText('flowAdmitted', fmtNum(r.active_requests));
    setText('flowPrefill', sched.max_prefill_batch_width == null ? '—' : `≤ ${fmtNum(sched.max_prefill_batch_width)}`);
    setText('flowDecode', sched.max_decode_batch_width == null ? '—' : `≤ ${fmtNum(sched.max_decode_batch_width)}`);
    setText('flowComplete', fmtNum(r.completed_requests));
    ['flowQueuedOrb','flowAdmittedOrb','flowPrefillOrb','flowDecodeOrb'].forEach(id => $(id)?.classList.toggle('active', num(r.active_requests) > 0 || num(r.queued_requests) > 0));
    $('flowCompleteOrb')?.classList.toggle('active', num(r.completed_requests) > 0);

    setText('schedMaxActive', fmtNum(sched.max_active_requests));
    setText('schedTokenBudget', fmtNum(sched.token_budget_per_cycle));
    setText('schedPrefillQuantum', fmtNum(sched.prefill_quantum_tokens));
    setText('schedPrefillBatch', fmtNum(sched.max_prefill_batch_width));
    setText('schedDecodeBatch', fmtNum(sched.max_decode_batch_width));
    setText('schedStreamQueue', fmtNum(sched.stream_queue_capacity));

    setText('deviceMemoryLabel', `${fmtBytes(r.current_device_bytes)} / peak ${fmtBytes(r.peak_device_bytes)}`);
    setWidth('deviceMemoryBar', pct(r.current_device_bytes, r.peak_device_bytes));
    setText('kvMemoryLabel', `${fmtBytes(r.current_kv_bytes)} / peak ${fmtBytes(r.peak_kv_bytes)}`);
    setWidth('kvMemoryBar', pct(r.current_kv_bytes, r.peak_kv_bytes));
    setText('admissionLabel', `${fmtBytes(caps.admission_reserved_bytes)} / ${fmtBytes(caps.admission_capacity_bytes)}`);
    setWidth('admissionBar', pct(caps.admission_reserved_bytes, caps.admission_capacity_bytes));
    const poolAllocated = num(caps.kv_pool_allocated_bytes);
    const poolFree = num(caps.kv_pool_free_bytes);
    const poolUsed = Math.max(0, poolAllocated - poolFree);
    setText('kvPoolLabel', `${fmtBytes(poolUsed)} used / ${fmtBytes(poolAllocated)}`);
    setWidth('kvPoolBar', pct(poolUsed, poolAllocated));
    setText('preparedArtifacts', fmtBytes(r.current_prepared_artifact_bytes));
    setText('prefixCacheCap', fmtBool(caps.prefix_cache_enabled));
    setText('pagedKvCap', fmtBool(caps.physical_paged_kv));
    setText('greedyCap', fmtBool(caps.device_greedy_selection));
    setText('checkpointingChip', `Checkpointing ${fmtBool(caps.sequence_checkpointing)}`);
    renderOptionalSequenceStore(r);

    renderStrategyPage(planner);
    renderRawApi();
  }

  function candidateBadge(value, goodWhenTrue = true) {
    if (value === true) return `<span class="badge ${goodWhenTrue ? 'good' : 'warn'}">yes</span>`;
    if (value === false) return `<span class="badge ${goodWhenTrue ? 'bad' : 'good'}">no</span>`;
    return '<span class="badge">—</span>';
  }

  function renderCandidateRows(targetId, candidates, selected, extended) {
    const body = $(targetId); if (!body) return;
    body.textContent = '';
    if (!Array.isArray(candidates) || !candidates.length) {
      const tr = document.createElement('tr'); const td = document.createElement('td');
      td.colSpan = extended ? 9 : 7; td.className='empty-cell'; td.textContent='No candidate data reported.'; tr.appendChild(td); body.appendChild(tr); return;
    }
    candidates.forEach(c => {
      const tr = document.createElement('tr');
      if (c.strategy_id === selected) tr.classList.add('selected-row');
      const values = extended ? [
        c.strategy_id, c.disposition, c.eligible, c.memory_feasible, c.prepared_state_hot, fmtMs(c.estimated_transition_ms), fmtMs(c.estimated_horizon_ms), c.estimated_break_even_tokens == null ? '—' : `${fmtNum(c.estimated_break_even_tokens)} tok`, fmtBytes(c.prepared_artifact_bytes)
      ] : [
        c.strategy_id, c.disposition, c.memory_feasible, c.prepared_state_hot, fmtMs(c.estimated_transition_ms), fmtMs(c.estimated_horizon_ms), c.estimated_break_even_tokens == null ? '—' : `${fmtNum(c.estimated_break_even_tokens)} tok`
      ];
      values.forEach((value, index) => {
        const td = document.createElement('td');
        if ((extended && [2,3,4].includes(index)) || (!extended && [2,3].includes(index))) {
          const b = document.createElement('span'); b.className = `badge ${value === true ? 'good' : value === false ? 'bad' : ''}`; b.textContent = value === true ? 'yes' : value === false ? 'no' : '—'; td.appendChild(b);
        } else td.textContent = val(value);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderStrategyPage(planner) {
    setText('strategyPageSelected', planner.strategy_id);
    setText('strategyPageReason', planner.decision_reason || 'No planner decision reason reported.');
    setText('strategyPageBackend', planner.selected_backend || state.runtime?.backend);
    setText('strategyMode', planner.mode);
    setText('strategyManifest', planner.manifest_status || planner.manifest_id);
    setText('strategyObjective', planner.objective);
    setText('strategyEligible', fmtNum(planner.eligible_candidates));
    setText('strategyPrepared', fmtBool(planner.prepared_state_hot));
    setText('strategyKvPage', planner.kv_page_tokens == null ? '—' : `${fmtNum(planner.kv_page_tokens)} tok`);
    renderCandidateRows('strategyPageRows', planner.candidates || [], planner.strategy_id, true);
    setText('tacticBackend', planner.selected_backend || state.runtime?.backend);
    setText('tacticPrefillQuantum', planner.prefill_quantum_tokens == null ? '—' : `${fmtNum(planner.prefill_quantum_tokens)} tok`);
    setText('tacticPrefillLinear', planner.prefill_block_quantized_linear);
    setText('tacticDecodeLinear', planner.decode_block_quantized_linear);
    setText('tacticDecodeOutput', planner.decode_output_quantized_linear);
    setText('tacticPrefillAttention', planner.prefill_attention);
    setText('tacticDecodeAttention', planner.decode_attention);
  }

  function renderOptionalSequenceStore(r) {
    const entries = Object.entries(r).filter(([key]) => /sequence.*(store|state)|state_store/i.test(key));
    const module = $('sequenceStoreModule'); const grid = $('sequenceStoreGrid');
    if (!module || !grid) return;
    module.hidden = entries.length === 0;
    if (!entries.length) return;
    grid.textContent = '';
    entries.slice(0,10).forEach(([key,value]) => {
      const div = document.createElement('div'); const s = document.createElement('span'); const strong = document.createElement('strong');
      s.textContent = key.replaceAll('_',' '); strong.textContent = typeof value === 'number' ? fmtNum(value) : String(value); div.append(s,strong); grid.appendChild(div);
    });
  }

  function renderModel() {
    const m = state.model || {};
    setText('modelTitle', m.id || 'No model reported');
    setText('modelBackendBadge', m.backend || state.runtime?.backend);
    setText('modelFormat', m.format);
    setText('modelArch', m.architecture);
    setText('modelLayers', fmtNum(m.layers));
    setText('modelEmbedding', fmtNum(m.embedding));
    setText('modelContext', fmtNum(m.context_length));
    setText('modelVocab', fmtNum(m.vocabulary_size));
    setText('modelPageTitle', m.id || 'No model reported');
    setText('modelPageSubtitle', m.format ? `${m.format.toUpperCase()} · ${val(m.backend)} backend` : 'Waiting for /model.');
    setText('modelPageArch', m.architecture);
    setText('modelPageContext', fmtNum(m.context_length));
    setText('modelPageLayers', fmtNum(m.layers));
    setText('modelPageEmbedding', fmtNum(m.embedding));
    setText('modelsOutput', jsonText(state.models));
    renderRawApi();
  }

  function mergeEvents(events) {
    if (!Array.isArray(events)) return;
    events.forEach(event => {
      const key = event?.sequence ?? `${event?.unix_ms}-${event?.type}-${event?.request_id}`;
      state.eventMap.set(key, event);
    });
    if (state.eventMap.size > 400) {
      const sortedKeys = [...state.eventMap.entries()].sort((a,b) => num(a[1]?.sequence) - num(b[1]?.sequence)).map(([k]) => k);
      while (sortedKeys.length > 400) state.eventMap.delete(sortedKeys.shift());
    }
  }

  function eventsSorted() { return [...state.eventMap.values()].sort((a,b) => num(b.sequence) - num(a.sequence)); }
  function eventCategory(e) { return eventCategories[e?.type] || (String(e?.type || '').includes('regime') ? 'planner' : 'runtime'); }
  function eventTime(e) { const ms = Number(e?.unix_ms); return Number.isFinite(ms) ? new Date(ms).toLocaleTimeString([], {hour12:false}) : '—'; }

  function renderEvents() {
    const events = eventsSorted();
    const filter = ($('overviewEventFilter')?.value || '').trim().toLowerCase();
    const overview = $('overviewEvents');
    if (overview) {
      overview.textContent='';
      const filtered = events.filter(e => !filter || `${e.request_id} ${e.type} ${e.detail} ${e.sequence}`.toLowerCase().includes(filter)).slice(0,12);
      if (!filtered.length) { const empty=document.createElement('div'); empty.className='empty-state'; empty.textContent='No matching runtime events.'; overview.appendChild(empty); }
      filtered.forEach(e => {
        const row=document.createElement('div'); row.className='event-row';
        const t=document.createElement('time'); t.textContent=eventTime(e);
        const type=document.createElement('span'); type.className='event-type'; const dot=document.createElement('span'); dot.className=`status-dot ${eventCategory(e)==='error'?'status-bad':'status-ok'}`; const txt=document.createElement('span'); txt.textContent=val(e.type); type.append(dot,txt);
        const cat=document.createElement('span'); cat.className='event-category'; cat.textContent=eventCategory(e);
        const detail=document.createElement('span'); detail.className='event-detail'; detail.textContent=`#${val(e.sequence)} · req ${val(e.request_id)} · ${val(e.detail,'')}`;
        row.append(t,type,cat,detail); overview.appendChild(row);
      });
    }

    const types = [...new Set(events.map(e=>e.type).filter(Boolean))].sort();
    const select = $('eventTypeFilter');
    if (select) {
      const current=select.value; select.textContent=''; const opt=document.createElement('option'); opt.value=''; opt.textContent='All event types'; select.appendChild(opt);
      types.forEach(type => { const o=document.createElement('option'); o.value=type; o.textContent=type; select.appendChild(o); });
      if (types.includes(current)) select.value=current;
    }
    renderDiagnosticRows();
    renderRawApi();
  }

  function renderDiagnosticRows() {
    const body=$('diagnosticRows'); if(!body)return;
    const search=($('eventSearch')?.value||'').trim().toLowerCase(); const type=$('eventTypeFilter')?.value||'';
    const filtered=eventsSorted().filter(e => (!type || e.type===type) && (!search || `${e.sequence} ${e.request_id} ${e.type} ${e.detail} ${eventCategory(e)}`.toLowerCase().includes(search)));
    setText('eventCount', `${filtered.length} event${filtered.length===1?'':'s'}`);
    body.textContent='';
    if(!filtered.length){ const tr=document.createElement('tr'); const td=document.createElement('td'); td.colSpan=6; td.className='empty-cell'; td.textContent='No matching events.'; tr.appendChild(td); body.appendChild(tr); return; }
    filtered.forEach(e=>{ const tr=document.createElement('tr'); [eventTime(e),val(e.sequence),eventCategory(e),val(e.type),val(e.request_id),val(e.detail,'')].forEach(v=>{ const td=document.createElement('td'); td.textContent=v; tr.appendChild(td); }); body.appendChild(tr); });
  }

  function renderRawApi() {
    let value = {};
    if(state.rawTab==='runtime') value=state.runtime;
    else if(state.rawTab==='events') value=eventsSorted().slice(0,100);
    else if(state.rawTab==='model') value=state.model;
    else value=state.lastResponse;
    setText('rawApiOutput', jsonText(value));
  }

  async function pollOnce() {
    if (state.polling) return;
    state.polling = true;
    try {
      const [health, runtime, events] = await Promise.all([fetchJson('/health'), fetchJson('/runtime'), fetchJson('/events')]);
      const wasConnected = state.connected;
      state.health = health; state.runtime = runtime; mergeEvents(events);
      connectionState(health?.status === 'ok', health?.status === 'ok' ? 'Connected' : val(health?.status,'Degraded'));
      renderRuntime(); renderEvents();
      if (!wasConnected || !state.model) await refreshModelData();
      state.retryMs = document.hidden ? 3000 : 800;
    } catch (error) {
      connectionState(false,'Disconnected');
      setText('stripBackend','—');
      state.retryMs = Math.min(Math.max(state.retryMs * 1.6, 1600), 8000);
      if ($('rawApiOutput') && state.rawTab === 'runtime') setText('rawApiOutput', jsonText({error: String(error.message || error)}));
    } finally {
      state.polling = false;
      schedulePoll();
    }
  }

  function schedulePoll() {
    clearTimeout(state.pollTimer);
    const delay = document.hidden ? Math.max(3000,state.retryMs) : state.retryMs;
    state.pollTimer = setTimeout(pollOnce, delay);
  }

  async function refreshModelData() {
    try { state.model = await fetchJson('/model'); } catch {}
    try { state.models = await fetchJson('/v1/models'); } catch {}
    renderModel();
  }

  async function refreshMetrics() {
    try { state.metricsText = await (await api('/metrics')).text(); setText('metricsOutput', state.metricsText); }
    catch(error){ setText('metricsOutput', `# Failed to load /metrics\n${error.message || error}`); }
  }

  function makePayload(prompt, stream, mode, values) {
    const common = { max_tokens: values.maxTokens, temperature: values.temperature, top_p: values.topP, top_k: values.topK, seed: values.seed, stream };
    if (mode === 'chat') return { messages:[{role:'user',content:prompt}], ...common };
    return { prompt, ...common };
  }

  function endpointForMode(mode) { return mode === 'native' ? '/generate' : mode === 'chat' ? '/v1/chat/completions' : '/v1/completions'; }

  function extractStreamText(mode, data) {
    if (mode === 'native') return data?.text ?? '';
    if (mode === 'chat') return data?.choices?.[0]?.delta?.content ?? '';
    return data?.choices?.[0]?.text ?? '';
  }

  function extractNonStreamText(mode, data) {
    if (mode === 'native') return data?.text ?? '';
    if (mode === 'chat') return data?.choices?.[0]?.message?.content ?? '';
    return data?.choices?.[0]?.text ?? '';
  }

  async function runRequest({prompt, mode='native', stream=true, values, output, status, quick=false}) {
    if (!prompt.trim()) { output.textContent='Enter a prompt first.'; return; }
    const controller = new AbortController(); state.abortController = controller;
    if (!quick) { $('abortPlayground').disabled=false; $('runPlayground').disabled=true; }
    output.textContent=''; if(status) status.textContent=stream?'Streaming…':'Running…';
    try {
      const response = await api(endpointForMode(mode), { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(makePayload(prompt,stream,mode,values)), signal:controller.signal });
      if (stream) {
        if (!response.body) throw new Error('Streaming response body is unavailable.');
        const reader=response.body.getReader(); const decoder=new TextDecoder(); let buffer=''; let accumulated='';
        while(true){
          const {value,done}=await reader.read(); if(done)break; buffer+=decoder.decode(value,{stream:true});
          const frames=buffer.split('\n\n'); buffer=frames.pop()||'';
          for(const frame of frames){
            const line=frame.split('\n').find(l=>l.startsWith('data:'));
            if(!line)continue; const payload=line.slice(5).trim(); if(payload==='[DONE]')continue;
            try { const data=JSON.parse(payload); state.lastResponse=data; accumulated+=extractStreamText(mode,data); output.textContent=accumulated; }
            catch { /* keep malformed server fragment visible through diagnostics, not as model text */ }
          }
        }
        if(status) status.textContent='Complete.';
      } else {
        const data=await response.json(); state.lastResponse=data; output.textContent=extractNonStreamText(mode,data) || jsonText(data); if(status)status.textContent='Complete.';
        if(mode==='native'){ state.lastNativeMetrics=data?.metrics||null; renderNativeInspector(); }
      }
      renderRawApi();
    } catch(error) {
      if (error.name === 'AbortError') { output.textContent += '\n\n[stream aborted by browser]'; if(status)status.textContent='Aborted.'; }
      else { output.textContent = `Request failed:\n${error.message || error}`; if(status)status.textContent='Request failed.'; }
    } finally {
      state.abortController=null;
      if(!quick){ $('abortPlayground').disabled=true; $('runPlayground').disabled=false; }
    }
  }

  function renderNativeInspector() {
    const m=state.lastNativeMetrics; const waterfall=$('timingWaterfall'); const facts=$('requestFacts'); if(!waterfall||!facts)return;
    waterfall.textContent=''; facts.textContent='';
    if(!m){ const e=document.createElement('div'); e.className='empty-state'; e.textContent='Run a non-streaming Native AIR request to populate detailed request metrics.'; waterfall.appendChild(e); return; }
    const stages=[['Queue',m.queue_ms],['Plan prep',m.plan_preparation_ms],['Prefill',m.prefill_ms],['TTFT',m.ttft_ms],['Decode',m.decode_ms]];
    const max=Math.max(...stages.map(([,v])=>num(v)),1);
    stages.forEach(([name,v])=>{ const row=document.createElement('div'); row.className='waterfall-row'; const label=document.createElement('span'); label.textContent=name; const track=document.createElement('div'); track.className='waterfall-track'; const bar=document.createElement('i'); bar.style.width=`${Math.max(1,pct(v,max))}%`; track.appendChild(bar); const strong=document.createElement('strong'); strong.textContent=fmtMs(v); row.append(label,track,strong); waterfall.appendChild(row); });
    const values=[['Total',fmtMs(m.total_ms)],['Request ID',m.request_id],['Sequence ID',m.sequence_id],['Strategy',m.strategy_id],['Planner mode',m.planner_mode],['Prefix reused',`${fmtNum(m.prefix_reused_tokens)} tok`],['Prefill speed',`${fmtNum(m.prefill_tokens_per_second)} tok/s`],['Decode speed',`${fmtNum(m.decode_tokens_per_second)} tok/s`],['KV bytes',fmtBytes(m.kv_bytes)],['Prepared bytes',fmtBytes(m.plan_preparation_bytes)]];
    values.forEach(([label,value])=>{ const div=document.createElement('div'); const s=document.createElement('span'); s.textContent=label; const strong=document.createElement('strong'); strong.textContent=val(value); div.append(s,strong); facts.appendChild(div); });
  }

  function activateView(name) {
    document.querySelectorAll('[data-view-section]').forEach(el=>el.classList.toggle('active',el.dataset.viewSection===name));
    document.querySelectorAll('.nav-link').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
    history.replaceState(null,'',`#${name}`);
    window.scrollTo({top:0,behavior:'smooth'});
    if(name==='diagnostics') refreshMetrics();
    if(name==='model') refreshModelData();
  }

  function bindUI() {
    document.querySelectorAll('.nav-link').forEach(btn=>btn.addEventListener('click',()=>activateView(btn.dataset.view)));
    document.querySelectorAll('[data-jump]').forEach(btn=>btn.addEventListener('click',()=>activateView(btn.dataset.jump)));
    $('refreshNow')?.addEventListener('click',()=>{ clearTimeout(state.pollTimer); pollOnce(); });
    $('overviewEventFilter')?.addEventListener('input',renderEvents);
    $('eventSearch')?.addEventListener('input',renderDiagnosticRows);
    $('eventTypeFilter')?.addEventListener('change',renderDiagnosticRows);
    $('refreshMetrics')?.addEventListener('click',refreshMetrics);
    document.querySelectorAll('#rawApiTabs .segment').forEach(btn=>btn.addEventListener('click',()=>{ state.rawTab=btn.dataset.raw; document.querySelectorAll('#rawApiTabs .segment').forEach(x=>x.classList.toggle('active',x===btn)); renderRawApi(); }));
    document.querySelectorAll('#playgroundModeTabs .segment').forEach(btn=>btn.addEventListener('click',()=>{ state.playgroundMode=btn.dataset.mode; document.querySelectorAll('#playgroundModeTabs .segment').forEach(x=>x.classList.toggle('active',x===btn)); }));
    $('quickRun')?.addEventListener('click',()=>runRequest({ prompt:$('quickPrompt').value, mode:'native', stream:$('quickStream').checked, values:{maxTokens:num($('quickMaxTokens').value),temperature:num($('quickTemperature').value),topP:1,topK:0,seed:0}, output:$('quickResponse'), status:null, quick:true }));
    $('runPlayground')?.addEventListener('click',()=>runRequest({ prompt:$('playgroundPrompt').value, mode:state.playgroundMode, stream:$('streamToggle').checked, values:{maxTokens:num($('maxTokens').value),temperature:num($('temperature').value),topP:num($('topP').value),topK:num($('topK').value),seed:num($('seed').value)}, output:$('playgroundResponse'), status:$('playgroundStatus') }));
    $('abortPlayground')?.addEventListener('click',()=>state.abortController?.abort());
    document.addEventListener('visibilitychange',()=>{ state.retryMs=document.hidden?3000:800; clearTimeout(state.pollTimer); schedulePoll(); });
    window.addEventListener('hashchange',()=>{ const h=location.hash.slice(1); if(['overview','strategy','playground','diagnostics','model'].includes(h)) activateView(h); });
  }

  function init() {
    bindUI();
    const start=location.hash.slice(1); if(['overview','strategy','playground','diagnostics','model'].includes(start)) activateView(start);
    renderNativeInspector();
    pollOnce();
  }

  init();
})();
