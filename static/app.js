const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const state = { imageData: null, imageName: '', imageSize: 0, lastResult: null, batchResults: [], demoCases: [], batchFileName: '', mediaFile: null, mediaObjectUrl: null, mediaResult: null, promiseVideoId: null, promiseMetadata: null, promiseResult: null, youtubeNextPageToken: null, youtubeLastQuery: "" };
const pageLabels = { overview: 'Overview', analyze: 'Video analyzer', media: 'Worth watching', promise: 'Title Promise', batch: 'Batch audit', model: 'Model lab', extension: 'Browser extension' };
const toastEl = $('#toast');
let toastTimer;

function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2700);
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
function dateLabel() {
  const now = new Date();
  $('#today-label').textContent = now.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}
function goToPage(name) {
  if (!pageLabels[name]) return;
  $$('.page-section').forEach(section => section.classList.toggle('active', section.id === `page-${name}`));
  $$('.nav-link').forEach(button => button.classList.toggle('active', button.dataset.page === name));
  $('#crumb-current').textContent = pageLabels[name];
  if (name === 'overview') refreshDashboard();
  if (name === 'model') refreshModelDetails();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
$$('.nav-link').forEach(button => button.addEventListener('click', () => goToPage(button.dataset.page)));
$$('[data-go]').forEach(button => button.addEventListener('click', () => goToPage(button.dataset.go)));
$('#refresh-btn').addEventListener('click', async () => { await Promise.all([refreshHealth(), refreshDashboard(), refreshHistory(), refreshModelDetails()]); toast('Workspace refreshed'); });

async function api(path, options = {}) {
  const headers = { ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status})`);
  return body;
}
async function refreshHealth() {
  try {
    const health = await api('/api/health');
    $('#api-status-text').textContent = 'Local API connected';
    $('.api-pill').classList.add('online'); $('.api-pill').classList.remove('offline');
    updateModelBanner(health);
  } catch (error) {
    $('#api-status-text').textContent = 'Local API unavailable';
    $('.api-pill').classList.add('offline'); $('.api-pill').classList.remove('online');
    $('#metric-mode').textContent = 'API OFFLINE';
    $('#metric-model-desc').textContent = 'Start the local server to use analysis features.';
  }
}
function updateModelBanner(health) {
  const trained = health.mode === 'trained_model' && health.model_loaded;
  $('#metric-mode').textContent = trained ? 'TRAINED DL' : 'DEMO MODE';
  $('#metric-mode').style.color = trained ? 'var(--mint)' : 'var(--amber)';
  $('#metric-model-desc').textContent = trained ? 'BiGRU + CNN gated fusion is active.' : 'Explainable rule-based scoring is active. Train a checkpoint to switch to DL inference.';
  $('#model-lab-name').textContent = health.model_name || 'Model status unavailable';
  $('#model-lab-desc').textContent = health.message || '';
  $('#model-status-pill').textContent = trained ? 'TRAINED MODEL' : 'DEMO MODE';
  $('#model-status-pill').style.color = trained ? 'var(--mint)' : 'var(--amber)';
  $('#model-status-pill').style.borderColor = trained ? '#69e3bc35' : '#f7bf6935';
  $('#model-lab-badge').innerHTML = `<i></i> ${trained ? 'TRAINED CHECKPOINT ACTIVE' : 'DEMO MODE · NO CHECKPOINT'}`;
}
async function refreshDashboard() {
  try {
    const data = await api('/api/dashboard');
    $('#metric-total').textContent = data.total_analyses;
    $('#metric-average').textContent = Math.round(data.average_risk || 0);
    $('#metric-average-bar').style.width = `${Math.max(0, Math.min(100, data.average_risk || 0))}%`;
    $('#metric-high').textContent = data.high_risk;
    const total = Math.max(1, data.low_risk + data.review + data.high_risk);
    $('#dist-low').style.width = `${data.low_risk / total * 100}%`;
    $('#dist-review').style.width = `${data.review / total * 100}%`;
    $('#dist-high').style.width = `${data.high_risk / total * 100}%`;
    if (data.model) updateModelBanner(data.model);
  } catch (_) { /* Keep the dashboard readable when the API is offline. */ }
}
function riskClass(score) { return score >= 62 ? 'high' : score >= 35 ? 'review' : 'low'; }
function formatWhen(value) {
  if (!value) return 'Just now';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? 'Saved locally' : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
async function refreshHistory() {
  try {
    const { items } = await api('/api/history?limit=6');
    const container = $('#recent-list');
    if (!items.length) {
      container.innerHTML = '<div class="empty-state compact"><div class="empty-illustration">◉</div><strong>Your analysis history starts here</strong><p>Analyze a title and thumbnail to populate this feed.</p></div>';
      return;
    }
    container.innerHTML = items.map(item => {
      const klass = riskClass(item.risk_score);
      return `<div class="recent-item"><div class="recent-symbol ${klass}">${klass === 'high' ? '⚑' : klass === 'review' ? '⌁' : '✓'}</div><div class="recent-info"><strong title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</strong><small>${escapeHtml(formatWhen(item.created_at))} · ${escapeHtml(item.model_mode === 'trained_model' ? 'Trained DL' : 'Demo scorer')}</small></div><span class="risk-mini ${klass}">${item.risk_score}/100</span></div>`;
    }).join('');
  } catch (_) { /* ignore */ }
}

function bytesLabel(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
function setImage(dataUrl, name = 'sample-thumbnail.jpg', size = 0) {
  state.imageData = dataUrl; state.imageName = name; state.imageSize = size;
  if (dataUrl) {
    $('#thumbnail-preview').src = dataUrl;
    $('#thumbnail-file-name').textContent = name;
    $('#thumbnail-file-size').textContent = size ? `${bytesLabel(size)} · ready for analysis` : 'Sample thumbnail · ready for analysis';
    $('#thumbnail-preview-wrap').classList.remove('hidden');
    $('#upload-zone').classList.add('hidden');
    $('#remove-image').disabled = false;
  } else {
    $('#thumbnail-preview-wrap').classList.add('hidden');
    $('#upload-zone').classList.remove('hidden');
    $('#thumbnail-file').value = '';
    $('#remove-image').disabled = true;
  }
}
function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Could not read that image file.'));
    reader.readAsDataURL(file);
  });
}
$('#thumbnail-file').addEventListener('change', async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { toast('Image is larger than 5 MB. Choose a smaller thumbnail.'); event.target.value = ''; return; }
  if (!file.type.startsWith('image/')) { toast('Please choose a supported image.'); event.target.value = ''; return; }
  try { setImage(await readAsDataURL(file), file.name, file.size); }
  catch (error) { toast(error.message); }
});
$('#remove-image').addEventListener('click', () => { setImage(null); toast('Thumbnail removed'); });
const uploadZone = $('#upload-zone');
['dragenter', 'dragover'].forEach(name => uploadZone.addEventListener(name, event => { event.preventDefault(); uploadZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(name => uploadZone.addEventListener(name, event => { event.preventDefault(); uploadZone.classList.remove('dragover'); }));
uploadZone.addEventListener('drop', async event => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  if (!file.type.startsWith('image/') || file.size > 5 * 1024 * 1024) { toast('Choose an image file under 5 MB.'); return; }
  try { setImage(await readAsDataURL(file), file.name, file.size); }
  catch (error) { toast(error.message); }
});
$('#video-title').addEventListener('input', event => { $('#title-counter').textContent = `${event.target.value.length} / 500 characters`; });

async function loadDemoCases() {
  try {
    const { items } = await api('/api/demo-cases');
    state.demoCases = items;
    $('#example-list').innerHTML = items.slice(0, 3).map((item, index) => `<button class="example-chip" data-example="${index}">${escapeHtml(item.short_name)} <span>↗</span></button>`).join('');
    $$('.example-chip').forEach(button => button.addEventListener('click', () => loadExample(Number(button.dataset.example))));
  } catch (_) { /* built-in examples may be unavailable while offline */ }
}
async function loadExample(index) {
  const item = state.demoCases[index];
  if (!item) return;
  $('#video-title').value = item.title;
  $('#title-counter').textContent = `${item.title.length} / 500 characters`;
  setImage(null);
  if (item.image) {
    try {
      const response = await fetch(item.image);
      if (response.ok) {
        const blob = await response.blob();
        setImage(await readAsDataURL(new File([blob], item.image.split('/').pop(), { type: blob.type || 'image/jpeg' })), item.image.split('/').pop(), blob.size);
      }
    } catch (_) { /* title-only analysis still works */ }
  }
  goToPage('analyze');
  toast('Example loaded — ready to analyze');
}
$('#analyze-btn').addEventListener('click', async () => {
  const title = $('#video-title').value.trim();
  if (title.length < 3) { toast('Enter a title with at least 3 characters.'); $('#video-title').focus(); return; }
  const button = $('#analyze-btn');
  button.disabled = true; button.innerHTML = 'Analyzing signals <span class="loading-dot">···</span>';
  try {
    const result = await api('/api/analyze', { method: 'POST', body: JSON.stringify({ title, image_data: state.imageData, save_history: true }) });
    state.lastResult = result;
    renderResult(result);
    await Promise.all([refreshDashboard(), refreshHistory()]);
    toast('Analysis complete');
  } catch (error) {
    toast(error.message || 'Analysis could not be completed. Is the local API running?');
  } finally {
    button.disabled = false; button.innerHTML = 'Analyze signals <span>↗</span>';
  }
});
function renderResult(result) {
  $('#result-empty').classList.add('hidden');
  const content = $('#result-content'); content.classList.remove('hidden');
  const klass = riskClass(result.risk_score);
  const components = result.components || {};
  const componentLabels = { sensational_language: 'Sensational language', curiosity_gap: 'Curiosity gap', urgency: 'Urgency cues', visual_intensity: 'Thumbnail intensity' };
  const bars = Object.keys(componentLabels).filter(key => components[key] !== null && components[key] !== undefined).map(key => `<div class="signal-bar-row"><span>${componentLabels[key]}</span><div class="signal-bar"><i style="width:${Math.min(100, Math.max(0, components[key]))}%"></i></div><b>${components[key]}%</b></div>`).join('');
  const reasons = (result.reasons || []).map(reason => `<li><span class="reason-bullet">↗</span><span>${escapeHtml(reason)}</span></li>`).join('');
  const visual = result.visual_metrics || {};
  const visualStats = result.image_available ? [
    ['Saturation', visual.saturation], ['Contrast', visual.contrast], ['Brightness', visual.brightness]
  ].filter(row => row[1] !== undefined).map(([label, value]) => `<div class="visual-stat"><span>${label}</span><strong>${Math.round(Number(value) * 100)}<small>%</small></strong></div>`).join('') : '<div class="visual-stat" style="grid-column:1/-1"><span>Thumbnail</span><strong>Not provided</strong></div>';
  content.innerHTML = `
    <div class="result-score-head"><div><div class="result-risk-label">SCREENING ASSESSMENT</div><div class="result-label ${klass}">${escapeHtml(result.label)}</div><div class="result-submeta" style="justify-content:flex-start;gap:8px"><span class="risk-badge ${klass}">${escapeHtml(result.risk_band)} RISK</span><span>${result.image_available ? 'Text + thumbnail' : 'Title only'}</span></div></div><div><div class="score-number">${result.risk_score}<small> / 100</small></div></div><div style="position:absolute"></div></div>
    <div class="score-meter"><span class="${klass}" style="width:${result.risk_score}%"></span></div>
    <div class="result-submeta"><span>Risk index</span><span>${escapeHtml(result.model_mode === 'trained_model' ? 'Trained model output' : 'Heuristic demo index')}</span></div>
    <div class="result-section-title">SIGNAL BREAKDOWN</div><div class="signal-bars">${bars}</div>
    <div class="result-section-title">WHY THIS WAS FLAGGED</div><ul class="reason-list">${reasons}</ul>
    <div class="result-section-title">THUMBNAIL STYLE DIAGNOSTICS</div><div class="visual-stats">${visualStats}</div>
    <div class="result-disclaimer"><span>ⓘ</span><p>${escapeHtml(result.disclaimer || 'This is a screening aid, not a factual truth verdict.')}</p></div>
    <div class="result-footer-actions"><button class="outline-button" id="copy-result-btn">⎘ Copy report</button><button class="outline-button" id="new-analysis-btn">↻ New analysis</button></div>`;
  $('#copy-result-btn').addEventListener('click', async () => {
    const report = [`VERI-CLICK AI SCREENING REPORT`, `Title: ${result.title || $('#video-title').value}`, `Label: ${result.label}`, `Risk index: ${result.risk_score}/100`, `Mode: ${result.model_name}`, `Reasons: ${(result.reasons || []).join('; ')}`, `Note: ${result.disclaimer}`].join('\n');
    try { await navigator.clipboard.writeText(report); toast('Report copied'); }
    catch (_) { toast('Clipboard access was blocked by the browser.'); }
  });
  $('#new-analysis-btn').addEventListener('click', () => { $('#video-title').focus(); $('#video-title').select(); });
}

async function refreshMediaCapabilities() {
  const note = $('#media-engine-note');
  try {
    const capabilities = await api('/api/media-capabilities');
    if (capabilities.speech_ai_installed) {
      note.innerHTML = `<span class="media-engine-ready">● Speech AI ready</span> · Whisper <strong>${escapeHtml(capabilities.model)}</strong> · first use downloads its model; use a short clip on a low-RAM laptop.`;
    } else {
      note.innerHTML = `<span class="media-engine-missing">● Speech AI setup needed</span><br>In the activated project terminal, run <code>pip install -r requirements-media.txt</code>.`;
    }
  } catch (_) {
    note.textContent = 'Start the local API first. Media analysis runs on your computer.';
  }
}

const MEDIA_EXTENSIONS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.mpeg', '.mpg', '.m4v', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma']);
function mediaExtension(filename) {
  const match = String(filename || '').toLowerCase().match(/\.[a-z0-9]+$/);
  return match ? match[0] : '';
}
function setMediaFile(file) {
  if (state.mediaObjectUrl) URL.revokeObjectURL(state.mediaObjectUrl);
  state.mediaFile = file || null;
  state.mediaObjectUrl = null;
  state.mediaResult = null;
  $('#media-result-content').classList.add('hidden');
  $('#media-result-empty').classList.remove('hidden');
  $('#media-analyze-btn').disabled = !file;
  if (!file) {
    $('#media-file').value = '';
    $('#media-upload-zone').classList.remove('hidden');
    $('#media-preview-wrap').classList.add('hidden');
    $('#media-file-preview').innerHTML = '';
    return;
  }
  if (!MEDIA_EXTENSIONS.has(mediaExtension(file.name))) {
    toast('Choose a supported video or audio file.');
    setMediaFile(null);
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    toast('File exceeds 200 MB. Try a shorter/smaller clip.');
    setMediaFile(null);
    return;
  }
  state.mediaObjectUrl = URL.createObjectURL(file);
  const ext = mediaExtension(file.name);
  const isVideo = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.mpeg', '.mpg', '.m4v'].includes(ext);
  const player = isVideo
    ? `<video id="media-preview-video" controls preload="metadata" src="${state.mediaObjectUrl}"></video>`
    : `<audio id="media-preview-audio" controls preload="metadata" src="${state.mediaObjectUrl}"></audio>`;
  $('#media-file-preview').innerHTML = player;
  $('#media-upload-zone').classList.add('hidden');
  $('#media-file-name').textContent = file.name;
  $('#media-file-size').textContent = `${bytesLabel(file.size)} · ${isVideo ? 'video + audio expected' : 'audio'} · selected locally`;
  $('#media-preview-wrap').classList.remove('hidden');
  $('#media-upload-zone').classList.add('hidden');
}
$('#media-file').addEventListener('change', event => {
  const file = event.target.files?.[0];
  if (file) setMediaFile(file);
});
$('#remove-media-file').addEventListener('click', () => { setMediaFile(null); toast('Media file removed'); });
$('#media-analyze-btn').addEventListener('click', async () => {
  const file = state.mediaFile;
  const goal = $('#media-goal').value.trim();
  const title = $('#media-title').value.trim();
  if (!file) { toast('Choose a video or audio file first.'); return; }
  if (goal.length < 5) { toast('Tell us what you want from this video so we can estimate relevance.'); $('#media-goal').focus(); return; }
  const button = $('#media-analyze-btn');
  button.disabled = true;
  button.innerHTML = 'Transcribing & analyzing <span class="loading-dot">···</span>';
  $('#media-engine-note').textContent = 'Processing locally: loading speech model, transcribing audio, sampling video frames, and comparing the transcript with your goal. First run can take a while.';
  const formData = new FormData();
  formData.append('file', file, file.name);
  formData.append('title', title);
  formData.append('goal', goal);
  try {
    const result = await api('/api/media-analyze', { method: 'POST', body: formData });
    state.mediaResult = result;
    renderMediaResult(result);
    toast('Media analysis complete');
  } catch (error) {
    toast(error.message || 'Media analysis failed. Check the optional speech setup.');
    $('#media-engine-note').textContent = error.message || 'Media analysis failed.';
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze content <span>↗</span>';
  }
});
function renderMediaResult(result) {
  $('#media-result-empty').classList.add('hidden');
  const target = $('#media-result-content');
  target.classList.remove('hidden');
  const score = result.worth_score;
  const className = score === null || score === undefined ? 'fit-medium' : score >= 60 ? 'fit-strong' : score >= 35 ? 'fit-medium' : 'fit-weak';
  const scoreMarkup = score === null || score === undefined ? '<span class="media-no-score">Add a viewing goal</span>' : `<div class="media-fit-score ${className}">${score}<small>/100</small></div>`;
  const contextScore = result.context_match_score === null || result.context_match_score === undefined ? 'Not scored' : `${result.context_match_score}/100`;
  const specificity = Number(result.content_specificity_score ?? 0);
  const summary = (result.summary || []).map(sentence => `<li>${escapeHtml(sentence)}</li>`).join('') || '<li>Not enough speech to extract a summary.</li>';
  const moments = (result.key_moments || []).map((moment, index) => `<div class="media-moment"><button class="moment-seek" data-seek="${Number(moment.timestamp_seconds) || 0}" title="Seek preview to ${escapeHtml(moment.timestamp_label)}">${escapeHtml(moment.timestamp_label)} <span>↗</span></button><p>${escapeHtml(moment.excerpt || 'Speech detected in this interval.')}</p></div>`).join('') || '<p class="media-muted">No timestamped speech segments available.</p>';
  const frames = (result.sampled_frames || []).map(frame => `<figure class="media-frame"><img src="${frame.data_url}" alt="Sampled frame near ${escapeHtml(frame.timestamp_label)}" loading="lazy"><figcaption>${escapeHtml(frame.timestamp_label)} · brightness ${Math.round(Number(frame.brightness || 0) * 100)}% · contrast ${Math.round(Number(frame.contrast || 0) * 100)}%</figcaption></figure>`).join('');
  const transcript = escapeHtml(result.transcript || 'No transcript text.');
  const duration = escapeHtml(result.duration_label || 'Unknown duration');
  const title = escapeHtml(result.title || result.filename || 'Uploaded media');
  target.innerHTML = `
    <div class="media-verdict-head"><div><div class="result-risk-label">WORTH-WATCHING ASSESSMENT</div><h3>${escapeHtml(result.recommendation || 'Content analysed')}</h3><p>${title}</p></div><div class="media-fit-score-wrap">${scoreMarkup}<small>Preliminary fit score</small></div></div>
    <div class="media-metric-grid"><div class="media-metric"><span>GOAL KEYWORD MATCH</span><strong>${contextScore}</strong><small>Transcript terms matching your goal</small></div><div class="media-metric"><span>CONTENT SPECIFICITY</span><strong>${specificity}/100</strong><small>Heuristic, not a quality benchmark</small></div><div class="media-metric"><span>MEDIA DETAILS</span><strong>${duration}</strong><small>${Number(result.word_count || 0).toLocaleString()} words · ${escapeHtml(result.language || 'unknown')} · ${escapeHtml(result.media_kind || '')}</small></div></div>
    <div class="media-score-basis">SCORE METHOD · 75% goal keyword match + 25% content-specificity estimate. This is a heuristic, not an 86% or other probability.</div>
    <div class="media-explanation"><strong>Why this recommendation?</strong><p>${escapeHtml(result.recommendation_reason || result.context_explanation || '')}</p>${(result.context_match_terms || []).length ? `<div class="media-term-chips">${result.context_match_terms.map(term => `<span>${escapeHtml(term)}</span>`).join('')}</div>` : ''}</div>
    <div class="media-section-title">CONTENT SUMMARY <span>EXTRACTED FROM TRANSCRIPT</span></div><ul class="media-summary-list">${summary}</ul>
    <div class="media-section-title">KEY MOMENTS <span>TRANSCRIPT WINDOWS</span></div><div class="media-moments">${moments}</div>
    ${frames ? `<div class="media-section-title">SAMPLED VIDEO FRAMES <span>UP TO 3 FRAMES</span></div><div class="media-frames">${frames}</div><p class="media-muted">${escapeHtml(result.frame_sampling_note || '')}</p>` : ''}
    <details class="media-transcript"><summary>Read transcript (${Number(result.word_count || 0).toLocaleString()} words)</summary><pre>${transcript}${result.transcript_truncated ? '\n\n[Transcript preview truncated for safety.]' : ''}</pre></details>
    ${result.title_promise_analysis ? renderTitlePromiseInline(result.title_promise_analysis) : ''}
    <p class="media-disclaimer">${escapeHtml(result.disclaimer || '')}</p>
    <div class="media-result-actions"><button class="outline-button" id="media-download-report">↓ Download report</button><button class="outline-button" id="media-copy-summary">⎘ Copy summary</button></div>`;
  $$('.moment-seek', target).forEach(button => button.addEventListener('click', () => {
    const player = $('#media-preview-video') || $('#media-preview-audio');
    if (!player) { toast('Preview player is unavailable for this file format.'); return; }
    try { player.currentTime = Number(button.dataset.seek) || 0; player.play().catch(() => {}); }
    catch (_) { toast('This browser cannot seek the selected media format.'); }
  }));
  const reportText = () => [
    'VERI-CLICK AI — WORTH WATCHING REPORT',
    `File: ${result.filename}`, `Title: ${result.title || ''}`, `Viewing goal: ${result.goal || ''}`,
    `Recommendation: ${result.recommendation}`, `Preliminary fit score: ${result.worth_score ?? 'not scored'}/100`,
    `Goal keyword match: ${result.context_match_score ?? 'not scored'}/100`,
    `Content specificity estimate: ${result.content_specificity_score}/100`,
    ...(result.title_promise_analysis ? ['', 'TITLE PROMISE:', JSON.stringify(result.title_promise_analysis, null, 2)] : []),
    '', 'SUMMARY:', ...(result.summary || []).map(text => `- ${text}`),
    '', 'KEY MOMENTS:', ...(result.key_moments || []).map(moment => `- ${moment.timestamp_label}: ${moment.excerpt}`),
    '', 'TRANSCRIPT:', result.transcript || '', '', `Limitations: ${result.disclaimer || ''}`,
  ].join('\n');
  $('#media-download-report').addEventListener('click', () => {
    const blob = new Blob([reportText()], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'vericlick_worth_watching_report.txt'; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
  });
  $('#media-copy-summary').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(reportText()); toast('Media report copied'); }
    catch (_) { toast('Clipboard was blocked by the browser. Use Download report instead.'); }
  });
}

function parseCSV(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(field); field = ''; }
    else if (char === '\n') { row.push(field.replace(/\r$/, '')); if (row.some(cell => cell.trim())) rows.push(row); row = []; field = ''; }
    else field += char;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, '')); if (row.some(cell => cell.trim())) rows.push(row); }
  return rows;
}
$('#batch-file').addEventListener('change', async event => {
  const file = event.target.files?.[0]; if (!file) return;
  if (file.size > 2 * 1024 * 1024) { toast('CSV exceeds the 2 MB file limit.'); event.target.value = ''; return; }
  try {
    const rows = parseCSV(await file.text());
    if (rows.length < 2) throw new Error('CSV needs a header and at least one data row.');
    const headers = rows[0].map(value => value.trim().toLowerCase());
    const titleIndex = ['video_title', 'title', 'video title', 'headline'].find(key => headers.includes(key));
    if (!titleIndex) throw new Error('Could not find a title column. Try video_title or title.');
    const colIndex = headers.indexOf(titleIndex);
    let items = rows.slice(1).map(row => ({ title: (row[colIndex] || '').trim() })).filter(item => item.title.length >= 3);
    if (!items.length) throw new Error('No valid titles were found in that CSV.');
    if (items.length > 200) { toast('Only the first 200 valid rows will be audited.'); items = items.slice(0, 200); }
    state.batchItems = items; state.batchFileName = file.name;
    $('#batch-file-name').textContent = file.name;
    $('#batch-file-meta').textContent = `${items.length} valid title(s) · ${bytesLabel(file.size)}`;
    $('#batch-file-row').classList.remove('hidden'); $('#batch-analyze-btn').disabled = false;
    toast(`${items.length} title(s) ready for audit`);
  } catch (error) { toast(error.message); }
});
$('#clear-batch-file').addEventListener('click', () => { state.batchItems = []; $('#batch-file').value = ''; $('#batch-file-row').classList.add('hidden'); $('#batch-analyze-btn').disabled = true; });
$('#download-demo-csv').addEventListener('click', () => downloadCSV('vericlick_sample_titles.csv', [
  { video_title: 'STOP! You will not believe this unbelievable secret', label: 1 },
  { video_title: 'How to replace a light switch safely: step-by-step guide', label: 0 },
  { video_title: 'The truth about this viral trend revealed', label: 1 },
  { video_title: 'Introduction to calculus: limits explained with examples', label: 0 },
  { video_title: 'I tried this hack and the result shocked everyone!', label: 1 },
  { video_title: 'A beginner guide to understanding neural networks', label: 0 },
]));
$('#batch-analyze-btn').addEventListener('click', async () => {
  const items = state.batchItems || [];
  if (!items.length) return;
  const button = $('#batch-analyze-btn'); button.disabled = true; button.textContent = 'Auditing titles…';
  try {
    const data = await api('/api/batch', { method: 'POST', body: JSON.stringify({ items }) });
    state.batchResults = data.items;
    renderBatchResults(data);
    toast('Batch audit complete');
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.innerHTML = 'Run batch audit <span>↗</span>'; }
});
function renderBatchResults(data) {
  const { summary, items } = data;
  $('#batch-result-subtitle').textContent = `${summary.total} title(s) screened · average risk ${summary.average_risk}/100`;
  const tiles = [['TOTAL TITLES', summary.total], ['HIGH RISK', summary.high_risk], ['NEEDS REVIEW', summary.review], ['AVERAGE RISK', `${summary.average_risk}/100`]];
  $('#batch-summary').innerHTML = tiles.map(([label, value]) => `<div class="summary-tile"><span>${label}</span><strong>${value}</strong></div>`).join('');
  $('#batch-summary').classList.remove('hidden'); $('#download-results-btn').disabled = false;
  $('#batch-results-body').innerHTML = items.map(item => {
    const klass = riskClass(item.risk_score); const reason = (item.reasons || [])[0] || 'No strong cue';
    return `<tr><td>${item.row}</td><td><span class="table-title" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</span></td><td><strong class="risk-mini ${klass}">${item.risk_score}/100</strong></td><td><span class="risk-badge ${klass}">${escapeHtml(item.risk_band)}</span></td><td>${escapeHtml(reason)}</td></tr>`;
  }).join('');
}
function downloadCSV(fileName, rows) {
  if (!rows?.length) { toast('No results to export yet.'); return; }
  const columns = Object.keys(rows[0]);
  const cell = value => { const str = String(value ?? ''); return /[",\r\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str; };
  const body = [columns.map(cell).join(','), ...rows.map(row => columns.map(column => cell(row[column])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([body], { type: 'text/csv;charset=utf-8;' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = fileName; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
}
$('#download-results-btn').addEventListener('click', () => {
  const rows = (state.batchResults || []).map(item => ({ row: item.row, title: item.title, risk_score: item.risk_score, label: item.label, risk_band: item.risk_band, model_mode: item.model_mode, reasons: (item.reasons || []).join(' | ') }));
  downloadCSV('vericlick_batch_audit.csv', rows);
});
$('#download-model-comparison').addEventListener('click', () => { window.location.href = '/api/model-comparison.csv'; });

async function refreshModelDetails() {
  try {
    const data = await api('/api/model-details');
    const map = {
      'multimodal_model.pt': 'Learned neural-network weights for the primary fusion model',
      'vocab.json': 'Vocabulary built from the training split',
      'model_config.json': 'Architecture, dataset and split metadata',
      'test_metrics.json': 'Held-out test metrics and confusion matrix',
      'model_comparison_metrics.json': 'Metrics for all four comparable DL experiments',
      'model_comparison.csv': 'Downloadable model comparison table',
      'comparison_models/bigru_text.pt': 'BiGRU title-only checkpoint',
      'comparison_models/cnn_image.pt': 'CNN thumbnail-only checkpoint',
      'comparison_models/mlp_features.pt': 'Engineered-feature MLP checkpoint',
      'comparison_models/multimodal_fusion.pt': 'Multimodal fusion comparison checkpoint'
    };
    $('#artifact-list').innerHTML = data.artifacts.map(item => `<div class="artifact-row"><span class="artifact-symbol">${item.name.endsWith('.pt') ? '◈' : item.name.endsWith('.csv') ? '≋' : '▤'}</span><div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(map[item.name] || 'Training output')}</small></div><span class="artifact-state ${item.exists ? 'ready' : 'missing'}">${item.exists ? 'AVAILABLE' : 'NOT YET'}</span></div>`).join('');
    if (data.test_metrics?.test) {
      const metric = data.test_metrics.test;
      const detail = document.createElement('div'); detail.className = 'metric-disclaimer';
      detail.innerHTML = `<strong style="color:#b5a4ff">Primary fusion model · held-out test</strong><br>Accuracy ${(metric.accuracy * 100).toFixed(1)}% · Precision ${(metric.precision * 100).toFixed(1)}% · Recall ${(metric.recall * 100).toFixed(1)}% · F1 ${Number(metric.f1).toFixed(3)}<br><span style="color:#8a94ae">Evaluated on ${data.test_metrics.config?.test_rows ?? '?'} held-out row(s). Results are dataset-specific.</span>`;
      $('#artifact-list').appendChild(detail);
    }
    const summary = $('#comparison-summary');
    const body = $('#comparison-results-body');
    $('#download-model-comparison').disabled = !(data.comparison?.status === 'completed');
    if (data.comparison?.status === 'completed' && Array.isArray(data.comparison.models) && data.comparison.models.length) {
      const split = data.comparison.split || {};
      summary.textContent = `${data.comparison.valid_rows} labeled title+thumbnail rows · ${split.train} train / ${split.validation} validation / ${split.test} held-out test · ${data.comparison.device} · ${data.comparison.elapsed_seconds}s. Same split and label definition for every model.`;
      body.innerHTML = data.comparison.models.map(item => {
        const m = item.test_metrics || {};
        const cm = m.confusion_matrix || [[0, 0], [0, 0]];
        const tn = cm[0]?.[0] ?? 0, fp = cm[0]?.[1] ?? 0, fn = cm[1]?.[0] ?? 0, tp = cm[1]?.[1] ?? 0;
        const pct = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
        const auc = m.roc_auc === null || m.roc_auc === undefined ? '—' : pct(m.roc_auc);
        return `<tr><td><strong>${escapeHtml(item.name || item.key)}</strong>${item.key === data.comparison.primary_model ? '<small class="comparison-primary">PRIMARY INFERENCE MODEL</small>' : ''}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td><strong>${pct(m.f1)}</strong></td><td>${auc}</td><td>${tn} / ${fp} · ${fn} / ${tp}</td><td>${Number(item.training_seconds || 0).toFixed(1)}s</td></tr>`;
      }).join('');
    } else {
      summary.textContent = 'No comparison run yet. Train all four models to populate this table; no result values are fabricated.';
      body.innerHTML = '<tr><td colspan="8"><div class="table-empty"><span>⌘</span><strong>Comparison not run yet</strong><p>Results appear here after real training on labeled titles and matching thumbnails.</p></div></td></tr>';
    }
  } catch (_) { /* retain initial not-yet-trained status */ }
}
$$('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(button.dataset.copy); toast('Training command copied'); }
  catch (_) { toast('Clipboard access was blocked by the browser.'); }
}));

// Title Promise Checker: embedded YouTube playback + public captions / pasted transcript.
function extractYouTubeId(raw) {
  const value = String(raw || '').trim();
  if (/^[A-Za-z0-9_-]{11}$/.test(value)) return value;
  let parsed;
  try { parsed = new URL(value.includes('://') ? value : `https://${value}`); }
  catch (_) { return null; }
  const host = parsed.hostname.toLowerCase().replace(/\.$/, '');
  if (!['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be', 'youtube-nocookie.com', 'www.youtube-nocookie.com'].includes(host)) return null;
  let id = parsed.searchParams.get('v') || '';
  if (!id && host === 'youtu.be') id = parsed.pathname.split('/').filter(Boolean)[0] || '';
  if (!id) {
    const parts = parsed.pathname.split('/').filter(Boolean);
    for (const marker of ['embed', 'shorts', 'live', 'v']) {
      const index = parts.indexOf(marker);
      if (index >= 0 && parts[index + 1]) { id = parts[index + 1]; break; }
    }
  }
  return /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null;
}
function promiseSetNote(message, isError = false) {
  const note = $('#promise-engine-note');
  if (note) { note.textContent = message; note.classList.toggle('promise-error', isError); }
  const source = $('#promise-source-note');
  if (source && isError) { source.textContent = message; source.classList.add('promise-error'); }
}
function updatePromiseCheckButton() {
  $('#promise-check-btn').disabled = !state.promiseVideoId || $('#promise-video-title').value.trim().length < 3;
}
async function loadPromiseVideo(fallbackTitle = '', fallbackAuthor = '') {
  const raw = $('#promise-video-url').value.trim();
  const videoId = extractYouTubeId(raw);
  if (!videoId) { toast('Paste a valid YouTube video URL or 11-character video ID.'); return; }
  state.promiseVideoId = videoId;
  state.promiseMetadata = { video_id: videoId, title: '', author_name: '' };
  state.promiseResult = null;
  $('#promise-video-title').value = fallbackTitle;
  updatePromiseCheckButton();
  $('#promise-result-content').classList.add('hidden');
  $('#promise-result-empty').classList.remove('hidden');
  $('#promise-player-placeholder').classList.add('hidden');
  $('#promise-player-wrap').classList.remove('hidden');
  $('#promise-player-wrap').innerHTML = `<iframe id="promise-youtube-frame" title="YouTube video player" src="https://www.youtube-nocookie.com/embed/${videoId}?rel=0" referrerpolicy="strict-origin-when-cross-origin" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>`;
  const watchUrl = `https://www.youtube.com/watch?v=${videoId}`;
  $('#promise-watch-original').href = watchUrl;
  $('#promise-watch-original').classList.remove('hidden');
  $('#promise-loaded-title').textContent = fallbackTitle || 'Loading public video title…';
  $('#promise-loaded-author').textContent = fallbackAuthor ? `${fallbackAuthor} · ${videoId}` : `Video ID: ${videoId}`;
  $('#promise-source-note').textContent = 'Video embedded. Fetching public metadata; if unavailable, enter the title manually.';
  $('#promise-source-note').classList.remove('promise-error');
  try {
    const meta = await api(`/api/youtube/metadata?video_url=${encodeURIComponent(raw || videoId)}`);
    state.promiseMetadata = meta;
    if (meta.title) $('#promise-video-title').value = meta.title;
    else if (fallbackTitle) $('#promise-video-title').value = fallbackTitle;
    $('#promise-loaded-title').textContent = meta.title || fallbackTitle || 'Enter the title below';
    $('#promise-loaded-author').textContent = meta.author_name ? `${meta.author_name} · ${videoId}` : (fallbackAuthor ? `${fallbackAuthor} · ${videoId}` : `Video ID: ${videoId}`);
    $('#promise-source-note').textContent = meta.title ? 'Video loaded. Public captions will be checked when you run the audit.' : 'Player loaded; automatic title lookup was unavailable. Enter the title manually.';
  } catch (error) {
    $('#promise-video-title').value = fallbackTitle || $('#promise-video-title').value;
    $('#promise-loaded-title').textContent = fallbackTitle || 'Enter the title below';
    $('#promise-loaded-author').textContent = fallbackAuthor ? `${fallbackAuthor} · ${videoId}` : `Video ID: ${videoId}`;
    $('#promise-source-note').textContent = 'Player loaded, but metadata lookup failed. Enter the title manually; captions may still be fetched.';
  }
  updatePromiseCheckButton();
  promiseSetNote('Ready. Leave transcript blank to try public YouTube captions, or paste timestamped transcript lines.');
  toast('YouTube player loaded');
}
$('#promise-load-video').addEventListener('click', () => loadPromiseVideo());
$('#promise-video-url').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); loadPromiseVideo(); } });
$$('[data-promise-pick]').forEach(card => card.addEventListener('click', () => {
  const videoId = card.dataset.promisePick;
  const title = card.dataset.promiseTitle || '';
  const author = card.dataset.promiseAuthor || '';
  $('#promise-video-url').value = `https://www.youtube.com/watch?v=${videoId}`;
  goToPage('promise');
  loadPromiseVideo(title, author);
}));
async function refreshYoutubeSearchStatus() {
  const note = $('#promise-youtube-search-status');
  if (!note) return;
  try {
    const status = await api('/api/youtube/search-status');
    note.textContent = status.message;
    note.classList.toggle('promise-error', !status.configured);
  } catch (error) {
    note.textContent = error.message || 'Could not check YouTube search configuration.';
    note.classList.add('promise-error');
  }
}

function renderYoutubeSearchItems(items, append = false) {
  const container = $('#promise-youtube-results');
  if (!append) container.innerHTML = '';
  if (!items || !items.length) {
    if (!append) container.innerHTML = '<div class="youtube-search-empty"><strong>No videos found</strong><p>Try a broader topic or another search phrase.</p></div>';
    return;
  }
  const cards = items.map(item => `<button class="promise-starter-card promise-search-result" type="button" data-youtube-search-result data-video-id="${escapeHtml(item.video_id)}" data-video-title="${escapeHtml(item.title)}" data-video-channel="${escapeHtml(item.channel_title)}">
    <span class="promise-starter-thumb"><img src="${escapeHtml(item.thumbnail_url)}" alt="" loading="lazy" referrerpolicy="no-referrer"><span class="promise-starter-play">▶</span><span class="promise-starter-duration">YOUTUBE</span></span>
    <span class="promise-starter-copy"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.channel_title)}${item.published_at ? ` · ${escapeHtml(new Date(item.published_at).toLocaleDateString())}` : ''}</small><span>Play &amp; audit title promise ↗</span></span>
  </button>`).join('');
  if (append) container.insertAdjacentHTML('beforeend', cards);
  else container.innerHTML = cards;
  $$('[data-youtube-search-result]', container).forEach(card => {
    if (card.dataset.bound === 'true') return;
    card.dataset.bound = 'true';
    card.addEventListener('click', () => {
      $('#promise-video-url').value = `https://www.youtube.com/watch?v=${card.dataset.videoId}`;
      loadPromiseVideo(card.dataset.videoTitle || '', card.dataset.videoChannel || '');
      $('.promise-player-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      toast('Video selected. Run Check title promise after captions are loaded or paste a transcript.');
    });
  });
}

async function searchYoutubeVideos(loadMore = false) {
  const query = (loadMore ? state.youtubeLastQuery : $('#promise-youtube-query').value).trim();
  if (query.length < 2) {
    toast('Enter at least two characters to search YouTube.');
    $('#promise-youtube-query').focus();
    return;
  }
  const button = loadMore ? $('#promise-youtube-load-more') : $('#promise-youtube-search-btn');
  const note = $('#promise-youtube-search-status');
  const priorText = button.textContent;
  button.disabled = true;
  button.textContent = loadMore ? 'Loading…' : 'Searching…';
  if (!loadMore) {
    state.youtubeNextPageToken = null;
    state.youtubeLastQuery = query;
    $('#promise-youtube-results').innerHTML = '';
  }
  note.classList.remove('promise-error');
  note.textContent = `Searching YouTube for “${query}”…`;
  const params = new URLSearchParams({ q: query, max_results: '12', order: $('#promise-youtube-order').value });
  if (loadMore && state.youtubeNextPageToken) params.set('page_token', state.youtubeNextPageToken);
  try {
    const result = await api(`/api/youtube/search?${params.toString()}`);
    renderYoutubeSearchItems(result.items || [], loadMore);
    state.youtubeNextPageToken = result.next_page_token || null;
    $('#promise-youtube-load-more').classList.toggle('hidden', !state.youtubeNextPageToken);
    note.textContent = `${(result.items || []).length} videos loaded for “${query}” · ${Number(result.total_results_estimate || 0).toLocaleString()} approximate results · ${result.provider}. Select a card to play and audit it.`;
    note.classList.remove('promise-error');
  } catch (error) {
    if (!loadMore) $('#promise-youtube-results').innerHTML = '';
    note.textContent = error.message || 'YouTube search failed.';
    note.classList.add('promise-error');
    if (!loadMore && /not configured|YOUTUBE_API_KEY/i.test(error.message || '')) {
      note.textContent = 'Setup needed: create a Google Cloud API key for YouTube Data API v3, put it in the local .env file as YOUTUBE_API_KEY=your_key, then restart the server. Never paste the key into this search field.';
    }
  } finally {
    button.disabled = false;
    button.textContent = loadMore ? 'Load more results' : 'Search videos ⌕';
  }
}
$('#promise-youtube-search-btn').addEventListener('click', () => searchYoutubeVideos(false));
$('#promise-youtube-query').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); searchYoutubeVideos(false); } });
$('#promise-youtube-order').addEventListener('change', () => { if ($('#promise-youtube-query').value.trim().length >= 2) searchYoutubeVideos(false); });
$('#promise-youtube-load-more').addEventListener('click', () => searchYoutubeVideos(true));
$('#promise-video-title').addEventListener('input', updatePromiseCheckButton);
$('#promise-check-btn').addEventListener('click', async () => {
  if (!state.promiseVideoId) { toast('Load a YouTube video first.'); return; }
  const title = $('#promise-video-title').value.trim();
  if (title.length < 3) { toast("Enter the video title."); $('#promise-video-title').focus(); return; }
  const button = $('#promise-check-btn');
  button.disabled = true; button.innerHTML = 'Checking transcript <span class="loading-dot">···</span>';
  promiseSetNote('Checking public captions and mapping transcript segments to the title promise. If captions are unavailable, the API will explain how to paste a transcript.');
  try {
    const result = await api('/api/youtube/promise-check', { method: 'POST', body: JSON.stringify({ video_url: state.promiseVideoId, title, transcript: $('#promise-transcript').value.trim() }) });
    state.promiseResult = result;
    renderPromiseResult(result);
    promiseSetNote(`Analysis complete · source: ${result.transcript_source}. ${result.title_promise_analysis.method || ''}`);
    toast('Title Promise audit complete');
  } catch (error) {
    promiseSetNote(error.message || 'Promise audit failed.', true);
    toast(error.message || 'Promise audit failed.');
    const target = $('#promise-result-content');
    target.classList.remove('hidden');
    target.innerHTML = `<div class="promise-error-card"><strong>Transcript analysis could not finish</strong><p>${escapeHtml(error.message || 'Check the video link and transcript.')}</p><p>Paste a timestamped transcript above if YouTube captions are unavailable, then run the check again.</p></div>`;
    $('#promise-result-empty').classList.add('hidden');
  } finally {
    button.innerHTML = 'Check title promise <span>↗</span>';
    updatePromiseCheckButton();
  }
});
function promiseStatusClass(status) {
  if (status === 'Fulfilled') return 'fulfilled';
  if (status === 'Partly Fulfilled') return 'partly';
  return 'not-enough';
}
function timestampToSeconds(value) {
  const match = String(value || '').match(/\[(\d{2}:\d{2}(?::\d{2})?)/);
  if (!match) return null;
  const parts = match[1].split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return parts[0] * 60 + parts[1];
}
function promiseproofStatusClass(status) {
  if (status === 'Supported') return 'supported';
  if (status === 'Partly supported') return 'partly-supported';
  return 'no-evidence';
}
function renderPromiseProof(analysis) {
  if (!analysis) return '';
  const total = Math.max(0, Number(analysis.total_claims) || 0);
  const coverage = Math.max(0, Math.min(100, Number(analysis.evidence_coverage_percent) || 0));
  const supported = Number(analysis.supported_count) || 0;
  const partly = Number(analysis.partly_supported_count) || 0;
  const missing = Number(analysis.no_evidence_count) || 0;
  const cards = (analysis.claims || []).map(claim => {
    const klass = promiseproofStatusClass(claim.status);
    const score = Math.max(0, Math.min(100, Number(claim.evidence_score_percent) || 0));
    const evidence = claim.evidence || null;
    let evidenceHtml = '<p class="promiseproof-no-snippet">No transcript window could be linked as useful evidence. This does not prove the claim is false.</p>';
    if (evidence) {
      const seconds = timestampToSeconds(evidence.timestamp_range);
      const timestamp = seconds === null
        ? `<span class="promise-time-unavailable">${escapeHtml(evidence.timestamp_range || 'Timestamp unavailable')}</span>`
        : `<button class="promise-evidence-time" data-promise-seek="${seconds}">${escapeHtml(evidence.timestamp_range)} ↗</button>`;
      evidenceHtml = `<div class="promiseproof-evidence"><div class="promise-evidence-head">${timestamp}<span>TRANSCRIPT EVIDENCE</span></div><p>${escapeHtml(evidence.summary || '')}</p>${evidence.matched_terms?.length ? `<small>Matched cues: ${escapeHtml(evidence.matched_terms.join(', '))}</small>` : ''}<small>${escapeHtml(evidence.evidence_reason || '')}</small></div>`;
    }
    return `<article class="promiseproof-claim ${klass}">
      <div class="promiseproof-claim-head"><span class="promiseproof-claim-id">${escapeHtml(claim.claim_id || 'CLAIM')}</span><span class="promiseproof-status ${klass}">${escapeHtml(claim.status || 'No evidence found')}</span><span class="promiseproof-score">${score}/100 evidence score</span></div>
      <h4>${escapeHtml(claim.claim_text || '')}</h4>
      <p class="promiseproof-reasoning">${escapeHtml(claim.reasoning || '')}</p>
      ${evidenceHtml}
    </article>`;
  }).join('');
  const limitedClass = analysis.limited_evidence ? ' limited' : '';
  return `<section class="promiseproof-panel" aria-label="PromiseProof claim-wise evidence audit">
    <div class="promiseproof-heading"><div><span class="promise-score-kicker">PROMISEPROOF</span><h3>Claim-wise title vs video evidence</h3><p>Each checkable title promise is matched to the available transcript.</p></div><div class="promiseproof-coverage"><strong>${coverage}%</strong><span>evidence coverage</span></div></div>
    <div class="promise-score-track" role="progressbar" aria-label="Claim-wise evidence coverage" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${coverage}"><span style="width:${coverage}%"></span></div>
    <p class="promiseproof-summary">${Number(analysis.claims_with_some_evidence_count) || 0} of ${total} claims have at least partial transcript evidence. Coverage is not a factual-accuracy score.</p>
    <div class="promiseproof-counts"><span class="supported-count"><b>${supported}</b> Supported</span><span class="partial-count"><b>${partly}</b> Partly supported</span><span class="missing-count"><b>${missing}</b> No evidence found</span></div>
    <div class="promiseproof-claims">${cards || '<p class="promise-muted">No checkable title claims could be extracted. Review the title manually.</p>'}</div>
    <p class="promiseproof-limit${limitedClass}"><strong>${analysis.limited_evidence ? 'Limited evidence warning' : 'Evidence scope'}</strong> ${escapeHtml(analysis.limitation_note || 'Only the available transcript is assessed; factual claims are not independently verified.')}</p>
    <p class="promiseproof-method">${escapeHtml(analysis.method || 'Transparent title-claim rules and transcript matching')} · Generative LLM used: <b>${analysis.llm_used ? 'Yes' : 'No'}</b>.</p>
  </section>`;
}
function renderClickbaitAssessment(assessment) {
  if (!assessment) return '';
  const score = Math.max(0, Math.min(100, Number(assessment.risk_score) || 0));
  const reasons = (assessment.reasons || []).slice(0, 3).map(reason => `<li>${escapeHtml(reason)}</li>`).join('');
  const mode = assessment.model_mode === 'trained_model' ? 'Locally trained model' : 'Demo heuristic';
  return `<section class="promise-clickbait-panel"><div class="promiseproof-heading"><div><span class="promise-score-kicker">SEPARATE SIGNAL</span><h3>Title-only clickbait risk</h3><p>Shown separately from claim support.</p></div><div class="promise-clickbait-score"><strong>${score}/100</strong><span>${escapeHtml(assessment.label || 'Risk estimate')}</span></div></div>
    ${reasons ? `<ul class="promise-filler-list">${reasons}</ul>` : ''}<p class="promise-score-note">${escapeHtml(mode)} · ${escapeHtml(assessment.disclaimer || 'Risk estimate, not a truth verdict.')}</p></section>`;
}

function renderPromiseResult(result) {
  const analysis = result.title_promise_analysis || {};
  const promiseproof = result.promiseproof_analysis || {};
  const status = analysis.fulfillment_status || 'Not Enough Evidence';
  const klass = promiseStatusClass(status);
  const matched = (analysis.matched_terms || []).map(term => `<span>${escapeHtml(term)}</span>`).join('');
  const evidence = (analysis.evidence_timestamps || []).map((item, index) => {
    const seconds = timestampToSeconds(item.timestamp_range);
    const jump = seconds === null ? `<span class="promise-time-unavailable">${escapeHtml(item.timestamp_range || 'Timestamp unavailable')}</span>` : `<button class="promise-evidence-time" data-promise-seek="${seconds}">${escapeHtml(item.timestamp_range)} ↗</button>`;
    return `<article class="promise-evidence-item"><div class="promise-evidence-head">${jump}<span>EVIDENCE ${String(index + 1).padStart(2, '0')}</span></div><p>${escapeHtml(item.summary || '')}</p>${item.matched_terms ? `<small>Matched terms: ${escapeHtml(item.matched_terms)}</small>` : ''}</article>`;
  }).join('') || '<p class="promise-muted">No specific supporting transcript window was identified.</p>';
  const filler = (analysis.filler_detected || []).map(text => `<li>${escapeHtml(text)}</li>`).join('');
  const transcript = escapeHtml(result.transcript || 'Transcript unavailable');
  const alignmentScore = Math.max(0, Math.min(100, Number(analysis.alignment_score_percent) || 0));
  const topicCoverage = Math.max(0, Math.min(100, Number(analysis.topic_term_coverage_percent) || 0));
  const evidenceStrength = Math.max(0, Math.min(100, Number(analysis.evidence_strength_percent) || 0));
  const semanticIndex = analysis.semantic_similarity_index_percent == null ? null : Math.max(0, Math.min(100, Number(analysis.semantic_similarity_index_percent) || 0));
  const analysisEngine = analysis.analysis_engine || analysis.method || 'Transparent transcript rules';
  const matchedCount = Number(analysis.matched_terms?.length || 0);
  const coreCount = Number(analysis.core_terms?.length || 0);
  $('#promise-result-empty').classList.add('hidden');
  const target = $('#promise-result-content'); target.classList.remove('hidden');
  target.innerHTML = `<div class="promise-verdict ${klass}"><div><span class="promise-status-label">FULFILMENT STATUS</span><h2>${escapeHtml(status)}</h2></div><span class="promise-status-mark">${klass === 'fulfilled' ? '✓' : klass === 'partly' ? '◐' : '?'}</span></div>
    <section class="promise-alignment-card" aria-label="Title and transcript alignment score">
      <div class="promise-alignment-head"><div><span class="promise-score-kicker">TITLE–TRANSCRIPT MATCH INDEX</span><p>Composite alignment estimate</p></div><strong>${alignmentScore}%</strong></div>
      <div class="promise-score-track" role="progressbar" aria-label="Title transcript alignment index" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${alignmentScore}"><span style="width:${alignmentScore}%"></span></div>
      <div class="promise-score-breakdown"><span>Topic-term coverage <b>${topicCoverage}% (${matchedCount}/${coreCount})</b></span><span>Best evidence strength <b>${evidenceStrength}%</b></span><span>Semantic similarity index <b>${semanticIndex === null ? 'Not enabled' : `${semanticIndex}/100`}</b></span></div>
      <p class="promise-score-note">This is a transparent 0–100 alignment index, <b>not a probability</b> that the title is true or fulfilled. ${escapeHtml(analysisEngine)}. Generative LLM used: <b>${analysis.llm_used ? 'Yes' : 'No'}</b>.</p>
    </section>
    ${renderPromiseProof(promiseproof)}
    ${renderClickbaitAssessment(result.clickbait_assessment)}
    <div class="promise-field-label">EXTRACTED PROMISE</div><p class="promise-extracted">${escapeHtml(analysis.extracted_promise || '')}</p>
    <div class="promise-field-label">OBJECTIVE REASONING</div><p class="promise-reasoning">${escapeHtml(analysis.reasoning || '')}</p>
    <div class="promise-stat-grid"><div><small>TRANSCRIPT TERMS</small><strong>${matchedCount} / ${coreCount}</strong></div><div><small>TRANSCRIPT WORDS</small><strong>${Number(result.transcript_word_count || 0).toLocaleString()}</strong></div><div><small>CAPTION SOURCE</small><strong>${escapeHtml(result.transcript_source || 'Transcript')}</strong></div></div>
    ${matched ? `<div class="promise-term-list">${matched}</div>` : ''}
    <div class="promise-field-label">TIMESTAMPED EVIDENCE <span>Click a timestamp to jump the player</span></div><div class="promise-evidence-list">${evidence}</div>
    ${filler ? `<div class="promise-field-label">POSSIBLE FILLER / PROMOTIONS</div><ul class="promise-filler-list">${filler}</ul>` : ''}
    <details class="media-transcript promise-full-transcript"><summary>Read retrieved transcript</summary><pre>${transcript}</pre></details>
    <p class="media-disclaimer">${escapeHtml(result.disclaimer || '')}<br>Method: ${escapeHtml(analysis.method || 'Transcript evidence rules')}.</p>
    <div class="media-result-actions"><button class="outline-button" id="promise-download-report">↓ Download JSON report</button><button class="outline-button" id="promise-copy-report">⎘ Copy report</button></div>`;
  $$('.promise-evidence-time', target).forEach(button => button.addEventListener('click', () => {
    if (!state.promiseVideoId) return;
    const seconds = Math.max(0, Number(button.dataset.promiseSeek) || 0);
    const iframe = $('#promise-youtube-frame');
    if (iframe) iframe.src = `https://www.youtube-nocookie.com/embed/${state.promiseVideoId}?start=${seconds}&autoplay=1&rel=0`;
  }));
  const report = { title_promise_analysis: analysis, promiseproof_analysis: promiseproof, clickbait_assessment: result.clickbait_assessment || null, video_title: result.title, video_url: result.watch_url, transcript_source: result.transcript_source, disclaimer: result.disclaimer };
  const reportText = JSON.stringify(report, null, 2);
  $('#promise-download-report').addEventListener('click', () => {
    const blob = new Blob([reportText], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'vericlick_title_promise_report.json'; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
  });
  $('#promise-copy-report').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(reportText); toast('Title Promise report copied'); }
    catch (_) { toast('Clipboard blocked; use Download JSON report instead.'); }
  });
}
function renderTitlePromiseInline(analysis) {
  if (!analysis) return '';
  const klass = promiseStatusClass(analysis.fulfillment_status || 'Not Enough Evidence');
  const evidence = (analysis.evidence_timestamps || []).slice(0, 2).map(item => `<li><strong>${escapeHtml(item.timestamp_range)}</strong> ${escapeHtml(item.summary)}</li>`).join('');
  return `<div class="promise-inline"><div class="promise-inline-head"><span>TITLE PROMISE CHECK</span><strong class="promise-inline-status ${klass}">${escapeHtml(analysis.fulfillment_status || 'Not Enough Evidence')}</strong></div><p>${escapeHtml(analysis.extracted_promise || '')}</p><small>${escapeHtml(analysis.reasoning || '')}</small>${evidence ? `<ul>${evidence}</ul>` : ''}</div>`;
}

// Startup
(async function init() {
  dateLabel();
  await Promise.all([refreshHealth(), refreshDashboard(), refreshHistory(), loadDemoCases(), refreshModelDetails(), refreshMediaCapabilities(), refreshYoutubeSearchStatus()]);
})();
