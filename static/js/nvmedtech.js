// NVIDIA MedTech (NV-Segment-CT / VISTA-3D) tab — mirrors the MONAI Bundle tab.
// Uses shared escapeHtml()/escapeJsString() from main.js. Research use only.

let _nvArtifacts = { base: [], finetuned: [] };

// ── Bundle card (papers + data org from the bundle's metadata.json) ─────────
async function nvLoadBundleCard() {
    const el = document.getElementById('nv-bundle-card');
    if (!el) return;
    try {
        const b = await (await fetch('/api/nvmedtech/bundle')).json();
        const papers = (b.papers || []).map(function (p) {
            const t = escapeHtml(p.title || p.url || '');
            return p.url
                ? '<li style="margin-bottom:0.2rem;"><a href="' + escapeHtml(p.url) + '" target="_blank" rel="noopener noreferrer">' + t + '</a></li>'
                : '<li style="margin-bottom:0.2rem;">' + t + '</li>';
        }).join('');
        const d = b.data_org || {};
        const tree = (d.layout || []).map(escapeHtml).join('\n');
        const rowStyle = 'font-size:0.75rem; color:#555; overflow-wrap:anywhere; word-break:break-word;';
        el.innerHTML =
            '<div class="model-item model-item-segmentation" style="min-width:0; overflow:hidden; display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">' +
                '<div style="flex:1 1 220px; min-width:0;">' +
                    '<div style="display:flex; align-items:center; gap:6px; margin-bottom:0.35rem;">' +
                        '<span class="model-name" style="overflow-wrap:anywhere;">' + escapeHtml(b.name) + '</span>' +
                        '<span class="model-tag model-tag-segmentation">Segmentation</span>' +
                    '</div>' +
                    '<div style="font-size:0.78rem; color:#555; overflow-wrap:anywhere; margin-bottom:0.4rem;">' + escapeHtml(b.description || '') + '</div>' +
                    (b.monai_version ? '<div style="font-size:0.72rem; color:#888;">MONAI ' + escapeHtml(b.monai_version) + ' · PyTorch ' + escapeHtml(b.pytorch_version || '?') + ' (dedicated images)</div>' : '') +
                    '<div style="font-size:0.72rem; margin-top:0.4rem;"><i class="fab fa-github"></i> <a href="' + escapeHtml(b.source_url) + '" target="_blank" rel="noopener noreferrer">NV-Segment-CT on GitHub</a></div>' +
                '</div>' +
                '<div style="flex:2 1 420px; min-width:0; display:flex; flex-wrap:wrap; gap:1rem; padding-left:1rem; border-left:1px solid #eee;">' +
                    (papers ? '<div style="flex:1 1 260px; min-width:0;"><div style="font-weight:600; font-size:0.8rem; margin-bottom:0.25rem;"><i class="fas fa-book"></i> Key papers</div><ul style="font-size:0.75rem; margin:0; padding-left:1.1rem; overflow-wrap:anywhere; word-break:break-word;">' + papers + '</ul></div>' : '') +
                    ((d.modality || tree)
                        ? '<div style="flex:1 1 260px; min-width:0;">' +
                            '<div style="font-weight:600; font-size:0.8rem; margin-bottom:0.25rem;"><i class="fas fa-folder-tree"></i> Training data layout' + (d.layout_source ? ' <span style="font-weight:normal; color:#888;">(from ' + escapeHtml(d.layout_source) + ')</span>' : '') + '</div>' +
                            (d.modality ? '<div style="' + rowStyle + '"><strong>Modality:</strong> ' + escapeHtml(d.modality) + '</div>' : '') +
                            (d.labels ? '<div style="' + rowStyle + '"><strong>Classes:</strong> ' + escapeHtml(d.labels) + '</div>' : '') +
                            (d.data_source ? '<div style="' + rowStyle + '"><strong>Source:</strong> ' + escapeHtml(d.data_source) + '</div>' : '') +
                            (tree ? '<pre style="background:#f6f8fa; border:1px solid #eee; border-radius:6px; padding:0.5rem; font-size:0.72rem; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; margin:0.3rem 0 0;">' + tree + '</pre>' : '') +
                            (d.training_data_note ? '<div style="' + rowStyle + ' margin-top:0.3rem; padding:0.35rem 0.5rem; background:#fff8e1; border:1px solid #ffe082; border-radius:6px;"><i class="fas fa-circle-info"></i> ' + escapeHtml(d.training_data_note) + '</div>' : '') +
                          '</div>'
                        : '') +
                '</div>' +
            '</div>';
    } catch (e) {
        el.innerHTML = '<p style="color:#c62828;">Could not load bundle info: ' + escapeHtml(e.message) + '</p>';
    }
}

// ── Stage ────────────────────────────────────────────────────────────────────
async function nvStage() {
    const s3Uri = document.getElementById('nv-stage-s3').value.trim();
    const out = document.getElementById('nv-stage-result');
    if (!s3Uri) { alert('destination S3 URI is required'); return; }
    out.style.display = 'block';
    out.innerHTML = 'Staging… (packages configs + 872 MB checkpoint + handler; can take several minutes)';
    try {
        const resp = await fetch('/api/nvmedtech/stage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ s3_uri: s3Uri }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'stage failed');
        out.innerHTML =
            '<div class="result-row"><span>Artifact</span> ' + escapeHtml(data.model_artifacts) + '</div>' +
            '<div class="result-row"><span>Includes</span> configs + models/model.pt + code/inference.py (deps baked in image)</div>';
        document.getElementById('nv-model-artifacts').value = data.model_artifacts;
        nvLoadArtifacts();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

// ── Fine-tune ────────────────────────────────────────────────────────────────
async function nvFinetune() {
    const out = document.getElementById('nv-finetune-result');
    let extra = {};
    const raw = document.getElementById('nv-overrides').value.trim();
    if (raw) {
        try { extra = JSON.parse(raw); }
        catch (e) { alert('Extra overrides must be valid JSON: ' + e.message); return; }
    }
    const num = function (id) {
        const v = document.getElementById(id).value.trim();
        return v === '' ? null : Number(v);
    };
    const body = {
        model_artifacts: document.getElementById('nv-model-artifacts').value.trim(),
        data_s3_uri: document.getElementById('nv-data-s3').value.trim(),
        instance_type: document.getElementById('nv-instance-type').value.trim() || 'ml.g5.4xlarge',
        extra_overrides: extra,
    };
    const epochs = num('nv-epochs'); if (epochs !== null) body.epochs = epochs;
    const lr = num('nv-lr');         if (lr !== null) body.learning_rate = lr;
    if (!body.model_artifacts || !body.data_s3_uri) {
        alert('bundle artifact and dataset S3 URI are required'); return;
    }
    out.style.display = 'block';
    out.innerHTML = 'Launching…';
    try {
        const resp = await fetch('/api/nvmedtech/finetune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'fine-tune failed');
        out.innerHTML =
            '<div class="result-row"><span>Job</span> ' + escapeHtml(data.job_name) + '</div>' +
            '<div class="result-row"><span>Status</span> ' + escapeHtml(data.status) + '</div>' +
            '<div class="result-row"><span>Config</span> ' + escapeHtml(data.config.train_config) + '</div>';
        nvRefreshJobs();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

// ── Jobs (Active / Completed / Failed / Archived tabs) ──────────────────────
let _nvAllJobs = [];

function nvSwitchJobsTab(tab) {
    window._nvJobsTab = tab;
    ['active', 'completed', 'failed', 'archived'].forEach(function (t) {
        const el = document.getElementById('nv-tab-' + t);
        if (el) el.classList.toggle('active', t === tab);
    });
    nvDisplayJobs(_nvAllJobs);
}

function nvDisplayJobs(jobs) {
    const container = document.getElementById('nv-jobs-list');
    if (!container) return;
    const tab = window._nvJobsTab || 'completed';
    const ACTIVE = new Set(['InProgress', 'Stopping']);
    const COMPLETED = new Set(['Completed', 'Stopped']);
    const visible = jobs.filter(function (j) {
        if (tab === 'archived')  return j.archived;
        if (tab === 'failed')    return !j.archived && j.status === 'Failed';
        if (tab === 'completed') return !j.archived && COMPLETED.has(j.status);
        return !j.archived && ACTIVE.has(j.status);
    });
    if (!visible.length) {
        const msgs = { active: 'No active jobs.', completed: 'No completed jobs.', failed: 'No failed jobs.', archived: 'No archived jobs.' };
        container.innerHTML = '<p style="color:#888; padding:0.5rem;">' + (msgs[tab] || 'No jobs.') + '</p>';
        return;
    }
    container.innerHTML = '';
    visible.forEach(function (job) {
        const statusClass = {
            'InProgress': 'status-running', 'Completed': 'status-completed',
            'Failed': 'status-failed', 'Stopping': 'status-stopping', 'Stopped': 'status-stopped',
        }[job.status] || 'status-unknown';
        const cfg = job.config || {};
        const el = document.createElement('div');
        el.className = 'job-item';
        el.innerHTML =
            '<div class="job-header">' +
                '<div class="job-id">' + escapeHtml(job.job_name) + '</div>' +
                '<div class="job-status ' + statusClass + '">' + escapeHtml(job.status || 'unknown') + '</div>' +
            '</div>' +
            '<div class="job-details">' +
                (cfg.data_s3_uri ? '<div style="font-size:0.8rem;"><strong>Data:</strong> <code style="font-size:0.75rem;">' + escapeHtml(cfg.data_s3_uri) + '</code></div>' : '') +
                (cfg.instance_type ? '<div style="font-size:0.8rem;"><strong>Instance:</strong> ' + escapeHtml(cfg.instance_type) + '</div>' : '') +
                (job.model_artifacts ? '<div style="font-size:0.8rem;"><strong>Artifacts:</strong> <code style="font-size:0.75rem;">' + escapeHtml(job.model_artifacts) + '</code></div>' : '') +
                (job.failure_reason ? '<div style="font-size:0.8rem; color:#c62828;"><strong>Reason:</strong> ' + escapeHtml(job.failure_reason) + '</div>' : '') +
            '</div>';
        container.appendChild(el);
    });
}

async function nvRefreshJobs() {
    const list = document.getElementById('nv-jobs-list');
    if (!list) return;
    list.innerHTML = 'Loading…';
    try {
        const data = await (await fetch('/api/nvmedtech/jobs')).json();
        _nvAllJobs = data.jobs || [];
        nvDisplayJobs(_nvAllJobs);
    } catch (e) {
        list.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

// ── Deploy (base vs fine-tuned; async only; live status) ────────────────────
async function nvLoadArtifacts() {
    try {
        _nvArtifacts = await (await fetch('/api/nvmedtech/artifacts')).json();
    } catch (e) {
        _nvArtifacts = { base: [], finetuned: [] };
    }
    nvOnDeploySourceChange();
}

function _nvSelectedSource() {
    const el = document.querySelector('input[name="nv_deploy_source"]:checked');
    return el ? el.value : 'base';
}

function nvOnDeploySourceChange() {
    const source = _nvSelectedSource();
    const label = document.getElementById('nv-deploy-artifact-label');
    const sel = document.getElementById('nv-deploy-artifact-select');
    const ftList = document.getElementById('nv-deploy-finetuned-list');
    if (source === 'finetuned') {
        label.textContent = 'Fine-tuned bundle (completed job)';
        sel.style.display = 'none';
        ftList.style.display = 'block';
        const items = _nvArtifacts.finetuned || [];
        if (!items.length) {
            ftList.innerHTML = '<p style="color:#888; padding:0.6rem;">No completed fine-tune jobs with model artifacts found.</p>';
            return;
        }
        ftList.innerHTML = items.map(function (it) {
            const art = it.model_artifacts || '';
            return '<div style="padding:0.6rem 0.5rem; border-bottom:1px solid #f0f0f0; cursor:pointer;"' +
                ' onmouseover="this.style.background=\'#f5f5f5\'" onmouseout="this.style.background=\'\'"' +
                ' onclick="nvSelectFinetuned(this, \'' + escapeJsString(art) + '\', \'' + escapeJsString(it.name) + '\')">' +
                '<div style="font-weight:600; font-size:0.9rem;">' + escapeHtml(it.name) + '</div>' +
                '<div style="font-size:0.78rem; color:#666; margin-top:2px; word-break:break-all;">' + escapeHtml(art) + '</div>' +
                (it.instance_type ? '<div style="font-size:0.75rem; color:#888; margin-top:2px;">' + escapeHtml(it.instance_type) + '</div>' : '') +
                '</div>';
        }).join('');
        return;
    }
    label.textContent = 'Base bundle (staged)';
    ftList.style.display = 'none';
    sel.style.display = '';
    const items = (_nvArtifacts.base || []).filter(function (b) { return b.model_artifacts; });
    if (!items.length) {
        sel.innerHTML = '<option value="" disabled selected>Not staged yet — stage the bundle first</option>';
        return;
    }
    sel.innerHTML = '<option value="" disabled selected>Select…</option>' + items.map(function (it) {
        return '<option value="' + escapeHtml(it.model_artifacts) + '">' + escapeHtml(it.name) + '</option>';
    }).join('');
}

function nvSelectFinetuned(row, artifact, jobName) {
    if (!artifact) return;
    document.getElementById('nv-deploy-artifacts').value = artifact;
    const nameField = document.getElementById('nv-endpoint-name');
    if (!nameField.value) {
        nameField.value = (jobName || 'nvmedtech').toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 60).replace(/-+$/, '') + '-ep';
    }
    const list = document.getElementById('nv-deploy-finetuned-list');
    Array.prototype.forEach.call(list.children, function (c) { c.style.background = ''; });
    if (row) row.style.background = '#e8f0fe';
}

function nvOnArtifactSelect() {
    const sel = document.getElementById('nv-deploy-artifact-select');
    if (sel.value) {
        document.getElementById('nv-deploy-artifacts').value = sel.value;
        const nameField = document.getElementById('nv-endpoint-name');
        if (!nameField.value) nameField.value = 'nvmedtech-ct-async';
    }
}

function nvRenderDeployStatus(out, epName, status) {
    const cls = {
        'InService': 'status-completed', 'Failed': 'status-failed',
        'Creating': 'status-running', 'Updating': 'status-running', 'Deleting': 'status-stopping',
    }[status] || 'status-unknown';
    out.innerHTML =
        '<div class="result-row"><span>Endpoint</span> ' + escapeHtml(epName) + '</div>' +
        '<div class="result-row"><span>Type</span> async</div>' +
        '<div class="result-row"><span>Status</span> <span class="job-status ' + cls + '">' + escapeHtml(status) + '</span></div>';
}

async function nvPollDeployStatus(out, epName) {
    const TERMINAL = new Set(['InService', 'Failed', 'Deleted']);
    let tries = 0;
    const timer = setInterval(async function () {
        tries += 1;
        try {
            const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(epName) + '/status');
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || 'status check failed');
            const status = d.status || 'Creating';
            nvRenderDeployStatus(out, epName, status);
            if (status === 'Failed' && d.failure_reason) {
                out.insertAdjacentHTML('beforeend', '<div class="result-row" style="color:#c62828;"><span>Reason</span> ' + escapeHtml(d.failure_reason) + '</div>');
            }
            if (TERMINAL.has(status)) { clearInterval(timer); nvLoadDeployments(); }
        } catch (e) { /* transient */ }
        if (tries >= 120) clearInterval(timer);
    }, 10000);
}

async function nvDeploy() {
    const out = document.getElementById('nv-deploy-result');
    const body = {
        endpoint_name: document.getElementById('nv-endpoint-name').value.trim(),
        model_artifacts: document.getElementById('nv-deploy-artifacts').value.trim(),
        output_s3_uri: document.getElementById('nv-deploy-output').value.trim(),
        instance_type: document.getElementById('nv-deploy-instance').value.trim() || 'ml.g5.xlarge',
        instance_count: Number(document.getElementById('nv-deploy-count').value.trim() || 1),
        max_concurrent: Number(document.getElementById('nv-deploy-concurrent').value.trim() || 4),
    };
    if (!body.endpoint_name || !body.model_artifacts || !body.output_s3_uri) {
        alert('endpoint name, model artifact, and async output S3 location are required'); return;
    }
    out.style.display = 'block';
    out.innerHTML = 'Deploying async endpoint…';
    try {
        const resp = await fetch('/api/nvmedtech/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'deploy failed');
        const epName = data.endpoint_name || body.endpoint_name;
        nvRenderDeployStatus(out, epName, data.status || 'Creating');
        nvPollDeployStatus(out, epName);
        nvLoadDeployments();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

// ── Deployed models ──────────────────────────────────────────────────────────
function _nvIsBundleDeployment(d) {
    const img = (d.image_uri || '').toLowerCase();
    const name = (d.endpoint_name || '').toLowerCase();
    return img.indexOf('nvmedtech') !== -1 || name.indexOf('nvmedtech') !== -1;
}

async function nvLoadDeployments() {
    const container = document.getElementById('nv-deployments-list');
    if (!container) return;
    try {
        const res = await (await fetch('/api/deployment/deployments')).json();
        const items = (res.deployments || []).filter(_nvIsBundleDeployment);
        nvDisplayDeployments(items);
    } catch (e) {
        container.innerHTML = '<p style="color:#c62828; padding:0.5rem;">Error loading deployments: ' + escapeHtml(e.message) + '</p>';
    }
}

function nvDisplayDeployments(deployments) {
    const container = document.getElementById('nv-deployments-list');
    if (!container) return;
    if (!deployments.length) {
        container.innerHTML = '<p style="color:#888; padding:0.5rem;">No NV-Segment-CT endpoints deployed yet.</p>';
        return;
    }
    const statusClass = {
        'Creating': 'status-running', 'Updating': 'status-running',
        'InService': 'status-completed', 'Failed': 'status-failed',
        'Deleting': 'status-stopping', 'Deleted': 'status-stopped',
    };
    container.innerHTML = deployments.map(function (d) {
        const cnt = d.instance_count != null ? d.instance_count : 1;
        const inService = d.status === 'InService';
        const actions = [];
        if (inService && cnt > 0) actions.push('<button class="btn btn-secondary" onclick="nvScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', 0)" title="Scale to 0 (stop billing)"><i class="fas fa-power-off"></i> Scale to 0</button>');
        if (inService && cnt === 0) actions.push('<button class="btn btn-secondary" onclick="nvScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', 1)" style="color:#1e8449;"><i class="fas fa-play"></i> Scale up</button>');
        if (inService && cnt > 1) actions.push('<button class="btn btn-secondary" onclick="nvScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', ' + (cnt - 1) + ')"><i class="fas fa-minus"></i></button>');
        if (inService && cnt > 0) actions.push('<button class="btn btn-secondary" onclick="nvScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', ' + (cnt + 1) + ')"><i class="fas fa-plus"></i></button>');
        if (!['InService', 'Failed', 'Deleting', 'Deleted'].includes(d.status)) actions.push('<button class="btn btn-secondary" onclick="nvRefreshDeployment(\'' + escapeJsString(d.endpoint_name) + '\')"><i class="fas fa-sync"></i> Refresh</button>');
        if (!['Deleting', 'Deleted'].includes(d.status)) actions.push('<button class="btn btn-secondary" onclick="nvDeleteDeployment(\'' + escapeJsString(d.endpoint_name) + '\')" style="color:#c62828;"><i class="fas fa-trash"></i> Delete</button>');
        return '<div class="job-item" style="padding:1rem 1.25rem;">' +
            '<div class="job-header" style="margin-bottom:0.75rem;">' +
                '<div>' +
                    '<div class="job-id" style="font-size:1rem;">' + escapeHtml(d.endpoint_name) + '</div>' +
                    '<span style="font-size:0.75rem; color:#666;">⚡ Async · ' + escapeHtml(d.instance_type || 'N/A') + ' × ' + cnt +
                        (cnt === 0 ? ' <span style="color:#e67e22; font-weight:600;">— scaled to 0</span>' : '') + '</span>' +
                '</div>' +
                '<div class="job-status ' + (statusClass[d.status] || 'status-unknown') + '">' + escapeHtml(d.status) + '</div>' +
            '</div>' +
            '<div style="display:grid; grid-template-columns:auto 1fr; gap:0.3rem 1rem; font-size:0.82rem; margin-bottom:0.75rem;">' +
                '<strong>Created:</strong> <span>' + (d.created_at ? escapeHtml(new Date(d.created_at).toLocaleString()) : 'N/A') + '</span>' +
                (d.output_s3_uri ? '<strong>Output:</strong> <span style="word-break:break-all; font-family:monospace; font-size:0.75rem;">' + escapeHtml(d.output_s3_uri) + '</span>' : '') +
                (d.failure_reason ? '<strong style="color:#c62828;">Error:</strong> <span style="color:#c62828;">' + escapeHtml(d.failure_reason) + '</span>' : '') +
            '</div>' +
            '<div class="job-actions">' + actions.join('') + '</div>' +
        '</div>';
    }).join('');
}

async function nvScaleEndpoint(name, count) {
    if (count === 0 && !confirm('Scale ' + name + ' to 0 instances? This stops billing but keeps the endpoint.')) return;
    try {
        const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(name) + '/scale', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ desired_count: count }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'scale failed');
        nvLoadDeployments();
    } catch (e) { alert('Scale failed: ' + e.message); }
}

async function nvDeleteDeployment(name) {
    if (!confirm('Delete endpoint ' + name + '? This removes the endpoint, config, and model.')) return;
    try {
        const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(name), { method: 'DELETE' });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'delete failed');
        nvLoadDeployments();
    } catch (e) { alert('Delete failed: ' + e.message); }
}

async function nvRefreshDeployment(name) {
    try {
        await fetch('/api/deployment/deployments/' + encodeURIComponent(name) + '/status');
        nvLoadDeployments();
    } catch (e) { /* transient */ }
}

if (typeof registerTabInit === 'function') {
    registerTabInit('nvmedtech', function () {
        nvLoadBundleCard(); nvRefreshJobs(); nvLoadArtifacts(); nvLoadDeployments();
    });
}
