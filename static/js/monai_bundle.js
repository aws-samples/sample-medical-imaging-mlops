// MONAI Bundle tab — stage, fine-tune, and track jobs.
// Uses shared escapeHtml() from main.js. Research use only.

let _mbStageConfirmed = false;

function mbConfirmStage() {
    const cb = document.getElementById('mb-stage-confirm');
    _mbStageConfirmed = !!(cb && cb.checked);
    const box = document.getElementById('mb-stage-confirm-box');
    if (box) {
        box.style.background = _mbStageConfirmed ? '#e8f5e9' : '#fff8e1';
        box.style.borderColor = _mbStageConfirmed ? '#a5d6a7' : '#ffe082';
    }
}

function mbRenderDataLayout(dl) {
    if (!dl) return '';
    const tree = (dl.layout || []).map(escapeHtml).join('\n');
    const notes = (dl.notes || []).map(function (n) {
        return '<li>' + escapeHtml(n) + '</li>';
    }).join('');
    let io = '';
    if (dl.expected_inputs || dl.expected_outputs) {
        io = '<div style="font-size:0.75rem; color:#666; margin-top:0.4rem; overflow-wrap:anywhere; word-break:break-word;">Bundle I/O: ' +
             escapeHtml(JSON.stringify({ inputs: dl.expected_inputs, outputs: dl.expected_outputs })) + '</div>';
    }
    return '' +
        '<div style="margin-top:0.75rem; padding-top:0.6rem; border-top:1px solid #eee; min-width:0;">' +
            '<div style="font-weight:600; font-size:0.85rem; margin-bottom:0.3rem;">' +
                '<i class="fas fa-folder-tree"></i> Expected fine-tuning data layout' +
                (dl.format ? ' <span style="font-weight:normal; color:#666;">(' + escapeHtml(dl.format) + ')</span>' : '') +
            '</div>' +
            (dl.description ? '<div style="font-size:0.8rem; color:#555; margin-bottom:0.4rem; overflow-wrap:anywhere;">' + escapeHtml(dl.description) + '</div>' : '') +
            '<pre style="background:#f6f8fa; border:1px solid #eee; border-radius:6px; padding:0.6rem; font-size:0.75rem; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; margin:0;">' + tree + '</pre>' +
            (notes ? '<ul style="font-size:0.75rem; color:#666; margin:0.4rem 0 0; padding-left:1.1rem; overflow-wrap:anywhere;">' + notes + '</ul>' : '') +
            io +
        '</div>';
}

async function mbStageBundle() {
    const repoId = document.getElementById('mb-repo-id').value.trim();
    const s3Uri = document.getElementById('mb-stage-s3').value.trim();
    const out = document.getElementById('mb-stage-result');
    if (!repoId || !s3Uri) { alert('repo id and destination S3 URI are required'); return; }
    _mbStageConfirmed = false;
    out.style.display = 'block';
    out.innerHTML = 'Staging…';
    try {
        const resp = await fetch('/api/monai-bundle/stage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, s3_uri: s3Uri }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'stage failed');
        const papersList = (data.papers || []).map(function (p) {
            const title = escapeHtml(p.title || p.url || '');
            return p.url
                ? '<li><a href="' + escapeHtml(p.url) + '" target="_blank" rel="noopener noreferrer">' + title + '</a></li>'
                : '<li>' + title + '</li>';
        }).join('');
        const dorg = data.data_org || {};

        out.innerHTML =
            '<div class="result-row"><span>Task</span> ' + escapeHtml(data.task) + '</div>' +
            '<div class="result-row"><span>Artifact</span> ' + escapeHtml(data.model_artifacts) + '</div>' +
            '<div class="result-row"><span>Train cfg</span> ' + escapeHtml(data.default_train_config) + '</div>' +
            '<div class="result-row"><span>Infer cfg</span> ' + escapeHtml(data.default_inference_config) + '</div>' +
            (data.deps && data.deps.extra_packages_installed_at_startup && data.deps.extra_packages_installed_at_startup.length
                ? '<div class="result-row"><span>Extra deps</span> installed at endpoint start: ' +
                  escapeHtml(data.deps.extra_packages_installed_at_startup.join(', ')) + '</div>'
                : '') +
            (data.deps && data.deps.bundle_framework_versions
                ? '<div class="result-row"><span>Bundle expects</span> ' +
                  escapeHtml(Object.entries(data.deps.bundle_framework_versions).map(function(kv){return kv[0].replace('_version','')+' '+kv[1];}).join(', ')) +
                  ' <em>(verify vs. image if inference fails)</em></div>'
                : '') +
            mbRenderDataLayout(data.data_layout) +
            // Metadata-sourced papers + data organization, with the GitHub source
            // and an explicit confirmation prompt.
            '<div style="margin-top:0.75rem; padding-top:0.6rem; border-top:1px solid #eee;">' +
                '<div style="font-weight:600; font-size:0.85rem;"><i class="fas fa-circle-info"></i> Sourced from the bundle\'s metadata.json</div>' +
                (data.source_url ? '<div style="font-size:0.78rem; margin-top:0.2rem;"><i class="fab fa-github"></i> <a href="' + escapeHtml(data.source_url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(data.source_url) + '</a></div>' : '') +
                (papersList ? '<div style="font-size:0.8rem; font-weight:600; margin-top:0.4rem;">Key papers</div><ul style="font-size:0.76rem; margin:0.2rem 0 0; padding-left:1.1rem; overflow-wrap:anywhere;">' + papersList + '</ul>' : '') +
                (dorg.modality || dorg.labels || dorg.data_source
                    ? '<div style="font-size:0.8rem; font-weight:600; margin-top:0.4rem;">Data organization</div>' +
                      (dorg.modality ? '<div style="font-size:0.76rem; color:#555; overflow-wrap:anywhere;"><strong>Modality:</strong> ' + escapeHtml(dorg.modality) + '</div>' : '') +
                      (dorg.labels ? '<div style="font-size:0.76rem; color:#555; overflow-wrap:anywhere;"><strong>Classes:</strong> ' + escapeHtml(dorg.labels) + '</div>' : '') +
                      (dorg.data_source ? '<div style="font-size:0.76rem; color:#555; overflow-wrap:anywhere;"><strong>Source:</strong> ' + escapeHtml(dorg.data_source) + '</div>' : '')
                    : '') +
                '<div id="mb-stage-confirm-box" style="margin-top:0.6rem; padding:0.5rem 0.6rem; background:#fff8e1; border:1px solid #ffe082; border-radius:6px; font-size:0.78rem;">' +
                    '<label style="display:flex; align-items:flex-start; gap:0.4rem; cursor:pointer;">' +
                        '<input type="checkbox" id="mb-stage-confirm" onchange="mbConfirmStage()" style="margin-top:2px;">' +
                        '<span>I have reviewed the papers and data organization above (sourced from the linked metadata.json) and confirm they are correct before fine-tuning.</span>' +
                    '</label>' +
                '</div>' +
            '</div>';
        // Prefill the fine-tune artifact field for convenience (trainable bundles only).
        if (data.trainable !== false) {
            document.getElementById('mb-model-artifacts').value = data.model_artifacts;
        }
        if (typeof mbLoadStaged === 'function') mbLoadStaged();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

async function mbFinetune() {
    const out = document.getElementById('mb-finetune-result');
    let extra = {};
    const rawOverrides = document.getElementById('mb-overrides').value.trim();
    if (rawOverrides) {
        try { extra = JSON.parse(rawOverrides); }
        catch (e) { alert('Extra config overrides must be valid JSON'); return; }
    }
    const body = {
        model_artifacts: document.getElementById('mb-model-artifacts').value.trim(),
        data_s3_uri: document.getElementById('mb-data-s3').value.trim(),
        instance_type: document.getElementById('mb-instance').value.trim() || 'ml.g5.4xlarge',
        extra_overrides: extra,
    };
    const trainCfg = document.getElementById('mb-train-config').value.trim();
    if (trainCfg) body.train_config = trainCfg;

    // Discrete hyperparameters — only include when the user set a value.
    const num = (id) => {
        const v = document.getElementById(id).value.trim();
        return v === '' ? null : Number(v);
    };
    const epochs = num('mb-epochs');       if (epochs !== null) body.epochs = epochs;
    const cache  = num('mb-cache-rate');   if (cache  !== null) body.cache_rate = cache;
    const lr     = num('mb-lr');           if (lr     !== null) body.learning_rate = lr;
    const batch  = num('mb-batch');        if (batch  !== null) body.batch_size = batch;

    if (!body.model_artifacts || !body.data_s3_uri) { alert('bundle artifact and dataset S3 URI are required'); return; }

    // If the just-staged confirmation prompt is showing, require acknowledgement.
    const confirmBox = document.getElementById('mb-stage-confirm');
    if (confirmBox && !confirmBox.checked) {
        alert('Please review and confirm the bundle\'s papers and data organization (sourced from metadata.json) before fine-tuning — see the checkbox in the stage result.');
        confirmBox.focus();
        return;
    }

    out.style.display = 'block';
    out.innerHTML = 'Launching…';
    try {
        const resp = await fetch('/api/monai-bundle/finetune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'fine-tune failed');
        out.innerHTML =
            '<div class="result-row"><span>Job</span> ' + escapeHtml(data.job_name) + '</div>' +
            '<div class="result-row"><span>Status</span> ' + escapeHtml(data.status) + '</div>';
        mbRefreshJobs();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

let _mbAllJobs = [];

function mbSwitchJobsTab(tab) {
    window._mbJobsTab = tab;
    ['active', 'completed', 'failed', 'archived'].forEach(function (t) {
        const el = document.getElementById('mb-tab-' + t);
        if (el) el.classList.toggle('active', t === tab);
    });
    mbDisplayJobs(_mbAllJobs);
}

function mbDisplayJobs(jobs) {
    const container = document.getElementById('mb-jobs-list');
    if (!container) return;
    const tab = window._mbJobsTab || 'completed';
    const ACTIVE = new Set(['InProgress', 'Stopping']);
    const COMPLETED = new Set(['Completed', 'Stopped']);

    const visible = jobs.filter(function (j) {
        if (tab === 'archived')  return j.archived;
        if (tab === 'failed')    return !j.archived && j.status === 'Failed';
        if (tab === 'completed') return !j.archived && COMPLETED.has(j.status);
        return !j.archived && ACTIVE.has(j.status); // active
    });

    if (!visible.length) {
        const msgs = { active: 'No active jobs.', completed: 'No completed jobs.', failed: 'No failed jobs.', archived: 'No archived jobs.' };
        container.innerHTML = '<p style="color:#888; padding:0.5rem;">' + (msgs[tab] || 'No jobs.') + '</p>';
        return;
    }

    container.innerHTML = '';
    visible.forEach(function (job) {
        const statusClass = {
            'InProgress': 'status-running',
            'Completed':  'status-completed',
            'Failed':     'status-failed',
            'Stopping':   'status-stopping',
            'Stopped':    'status-stopped',
        }[job.status] || 'status-unknown';

        const cfg = job.config || {};
        const artifactHtml = job.model_artifacts
            ? '<div style="font-size:0.8rem;"><strong>Artifacts:</strong> <code style="font-size:0.75rem;">' + escapeHtml(job.model_artifacts) + '</code></div>'
            : '';
        const failHtml = job.failure_reason
            ? '<div style="font-size:0.8rem; color:#c62828;"><strong>Reason:</strong> ' + escapeHtml(job.failure_reason) + '</div>'
            : '';

        const el = document.createElement('div');
        el.className = 'job-item' + (job.archived ? ' job-archived' : '');
        el.innerHTML =
            '<div class="job-header">' +
                '<div class="job-id">' + escapeHtml(job.job_name) + (job.archived ? ' <span style="color:#aaa;font-size:0.75rem;">[archived]</span>' : '') + '</div>' +
                '<div class="job-status ' + statusClass + '">' + escapeHtml(job.status || 'unknown') + '</div>' +
            '</div>' +
            '<div class="job-details">' +
                (cfg.data_s3_uri ? '<div style="font-size:0.8rem;"><strong>Data:</strong> <code style="font-size:0.75rem;">' + escapeHtml(cfg.data_s3_uri) + '</code></div>' : '') +
                (cfg.instance_type ? '<div style="font-size:0.8rem;"><strong>Instance:</strong> ' + escapeHtml(cfg.instance_type) + '</div>' : '') +
                artifactHtml + failHtml +
            '</div>';
        container.appendChild(el);
    });
}

async function mbRefreshJobs() {
    const list = document.getElementById('mb-jobs-list');
    list.innerHTML = 'Loading…';
    try {
        const resp = await fetch('/api/monai-bundle/jobs');
        const data = await resp.json();
        _mbAllJobs = data.jobs || [];
        mbDisplayJobs(_mbAllJobs);
    } catch (e) {
        list.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

async function mbDeploy() {
    const out = document.getElementById('mb-deploy-result');
    const body = {
        endpoint_name: document.getElementById('mb-endpoint-name').value.trim(),
        model_artifacts: document.getElementById('mb-deploy-artifacts').value.trim(),
        output_s3_uri: document.getElementById('mb-deploy-output').value.trim(),
        instance_type: document.getElementById('mb-deploy-instance').value.trim() || 'ml.g5.xlarge',
        instance_count: Number(document.getElementById('mb-deploy-count').value.trim() || 1),
        max_concurrent: Number(document.getElementById('mb-deploy-concurrent').value.trim() || 4),
    };
    if (!body.endpoint_name || !body.model_artifacts || !body.output_s3_uri) {
        alert('endpoint name, trained model artifact, and async output S3 location are required'); return;
    }
    out.style.display = 'block';
    out.innerHTML = 'Deploying async endpoint…';
    try {
        const resp = await fetch('/api/monai-bundle/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'deploy failed');
        const epName = data.endpoint_name || body.endpoint_name;
        mbRenderDeployStatus(out, epName, data.endpoint_type || 'async', data.status || 'Creating');
        mbPollDeployStatus(out, epName, data.endpoint_type || 'async');
        if (typeof mbLoadDeployments === 'function') mbLoadDeployments();
    } catch (e) {
        out.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message) + '</span>';
    }
}

function mbRenderDeployStatus(out, epName, epType, status) {
    const cls = {
        'InService': 'status-completed',
        'Failed':    'status-failed',
        'Creating':  'status-running',
        'Updating':  'status-running',
        'Deleting':  'status-stopping',
    }[status] || 'status-unknown';
    out.innerHTML =
        '<div class="result-row"><span>Endpoint</span> ' + escapeHtml(epName) + '</div>' +
        '<div class="result-row"><span>Type</span> ' + escapeHtml(epType) + '</div>' +
        '<div class="result-row"><span>Status</span> <span class="job-status ' + cls + '">' + escapeHtml(status) + '</span></div>';
}

async function mbPollDeployStatus(out, epName, epType) {
    const TERMINAL = new Set(['InService', 'Failed', 'Deleted']);
    let tries = 0;
    const timer = setInterval(async function () {
        tries += 1;
        try {
            const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(epName) + '/status');
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || 'status check failed');
            const status = d.status || 'Creating';
            let extra = '';
            if (status === 'Failed' && d.failure_reason) {
                extra = '<div class="result-row" style="color:#c62828;"><span>Reason</span> ' + escapeHtml(d.failure_reason) + '</div>';
            }
            mbRenderDeployStatus(out, epName, epType, status);
            if (extra) out.insertAdjacentHTML('beforeend', extra);
            if (TERMINAL.has(status)) clearInterval(timer);
        } catch (e) {
            // transient; keep polling until the cap
        }
        if (tries >= 120) clearInterval(timer); // ~20 min at 10s
    }, 10000);
}

// ── Deploy source (base vs fine-tuned) ──────────────────────────────────────
let _mbArtifacts = { base: [], finetuned: [] };

async function mbLoadArtifacts() {
    try {
        _mbArtifacts = await (await fetch('/api/monai-bundle/artifacts')).json();
    } catch (e) {
        _mbArtifacts = { base: [], finetuned: [] };
    }
    mbOnDeploySourceChange();
}

function _mbSelectedSource() {
    const el = document.querySelector('input[name="mb_deploy_source"]:checked');
    return el ? el.value : 'base';
}

function mbOnDeploySourceChange() {
    const source = _mbSelectedSource();
    const label = document.getElementById('mb-deploy-artifact-label');
    const sel = document.getElementById('mb-deploy-artifact-select');
    const ftList = document.getElementById('mb-deploy-finetuned-list');

    if (source === 'finetuned') {
        // Rich clickable list of completed fine-tune jobs, like the deployment page.
        label.textContent = 'Fine-tuned bundle (completed job)';
        sel.style.display = 'none';
        ftList.style.display = 'block';
        const items = _mbArtifacts.finetuned || [];
        if (!items.length) {
            ftList.innerHTML = '<p style="color:#888; padding:0.6rem;">No completed fine-tune jobs with model artifacts found.</p>';
            return;
        }
        ftList.innerHTML = items.map(function (it) {
            const art = it.model_artifacts || '';
            const meta = [it.task, it.instance_type].filter(Boolean).map(escapeHtml).join(' · ');
            return '' +
                '<div style="padding:0.6rem 0.5rem; border-bottom:1px solid #f0f0f0; cursor:pointer;"' +
                ' onmouseover="this.style.background=\'#f5f5f5\'" onmouseout="this.style.background=\'\'"' +
                ' onclick="mbSelectFinetuned(this, \'' + escapeJsString(art) + '\', \'' + escapeJsString(it.name) + '\')">' +
                '<div style="font-weight:600; font-size:0.9rem;">' + escapeHtml(it.name) + '</div>' +
                '<div style="font-size:0.78rem; color:#666; margin-top:2px; word-break:break-all;">' + escapeHtml(art) + '</div>' +
                (meta ? '<div style="font-size:0.75rem; color:#888; margin-top:2px;">' + meta + '</div>' : '') +
                (it.data_s3_uri ? '<div style="font-size:0.75rem; color:#888; margin-top:2px; word-break:break-all;">data: ' + escapeHtml(it.data_s3_uri) + '</div>' : '') +
                '</div>';
        }).join('');
        return;
    }

    // Base bundles: simple dropdown.
    label.textContent = 'Base bundle (staged)';
    ftList.style.display = 'none';
    sel.style.display = '';
    const items = _mbArtifacts.base || [];
    if (!items.length) {
        sel.innerHTML = '<option value="" disabled selected>No staged base bundles — stage one first</option>';
        return;
    }
    sel.innerHTML = '<option value="" disabled selected>Select…</option>' + items.map(function (it) {
        const artifact = it.model_artifacts || '';
        const deployable = it.deployable !== false && artifact;
        const suffix = deployable ? '' : ' (not staged)';
        return '<option value="' + escapeHtml(artifact) + '"' + (deployable ? '' : ' disabled') + '>' +
            escapeHtml(it.name) + suffix + '</option>';
    }).join('');
}

function mbSelectFinetuned(row, artifact, jobName) {
    if (!artifact) return;
    document.getElementById('mb-deploy-artifacts').value = artifact;
    const nameField = document.getElementById('mb-endpoint-name');
    if (!nameField.value) {
        nameField.value = (jobName || 'monai-bundle').toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 60).replace(/-+$/, '') + '-ep';
    }
    // Highlight the chosen row.
    const list = document.getElementById('mb-deploy-finetuned-list');
    Array.prototype.forEach.call(list.children, function (c) { c.style.background = ''; });
    if (row) row.style.background = '#e8f0fe';
}

function mbOnArtifactSelect() {
    const sel = document.getElementById('mb-deploy-artifact-select');
    const artifact = sel.value;
    if (artifact) {
        document.getElementById('mb-deploy-artifacts').value = artifact;
        // Suggest an endpoint name from the selection if empty.
        const nameField = document.getElementById('mb-endpoint-name');
        if (!nameField.value) {
            const opt = sel.options[sel.selectedIndex];
            const base = (opt.textContent || 'monai-bundle').replace(/[^a-zA-Z0-9-]/g, '-').slice(0, 40).replace(/-+$/, '');
            nameField.value = base + '-ep';
        }
    }
}

// ── Staged bundles list ─────────────────────────────────────────────────────
async function mbLoadStaged() {
    const container = document.getElementById('mb-staged-list');
    if (!container) return;
    try {
        const res = await (await fetch('/api/monai-bundle/artifacts')).json();
        _mbArtifacts = res; // keep deploy selector in sync
        const staged = (res.base || []).filter(function (b) { return b.model_artifacts; });
        mbDisplayStaged(staged);
    } catch (e) {
        container.innerHTML = '<p style="color:#c62828; padding:0.5rem;">Error loading staged bundles: ' + escapeHtml(e.message) + '</p>';
    }
}

function mbDisplayStaged(items) {
    const container = document.getElementById('mb-staged-list');
    if (!container) return;
    if (!items.length) {
        container.innerHTML = '<p style="color:#888; padding:0.5rem;">No bundles staged yet. Stage one above to make it available for fine-tuning and deployment.</p>';
        return;
    }
    container.innerHTML = items.map(function (b) {
        const art = b.model_artifacts || '';
        const task = (b.task || 'segmentation').toLowerCase();
        const taskClass = task === 'segmentation' ? 'model-tag-segmentation' : 'model-tag-classification';
        const taskLabel = task.charAt(0).toUpperCase() + task.slice(1);

        // Key papers (external links, opened safely).
        const papers = (b.papers || []).map(function (p) {
            const title = escapeHtml(p.title || p.url || '');
            return p.url
                ? '<li style="margin-bottom:0.2rem;"><a href="' + escapeHtml(p.url) + '" target="_blank" rel="noopener noreferrer">' + title + '</a></li>'
                : '<li style="margin-bottom:0.2rem;">' + title + '</li>';
        }).join('');
        const papersCol = papers
            ? '<div style="flex:1 1 260px; min-width:0;">' +
                '<div style="font-weight:600; font-size:0.8rem; margin-bottom:0.25rem;"><i class="fas fa-book"></i> Key papers</div>' +
                '<ul style="font-size:0.75rem; margin:0; padding-left:1.1rem; overflow-wrap:anywhere; word-break:break-word;">' + papers + '</ul>' +
              '</div>'
            : '';

        // Data organization (from the bundle's metadata.json). The MSD layout tree
        // appears only when MedAnvil's fine-tune pipeline actually consumes it;
        // bundles with their own training-data format get an honest note instead.
        const trainable = b.trainable !== false;
        const d = b.data_org || {};
        const tree = (d.layout || []).map(escapeHtml).join('\n');
        const rowStyle = 'font-size:0.75rem; color:#555; overflow-wrap:anywhere; word-break:break-word;';
        const dataTitle = tree
            ? '<i class="fas fa-folder-tree"></i> Training data layout (MedAnvil fine-tune pipeline)'
            : '<i class="fas fa-right-left"></i> Model inputs &amp; classes';
        const noteHtml = d.training_data_note
            ? '<div style="' + rowStyle + ' margin-top:0.3rem; padding:0.35rem 0.5rem; background:#fff8e1; border:1px solid #ffe082; border-radius:6px;">' +
              '<i class="fas fa-circle-info"></i> ' + escapeHtml(d.training_data_note) + '</div>'
            : '';
        const dataCol = (d.modality || d.labels || d.data_source || tree || noteHtml)
            ? '<div style="flex:1 1 260px; min-width:0;">' +
                '<div style="font-weight:600; font-size:0.8rem; margin-bottom:0.25rem;">' + dataTitle + '</div>' +
                (d.modality ? '<div style="' + rowStyle + '"><strong>Modality:</strong> ' + escapeHtml(d.modality) + '</div>' : '') +
                (d.labels ? '<div style="' + rowStyle + '"><strong>Classes:</strong> ' + escapeHtml(d.labels) + '</div>' : '') +
                (d.data_source ? '<div style="' + rowStyle + '"><strong>Source:</strong> ' + escapeHtml(d.data_source) + '</div>' : '') +
                (tree ? '<pre style="background:#f6f8fa; border:1px solid #eee; border-radius:6px; padding:0.5rem; font-size:0.72rem; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; margin:0.3rem 0 0;">' + tree + '</pre>' : '') +
                noteHtml +
              '</div>'
            : '';

        const rightCols = (papersCol || dataCol)
            ? '<div style="flex:2 1 420px; min-width:0; display:flex; flex-wrap:wrap; gap:1rem; padding-left:1rem; border-left:1px solid #eee;">' + papersCol + dataCol + '</div>'
            : '';

        // GitHub link for the bundle. Prefer the source_url captured at stage
        // time; otherwise derive it from the bundle name so older records still
        // link out. Always also offer the model-zoo index as a prompt.
        const ZOO = 'https://github.com/Project-MONAI/model-zoo/tree/dev/models';
        const bundleUrl = b.source_url || (b.name ? ZOO + '/' + encodeURIComponent(b.name) : null);
        const derived = !b.source_url;
        const githubHtml =
            '<div style="font-size:0.72rem; margin-bottom:0.5rem;">' +
                '<i class="fab fa-github"></i> ' +
                (bundleUrl
                    ? '<a href="' + escapeHtml(bundleUrl) + '" target="_blank" rel="noopener noreferrer">MONAI bundle on GitHub</a>' +
                      (derived ? ' <span style="color:#999;">(name-derived — verify against the ' +
                                 '<a href="' + escapeHtml(ZOO) + '" target="_blank" rel="noopener noreferrer">model-zoo</a>)</span>' : '')
                    : 'Find this bundle in the <a href="' + escapeHtml(ZOO) + '" target="_blank" rel="noopener noreferrer">MONAI model-zoo</a>') +
            '</div>';

        // Left column: identity + artifact + actions.
        const leftCol =
            '<div style="flex:1 1 220px; min-width:0;">' +
                '<div style="display:flex; align-items:center; gap:6px; margin-bottom:0.35rem; flex-wrap:wrap;">' +
                    '<span class="model-name" style="overflow-wrap:anywhere;">' + escapeHtml(b.name) + '</span>' +
                    '<span class="model-tag ' + taskClass + '">' + escapeHtml(taskLabel) + '</span>' +
                    (!trainable ? '<span class="model-tag" style="background:#fff3e0; color:#e65100; border:1px solid #ffcc80;">Inference-only</span>' : '') +
                '</div>' +
                '<div style="font-size:0.78rem; color:#555; overflow-wrap:anywhere; margin-bottom:0.4rem;">' + escapeHtml(b.description || 'MONAI bundle staged for deployment.') + '</div>' +
                '<div style="font-size:0.72rem; color:#888; overflow-wrap:anywhere; word-break:break-all; margin-bottom:0.5rem;"><strong>Artifact:</strong> ' + escapeHtml(art) + '</div>' +
                githubHtml +
                '<div class="job-actions">' +
                    (trainable ? '<button class="btn btn-secondary" onclick="mbUseStagedForFinetune(\'' + escapeJsString(art) + '\')" title="Fill the fine-tune bundle artifact"><i class="fas fa-dumbbell"></i> Fine-tune</button>' : '') +
                    '<button class="btn btn-secondary" onclick="mbUseStagedForDeploy(\'' + escapeJsString(art) + '\', \'' + escapeJsString(b.name) + '\')" title="Fill the deploy artifact"><i class="fas fa-rocket"></i> Deploy</button>' +
                '</div>' +
            '</div>';

        return '' +
            '<div class="model-item model-item-' + escapeHtml(task) + '" style="min-width:0; overflow:hidden; display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">' +
                leftCol +
                rightCols +
            '</div>';
    }).join('');
}

function mbUseStagedForFinetune(artifact) {
    const el = document.getElementById('mb-model-artifacts');
    if (el) { el.value = artifact; el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
}

function mbUseStagedForDeploy(artifact, name) {
    const base = document.querySelector('input[name="mb_deploy_source"][value="base"]');
    if (base) { base.checked = true; if (typeof mbOnDeploySourceChange === 'function') mbOnDeploySourceChange(); }
    const sel = document.getElementById('mb-deploy-artifact-select');
    if (sel) sel.value = artifact;
    const art = document.getElementById('mb-deploy-artifacts');
    if (art) { art.value = artifact; art.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
    const nameField = document.getElementById('mb-endpoint-name');
    if (nameField && !nameField.value) {
        nameField.value = (name || 'monai-bundle').toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 60).replace(/-+$/, '') + '-ep';
    }
}

// ── Deployed models (MONAI bundle endpoints) ────────────────────────────────
function _mbIsBundleDeployment(d) {
    const img = (d.image_uri || '').toLowerCase();
    const name = (d.endpoint_name || '').toLowerCase();
    return img.indexOf('monai-bundle') !== -1 || name.indexOf('monai-bundle') !== -1;
}

async function mbLoadDeployments() {
    const container = document.getElementById('mb-deployments-list');
    if (!container) return;
    try {
        const res = await (await fetch('/api/deployment/deployments')).json();
        const all = res.deployments || [];
        const items = all.filter(_mbIsBundleDeployment);
        mbDisplayDeployments(items);
    } catch (e) {
        container.innerHTML = '<p style="color:#c62828; padding:0.5rem;">Error loading deployments: ' + escapeHtml(e.message) + '</p>';
    }
}

function mbDisplayDeployments(deployments) {
    const container = document.getElementById('mb-deployments-list');
    if (!container) return;
    if (!deployments.length) {
        container.innerHTML = '<p style="color:#888; padding:0.5rem;">No MONAI bundle endpoints deployed yet.</p>';
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
        if (inService && cnt > 0) {
            actions.push('<button class="btn btn-secondary" onclick="mbScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', 0)" title="Scale to 0 (stop billing)"><i class="fas fa-power-off"></i> Scale to 0</button>');
        }
        if (inService && cnt === 0) {
            actions.push('<button class="btn btn-secondary" onclick="mbScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', 1)" style="color:#1e8449;" title="Scale up to 1"><i class="fas fa-play"></i> Scale up</button>');
        }
        if (inService && cnt > 1) {
            actions.push('<button class="btn btn-secondary" onclick="mbScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', ' + (cnt - 1) + ')" title="Remove one instance"><i class="fas fa-minus"></i></button>');
        }
        if (inService && cnt > 0) {
            actions.push('<button class="btn btn-secondary" onclick="mbScaleEndpoint(\'' + escapeJsString(d.endpoint_name) + '\', ' + (cnt + 1) + ')" title="Add one instance"><i class="fas fa-plus"></i></button>');
        }
        if (!['InService', 'Failed', 'Deleting', 'Deleted'].includes(d.status)) {
            actions.push('<button class="btn btn-secondary" onclick="mbRefreshDeployment(\'' + escapeJsString(d.endpoint_name) + '\')"><i class="fas fa-sync"></i> Refresh</button>');
        }
        if (!['Deleting', 'Deleted'].includes(d.status)) {
            actions.push('<button class="btn btn-secondary" onclick="mbDeleteDeployment(\'' + escapeJsString(d.endpoint_name) + '\')" style="color:#c62828;"><i class="fas fa-trash"></i> Delete</button>');
        }
        return '' +
            '<div class="job-item" style="padding:1rem 1.25rem;">' +
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

async function mbScaleEndpoint(name, count) {
    if (count === 0 && !confirm('Scale ' + name + ' to 0 instances? This stops billing but the endpoint stays configured.')) return;
    try {
        const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(name) + '/scale', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ desired_count: count }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'scale failed');
        mbLoadDeployments();
    } catch (e) { alert('Scale failed: ' + e.message); }
}

async function mbDeleteDeployment(name) {
    if (!confirm('Delete endpoint ' + name + '? This removes the endpoint, config, and model.')) return;
    try {
        const r = await fetch('/api/deployment/deployments/' + encodeURIComponent(name), { method: 'DELETE' });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'delete failed');
        mbLoadDeployments();
    } catch (e) { alert('Delete failed: ' + e.message); }
}

async function mbRefreshDeployment(name) {
    try {
        await fetch('/api/deployment/deployments/' + encodeURIComponent(name) + '/status');
        mbLoadDeployments();
    } catch (e) { /* transient */ }
}

if (typeof registerTabInit === 'function') {
    registerTabInit('monai-bundle', function () { mbRefreshJobs(); mbLoadArtifacts(); mbLoadStaged(); mbLoadDeployments(); });
}