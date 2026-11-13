// Deployment JavaScript
// Depends on: html``/setHTML()/escapeHtml() from safe_html.js (loaded before this file)
if (typeof escapeHtml !== 'function') {
    throw new Error('escapeHtml() is required but not defined. Ensure safe_html.js is loaded before deployment.js.');
}

const INFERENCE_IMAGE_TEMPLATE = 'YOUR_ACCOUNT_ID.dkr.ecr.{region}.amazonaws.com/sm-inference-byoc-classification:latest';

let DEFAULT_INFERENCE_CONTAINERS = [
    { label: "classification-byoc (BYOC)",   uri: "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest" },
    { label: "segmentation-byoc (BYOC)",     uri: "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest" },
    { label: "pytorch-inference-gpu (AWS)",  uri: "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker" },
    { label: "pytorch-inference-cpu (AWS)",  uri: "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker" },
];

/**
 * Called by index.html after /api/config loads. Replaces placeholder account IDs
 * in the default container list with real values from .env.local.
 */
function updateInferenceContainersFromConfig(cfg) {
    if (!cfg.inference_image_classification && !cfg.inference_image_segmentation) return;
    DEFAULT_INFERENCE_CONTAINERS = [
        { label: "classification-byoc (BYOC)",   uri: cfg.inference_image_classification || DEFAULT_INFERENCE_CONTAINERS[0].uri },
        { label: "segmentation-byoc (BYOC)",     uri: cfg.inference_image_segmentation || DEFAULT_INFERENCE_CONTAINERS[1].uri },
        { label: "pytorch-inference-gpu (AWS)",  uri: "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker" },
        { label: "pytorch-inference-cpu (AWS)",  uri: "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker" },
    ];
    // Clear localStorage to pick up new defaults if account changed
    localStorage.removeItem('inference_containers');
    populateInferenceContainerDropdown();
}

function loadInferenceContainers() {
    const stored = localStorage.getItem('inference_containers');
    return stored ? JSON.parse(stored) : DEFAULT_INFERENCE_CONTAINERS;
}

function saveInferenceContainers(containers) {
    localStorage.setItem('inference_containers', JSON.stringify(containers));
}

function populateInferenceContainerDropdown() {
    const select = document.getElementById('dep-image-uri');
    if (!select) return;
    const containers = loadInferenceContainers();
    setHTML(select, joinHtml(containers.map(c =>
        html`<option value="${c.uri}">${c.label} — ${c.uri}</option>`
    ), ''));
}

function openAddInferenceContainerModal() {
    document.getElementById('new-inf-container-label').value = '';
    document.getElementById('new-inf-container-uri').value   = '';
    document.getElementById('add-inference-container-modal').style.display = 'flex';
}

function closeAddInferenceContainerModal() {
    document.getElementById('add-inference-container-modal').style.display = 'none';
}

function saveNewInferenceContainer() {
    const label = document.getElementById('new-inf-container-label').value.trim();
    const uri   = document.getElementById('new-inf-container-uri').value.trim();
    if (!uri) { notify('Image URI is required.', 'warning'); return; }
    const containers = loadInferenceContainers();
    containers.push({ label: label || uri, uri });
    saveInferenceContainers(containers);
    populateInferenceContainerDropdown();
    document.getElementById('dep-image-uri').value = uri;
    closeAddInferenceContainerModal();
}

async function removeSelectedInferenceContainer() {
    const select = document.getElementById('dep-image-uri');
    if (!select || !select.value) return;
    if (!(await confirmAction(`Remove "${select.options[select.selectedIndex].text}"?`, { danger: true, confirmLabel: 'Remove' }))) return;
    const containers = loadInferenceContainers().filter(c => c.uri !== select.value);
    saveInferenceContainers(containers);
    populateInferenceContainerDropdown();
}

registerTabInit('deployment', function () {
    populateInferenceContainerDropdown();
    updateOutputPatternDefault();
    loadDeployments();
    const form = document.getElementById('deployment-form');
    if (form) form.addEventListener('submit', deployModel);
    setInterval(loadDeployments, 20000);
});

function updateInferenceImageUri() {
    // no-op — dropdown is now used instead of a text input
}

function getDeployTaskType() {
    // Infer task type from the selected image URI label/value
    const select = document.getElementById('dep-image-uri');
    if (!select) return 'classification';
    const selected = select.options[select.selectedIndex];
    const text = ((selected?.text || '') + (selected?.value || '')).toLowerCase();
    return text.includes('segmentation') ? 'segmentation' : 'classification';
}

function updateOutputPatternDefault() {
    const patternInput = document.getElementById('dep-output-pattern');
    if (!patternInput) return;
    const taskType = getDeployTaskType();
    const current  = patternInput.value.trim();
    // Only update if the field still holds one of the known defaults (don't overwrite user edits)
    const knownDefaults = [
        '{patient_id}_segmentation.nii.gz',
        '{patient_id}_classification.json',
    ];
    if (!current || knownDefaults.includes(current)) {
        patternInput.value = taskType === 'segmentation'
            ? '{patient_id}_segmentation.nii.gz'
            : '{patient_id}_classification.json';
    }
}

function toggleAsyncFields() {
    const isAsync = document.querySelector('input[name="dep_inference_type"]:checked')?.value === 'async';
    const fields  = document.getElementById('async-fields');
    if (fields) fields.style.display = isAsync ? 'block' : 'none';
    if (isAsync) updateOutputPatternDefault();

    // Data capture is not supported on async endpoints
    const captureCheckbox = document.getElementById('dep-capture');
    const asyncNote = document.getElementById('dep-capture-async-note');
    if (captureCheckbox) {
        if (isAsync) {
            captureCheckbox.checked = false;
            captureCheckbox.disabled = true;
            captureCheckbox.closest('.checkbox-label').title = 'Data capture is not supported on async endpoints';
            if (asyncNote) asyncNote.style.display = 'block';
            toggleCaptureFields();
        } else {
            captureCheckbox.disabled = false;
            captureCheckbox.closest('.checkbox-label').title = '';
            if (asyncNote) asyncNote.style.display = 'none';
        }
    }
}

function toggleCaptureFields() {
    const enabled = document.getElementById('dep-capture')?.checked;
    const fields  = document.getElementById('capture-fields');
    if (fields) fields.style.display = enabled ? 'block' : 'none';
}

async function deployModel(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const isAsync       = document.querySelector('input[name="dep_inference_type"]:checked')?.value === 'async';
    const modelArtifacts = fd.get('model_artifacts')?.trim() || null;
    const importOnly    = !modelArtifacts;

    const payload = {
        endpoint_name:        fd.get('endpoint_name'),
        model_artifacts:      modelArtifacts,
        image_uri:            importOnly ? null : fd.get('image_uri'),
        role_arn:             importOnly ? null : fd.get('role_arn'),
        instance_type:        fd.get('instance_type'),
        instance_count:       parseInt(fd.get('instance_count') || '1'),
        inference_script:     fd.get('inference_script') || 'inference.py',
        async_endpoint:       isAsync,
        output_s3_uri:        fd.get('output_s3_uri')?.trim() || null,
        enable_data_capture:  document.getElementById('dep-capture')?.checked || false,
        capture_s3_uri:       fd.get('capture_s3_uri') || null,
        capture_sampling_pct: parseInt(fd.get('capture_sampling_pct') || '100'),
    };

    if (isAsync) {
        if (!payload.output_s3_uri) {
            notify('Output S3 URI is required for async endpoints', 'warning');
            return;
        }
        payload.max_concurrent = parseInt(fd.get('max_concurrent') || '10');
        const taskType = getDeployTaskType();
        const defaultPattern = taskType === 'segmentation'
            ? '{patient_id}_segmentation.nii.gz'
            : '{patient_id}_classification.json';
        payload.output_pattern = fd.get('output_pattern') || defaultPattern;
    }

    const btn    = e.target.querySelector('button[type="submit"]');
    const banner = document.getElementById('dep-status-banner');
    const action = importOnly ? 'Importing endpoint...' : 'Creating SageMaker endpoint...';

    btn.disabled = true;
    setHTML(btn, html`<i class="fas fa-spinner fa-spin"></i> ${importOnly ? 'Importing...' : 'Deploying...'})`);
    if (banner) {
        banner.className = 'sm-status-banner sm-status-submitting';
        setHTML(banner, html`<i class="fas fa-spinner fa-spin"></i> ${action}`);
        banner.style.display = 'flex';
    }

    try {
        const result = await apiCall('/api/deployment/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (banner) {
            banner.className = 'sm-status-banner sm-status-success';
            setHTML(banner, importOnly
                ? html`<i class="fas fa-check-circle"></i> Endpoint <strong>${result.endpoint_name}</strong> imported (${result.status})`
                : html`<i class="fas fa-check-circle"></i> Endpoint <strong>${result.endpoint_name}</strong> is being created`);
        }
        loadDeployments();
    } catch (err) {
        if (banner) {
            banner.className = 'sm-status-banner sm-status-error';
            setHTML(banner, html`<i class="fas fa-exclamation-circle"></i> ${err.message}`);
        }
    } finally {
        btn.disabled = false;
        setHTML(btn, html`<i class="fas fa-rocket"></i> Deploy Model`);
    }
}

async function loadDeployments() {
    try {
        const res = await apiCall('/api/deployment/deployments');
        displayDeployments(res.deployments);
    } catch (err) {
        console.error('Error loading deployments:', err);
    }
}

async function syncDeploymentsFromAWS() {
    const btn = document.getElementById('btn-sync-deployments');
    if (btn) { btn.disabled = true; setHTML(btn, html`<i class="fas fa-spinner fa-spin"></i> Syncing...`); }
    try {
        const result = await apiCall('/api/deployment/sync', { method: 'POST' });
        const added = (result.added || []).length;
        const updated = (result.updated || []).length;
        notify(`Synced from AWS: ${added} added, ${updated} updated`, 'success');
        await loadDeployments();
    } catch (error) {
        notify('Sync failed: ' + (error.message || error), 'error');
    } finally {
        if (btn) { btn.disabled = false; setHTML(btn, html`<i class="fas fa-cloud-download-alt"></i> Sync from AWS`); }
    }
}

function displayDeployments(deployments) {
    const container = document.getElementById('deployments-list');
    if (!container) return;

    // Update count badge and active endpoint summary
    const badge = document.getElementById('dep-count-badge');
    if (badge) {
        // An endpoint is truly active only if InService AND has at least 1 instance running
        const active = (deployments || []).filter(d => d.status === 'InService' && (d.instance_count == null || d.instance_count > 0));
        const scaledToZero = (deployments || []).filter(d => d.status === 'InService' && d.instance_count === 0);
        const total = (deployments || []).length;
        if (active.length > 0) {
            const summary = active.map(d => `${d.endpoint_name} (×${d.instance_count != null ? d.instance_count : 1})`).join(', ');
            let badgeText = html`${active.length} active / ${total} total — <span style="font-weight:normal;">${summary}</span>`;
            if (scaledToZero.length > 0) {
                badgeText = html`${badgeText}${html` <span style="font-weight:normal; color:#ffd; opacity:0.85;">(${scaledToZero.length} scaled to 0)</span>`}`;
            }
            setHTML(badge, badgeText);
            badge.style.background = '#1e8449';
        } else if (scaledToZero.length > 0) {
            badge.textContent = `0 active / ${total} total — ${scaledToZero.length} scaled to 0`;
            badge.style.background = '#888';
        } else {
            badge.textContent = `0 active / ${total} total`;
            badge.style.background = '#888';
        }
    }

    if (!deployments || deployments.length === 0) {
        setHTML(container, html`<p>No deployments yet.</p>`);
        return;
    }

    const statusClass = {
        'Creating':    'status-running',
        'Updating':    'status-running',
        'InService':   'status-completed',
        'Failed':      'status-failed',
        'Deleting':    'status-stopping',
        'Deleted':     'status-stopped',
    };

    setHTML(container, joinHtml(deployments.map(d => html`
        <div class="job-item" style="padding:1rem 1.25rem;">
            <div class="job-header" style="margin-bottom:0.75rem;">
                <div>
                    <div class="job-id" style="font-size:1rem;">${d.endpoint_name}</div>
                    <span style="font-size:0.75rem; color:#666;">${d.endpoint_type === 'async' ? '⚡ Async' : '🔄 Real-time'} · ${d.instance_type || 'N/A'} × ${d.instance_count != null ? d.instance_count : 1}${d.instance_count === 0 ? html` <span style="color:#e67e22; font-weight:600;">— scaled to 0</span>` : ''}</span>
                </div>
                <div class="job-status ${statusClass[d.status] || 'status-unknown'}">${d.status}</div>
            </div>
            <div style="display:grid; grid-template-columns:auto 1fr; gap:0.3rem 1rem; font-size:0.82rem; margin-bottom:0.75rem;">
                <strong>Created:</strong> <span>${d.created_at ? new Date(d.created_at).toLocaleString() : 'N/A'}</span>
                ${d.output_s3_uri ? html`<strong>Output:</strong> <span style="word-break:break-all; font-family:monospace; font-size:0.75rem;">${d.output_s3_uri}</span>` : ''}
                ${d.data_capture_enabled ? html`<strong>Capture:</strong> <span><span style="color:#1e8449;">● Enabled</span> → <span style="font-family:monospace; font-size:0.75rem;">${d.capture_s3_uri}</span></span>` : ''}
                ${d.failure_reason ? html`<strong style="color:#c62828;">Error:</strong> <span style="color:#c62828;">${d.failure_reason}</span>` : ''}
            </div>
            <div class="job-actions">
                ${d.status === 'InService' && d.endpoint_type === 'async' && (d.instance_count == null || d.instance_count > 0) ? html`
                <button class="btn btn-secondary" onclick="scaleEndpoint('${escapeJsString(d.endpoint_name)}', 0)" title="Scale to 0 instances (stop billing)">
                    <i class="fas fa-power-off"></i> Scale to 0
                </button>
                ` : ''}
                ${d.status === 'InService' && d.instance_count === 0 ? html`
                <button class="btn btn-secondary" onclick="scaleEndpoint('${escapeJsString(d.endpoint_name)}', 1)" title="Scale up to 1 instance" style="color:#1e8449;">
                    <i class="fas fa-play"></i> Scale up
                </button>
                ` : ''}
                ${d.status === 'InService' && d.instance_count != null && d.instance_count > 1 ? html`
                <button class="btn btn-secondary" onclick="scaleEndpoint('${escapeJsString(d.endpoint_name)}', ${jsArg((Number(d.instance_count) || 0) - 1)})" title="Remove one instance">
                    <i class="fas fa-minus"></i>
                </button>
                ` : ''}
                ${d.status === 'InService' && (d.instance_count == null || d.instance_count > 0) ? html`
                <button class="btn btn-secondary" onclick="scaleEndpoint('${escapeJsString(d.endpoint_name)}', ${jsArg((d.instance_count != null ? (Number(d.instance_count) || 0) : 1) + 1)})" title="Add one instance">
                    <i class="fas fa-plus"></i>
                </button>
                ` : ''}
                ${!['InService', 'Failed', 'Deleting', 'Deleted'].includes(d.status) ? html`
                <button class="btn btn-secondary" onclick="refreshDeployment('${escapeJsString(d.endpoint_name)}')">
                    <i class="fas fa-sync"></i> Refresh
                </button>` : ''}
                ${!['Deleting','Deleted'].includes(d.status) ? html`
                <button class="btn btn-secondary" onclick="deleteDeployment('${escapeJsString(d.endpoint_name)}')" style="color:#c62828;">
                    <i class="fas fa-trash"></i> Delete
                </button>` : ''}
            </div>
        </div>
    `), ''));
}

async function refreshDeployment(endpointName) {
    try {
        await apiCall(`/api/deployment/deployments/${endpointName}/status`);
        loadDeployments();
    } catch (err) {
        notify(`Error: ${err.message}`, 'error');
    }
}

async function deleteDeployment(endpointName) {
    if (!(await confirmAction(`Delete endpoint "${endpointName}"? This cannot be undone.`, { danger: true, confirmLabel: 'Delete' }))) return;
    try {
        await apiCall(`/api/deployment/deployments/${endpointName}`, { method: 'DELETE' });
        loadDeployments();
    } catch (err) {
        notify(`Error: ${err.message}`, 'error');
    }
}

async function scaleEndpoint(endpointName, desiredCount) {
    const action = desiredCount === 0 ? 'scale down to 0 (stops billing)' : `scale to ${desiredCount} instance(s)`;
    if (!(await confirmAction(`${action} for "${endpointName}"?`))) return;
    try {
        await apiCall(`/api/deployment/deployments/${endpointName}/scale`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ desired_count: desiredCount }),
        });
        loadDeployments();
    } catch (err) {
        notify(`Scale failed: ${err.message}`, 'error');
    }
}

async function openFillFromJobModal() {
    const modal = document.getElementById('fill-from-job-modal');
    const list  = document.getElementById('fill-job-list');
    modal.style.display = 'flex';
    setHTML(list, html`<p style="color:#888; padding:0.5rem;">Loading jobs...</p>`);

    try {
        const res  = await apiCall('/api/training/sagemaker/jobs');
        const jobs = (res.jobs || []).filter(j => j.status === 'Completed' && j.model_artifacts);

        if (!jobs.length) {
            setHTML(list, html`<p style="color:#888; padding:0.5rem;">No completed jobs with model artifacts found.</p>`);
            return;
        }

        setHTML(list, joinHtml(jobs.map(j => html`
            <div style="padding:0.6rem 0.5rem; border-bottom:1px solid #f0f0f0; cursor:pointer; border-radius:4px;"
                 onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background=''"
                 onclick="fillFromJob('${escapeJsString(j.model_artifacts)}', '${escapeJsString(j.job_name)}')">
                <div style="font-weight:600; font-size:0.9rem;">${j.job_name}</div>
                <div style="font-size:0.78rem; color:#666; margin-top:2px;">${j.model_artifacts}</div>
                <div style="font-size:0.75rem; color:#888; margin-top:2px;">
                    ${j.config?.model_name || ''} · ${j.config?.instance_type || ''} · ${j.config?.epochs || ''} epochs
                </div>
            </div>
        `), ''));
    } catch (err) {
        setHTML(list, html`<p style="color:#c62828; padding:0.5rem;">Error loading jobs: ${err.message}</p>`);
    }
}

function fillFromJob(modelArtifacts, jobName) {
    document.getElementById('dep-model-artifacts').value = modelArtifacts;
    // Auto-suggest endpoint name from job name if empty
    const epInput = document.getElementById('dep-endpoint-name');
    if (!epInput.value) {
        epInput.value = jobName.toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 63);
    }
    closeFillFromJobModal();
}

function closeFillFromJobModal() {
    document.getElementById('fill-from-job-modal').style.display = 'none';
}
