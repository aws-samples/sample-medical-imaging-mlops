// Deployment JavaScript

const INFERENCE_IMAGE_TEMPLATE = '763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker';

document.addEventListener('DOMContentLoaded', function () {
    loadDeployments();
    const form = document.getElementById('deployment-form');
    if (form) form.addEventListener('submit', deployModel);
    setInterval(loadDeployments, 20000);

    // Update inference image URI when region changes
    const regionSelect = document.getElementById('region');
    if (regionSelect) {
        regionSelect.addEventListener('change', updateInferenceImageUri);
        updateInferenceImageUri();
    }
});

function updateInferenceImageUri() {
    const region  = document.getElementById('region')?.value || 'us-east-1';
    const imgInput = document.getElementById('dep-image-uri');
    if (imgInput) {
        imgInput.value = INFERENCE_IMAGE_TEMPLATE.replace('{region}', region);
    }
}

function toggleAsyncFields() {
    const isAsync = document.getElementById('dep-async')?.checked;
    const fields  = document.getElementById('async-fields');
    if (fields) fields.style.display = isAsync ? 'block' : 'none';
}

async function deployModel(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const isAsync       = document.getElementById('dep-async')?.checked || false;
    const modelArtifacts = fd.get('model_artifacts')?.trim() || null;
    const importOnly    = !modelArtifacts;

    const payload = {
        endpoint_name:    fd.get('endpoint_name'),
        model_artifacts:  modelArtifacts,
        image_uri:        importOnly ? null : fd.get('image_uri'),
        role_arn:         importOnly ? null : fd.get('role_arn'),
        instance_type:    fd.get('instance_type'),
        instance_count:   parseInt(fd.get('instance_count') || '1'),
        inference_script: fd.get('inference_script') || 'inference.py',
        async_endpoint:   isAsync,
    };

    if (isAsync) {
        const outputUri = fd.get('output_s3_uri');
        if (!outputUri) {
            alert('Output S3 URI is required for async endpoints');
            return;
        }
        payload.output_s3_uri  = outputUri;
        payload.max_concurrent = parseInt(fd.get('max_concurrent') || '10');
    }

    const btn    = e.target.querySelector('button[type="submit"]');
    const banner = document.getElementById('dep-status-banner');
    const action = importOnly ? 'Importing endpoint...' : 'Creating SageMaker endpoint...';

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + (importOnly ? 'Importing...' : 'Deploying...');
    if (banner) {
        banner.className = 'sm-status-banner sm-status-submitting';
        banner.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${action}`;
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
            banner.innerHTML = importOnly
                ? `<i class="fas fa-check-circle"></i> Endpoint <strong>${result.endpoint_name}</strong> imported (${result.status})`
                : `<i class="fas fa-check-circle"></i> Endpoint <strong>${result.endpoint_name}</strong> is being created`;
        }
        loadDeployments();
    } catch (err) {
        if (banner) {
            banner.className = 'sm-status-banner sm-status-error';
            banner.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${err.message}`;
        }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-rocket"></i> Deploy Model';
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

function displayDeployments(deployments) {
    const container = document.getElementById('deployments-list');
    if (!container) return;

    if (!deployments || deployments.length === 0) {
        container.innerHTML = '<p>No deployments yet.</p>';
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

    container.innerHTML = deployments.map(d => `
        <div class="job-item">
            <div class="job-header">
                <div class="job-id">${d.endpoint_name}</div>
                <div class="job-status ${statusClass[d.status] || 'status-unknown'}">${d.status}</div>
            </div>
            <div class="job-details">
                <div><strong>Type:</strong> ${d.endpoint_type === 'async' ? '⚡ Async' : '🔄 Real-time'}</div>
                <div><strong>Instance:</strong> ${d.instance_type} × ${d.instance_count}</div>
                <div><strong>Created:</strong> ${d.created_at ? new Date(d.created_at).toLocaleString() : 'N/A'}</div>
                ${d.output_s3_uri ? `<div><strong>Output:</strong> <code style="font-size:0.78rem;">${d.output_s3_uri}</code></div>` : ''}
                ${d.endpoint_url ? `<div><strong>URL:</strong> <code style="font-size:0.78rem;">${d.endpoint_url}</code></div>` : ''}
                ${d.failure_reason ? `<div style="color:#c62828;"><strong>Failure:</strong> ${d.failure_reason}</div>` : ''}
            </div>
            <div class="job-actions">
                ${!['InService', 'Failed', 'Deleting', 'Deleted'].includes(d.status) ? `
                <button class="btn btn-secondary" onclick="refreshDeployment('${d.endpoint_name}')">
                    <i class="fas fa-sync"></i> Refresh
                </button>` : ''}
                ${!['Deleting','Deleted'].includes(d.status) ? `
                <button class="btn btn-secondary" onclick="deleteDeployment('${d.endpoint_name}')" style="color:#c62828;">
                    <i class="fas fa-trash"></i> Delete
                </button>` : ''}
            </div>
        </div>
    `).join('');
}

async function refreshDeployment(endpointName) {
    try {
        await apiCall(`/api/deployment/deployments/${endpointName}/status`);
        loadDeployments();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function deleteDeployment(endpointName) {
    if (!confirm(`Delete endpoint "${endpointName}"? This cannot be undone.`)) return;
    try {
        await apiCall(`/api/deployment/deployments/${endpointName}`, { method: 'DELETE' });
        loadDeployments();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function openFillFromJobModal() {
    const modal = document.getElementById('fill-from-job-modal');
    const list  = document.getElementById('fill-job-list');
    modal.style.display = 'flex';
    list.innerHTML = '<p style="color:#888; padding:0.5rem;">Loading jobs...</p>';

    try {
        const res  = await apiCall('/api/training/sagemaker/jobs');
        const jobs = (res.jobs || []).filter(j => j.status === 'Completed' && j.model_artifacts);

        if (!jobs.length) {
            list.innerHTML = '<p style="color:#888; padding:0.5rem;">No completed jobs with model artifacts found.</p>';
            return;
        }

        list.innerHTML = jobs.map(j => `
            <div style="padding:0.6rem 0.5rem; border-bottom:1px solid #f0f0f0; cursor:pointer; border-radius:4px;"
                 onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background=''"
                 onclick="fillFromJob('${j.model_artifacts}', '${j.job_name}')">
                <div style="font-weight:600; font-size:0.9rem;">${j.job_name}</div>
                <div style="font-size:0.78rem; color:#666; margin-top:2px;">${j.model_artifacts}</div>
                <div style="font-size:0.75rem; color:#888; margin-top:2px;">
                    ${j.config?.model_name || ''} · ${j.config?.instance_type || ''} · ${j.config?.epochs || ''} epochs
                </div>
            </div>
        `).join('');
    } catch (err) {
        list.innerHTML = `<p style="color:#c62828; padding:0.5rem;">Error loading jobs: ${err.message}</p>`;
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
