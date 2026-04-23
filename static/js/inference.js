// Inference JavaScript

document.addEventListener('DOMContentLoaded', function () {
    loadInferenceEndpoints();
    const form = document.getElementById('inference-form');
    if (form) form.addEventListener('submit', runInference);
});

function onInfInputTypeChange(val) {
    const browseBtn = document.getElementById('inf-browse-btn');
    const inputEl   = document.getElementById('inf-input-value');
    if (val === 'file_path') {
        browseBtn.style.display = '';
        inputEl.placeholder = '/path/to/image.dcm';
    } else {
        browseBtn.style.display = 'none';
        inputEl.placeholder = 's3://bucket/path/image.dcm';
        document.getElementById('inf-file-picker').value = '';
    }
}

async function loadInferenceEndpoints() {
    const select = document.getElementById('inf-endpoint');
    if (!select) return;
    try {
        // Try live AWS list first
        const res = await apiCall('/api/inference/endpoints');
        if (res.endpoints && res.endpoints.length > 0) {
            select.innerHTML = res.endpoints.map(e =>
                `<option value="${e.name}">${e.name} (${e.status})</option>`
            ).join('');
            return;
        }
    } catch (err) {
        console.warn('Could not load live endpoints:', err.message);
    }

    // Fallback: use locally tracked deployments
    try {
        const dep = await apiCall('/api/deployment/deployments');
        const active = (dep.deployments || []).filter(d => d.status === 'InService');
        if (active.length > 0) {
            select.innerHTML = active.map(d =>
                `<option value="${d.endpoint_name}">${d.endpoint_name}</option>`
            ).join('');
        } else {
            select.innerHTML = '<option value="" disabled selected>No InService endpoints found</option>';
        }
    } catch (err) {
        select.innerHTML = '<option value="" disabled selected>Could not load endpoints</option>';
    }
}

async function runInference(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const inputType  = fd.get('input_type');
    const inputValue = fd.get('input_value');
    const filePicker = document.getElementById('inf-file-picker');

    const btn    = e.target.querySelector('button[type="submit"]');
    const banner = document.getElementById('inf-status-banner');
    const result = document.getElementById('inference-result');
    const placeholder = document.getElementById('inference-result-placeholder');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
    if (banner) { banner.className = 'sm-status-banner sm-status-submitting'; banner.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Invoking endpoint...'; banner.style.display = 'flex'; }
    if (result) result.style.display = 'none';
    if (placeholder) placeholder.style.display = 'block';

    try {
        let res;
        if (inputType === 'file_path' && filePicker.files.length > 0) {
            // Send raw file bytes
            const fileBytes = await filePicker.files[0].arrayBuffer();
            res = await apiCall(`/api/inference/invoke-local?endpoint_name=${encodeURIComponent(fd.get('endpoint_name'))}&task_type=${encodeURIComponent(fd.get('task_type'))}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/dicom' },
                body: fileBytes,
            });
        } else {
            const payload = {
                endpoint_name: fd.get('endpoint_name'),
                task_type:     fd.get('task_type'),
                input_type:    inputType,
                input_value:   inputValue,
                output_s3_uri: fd.get('output_s3_uri') || null,
            };
            res = await apiCall('/api/inference/invoke', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
        }

        if (banner) { banner.className = 'sm-status-banner sm-status-success'; banner.innerHTML = '<i class="fas fa-check-circle"></i> Inference complete'; }
        if (placeholder) placeholder.style.display = 'none';
        if (result) result.style.display = 'block';

        if (fd.get('task_type') === 'classification') {
            displayClassificationResult(res);
        } else {
            displaySegmentationResult(res);
        }

        const savedTo = document.getElementById('inf-saved-to');
        if (savedTo) savedTo.textContent = res.saved_to ? `Saved to: ${res.saved_to}` : '';

    } catch (err) {
        if (banner) { banner.className = 'sm-status-banner sm-status-error'; banner.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${err.message}`; }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Run Inference';
    }
}

function displayClassificationResult(res) {
    const card = document.getElementById('inf-class-result');
    const segDiv = document.getElementById('inf-seg-result');
    if (card) card.style.display = 'block';
    if (segDiv) segDiv.style.display = 'none';

    const predClass  = res.predicted_class || res.predicted_class_index;
    const confidence = res.confidence != null ? (res.confidence * 100).toFixed(1) + '%' : 'N/A';

    const classEl = card?.querySelector('.inf-predicted-class');
    const confEl  = card?.querySelector('.inf-confidence');
    if (classEl) classEl.textContent = predClass;
    if (confEl)  confEl.textContent  = `Confidence: ${confidence}`;

    // Probability bars
    const barsEl = document.getElementById('inf-probabilities');
    if (barsEl && res.probabilities) {
        const sorted = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1]);
        barsEl.innerHTML = sorted.map(([cls, prob]) => `
            <div class="inf-prob-row">
                <span class="inf-prob-label">${cls}</span>
                <div class="inf-prob-bar-wrap">
                    <div class="inf-prob-bar" style="width:${(prob * 100).toFixed(1)}%"></div>
                </div>
                <span class="inf-prob-value">${(prob * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
}

function displaySegmentationResult(res) {
    const card = document.getElementById('inf-class-result');
    const segDiv = document.getElementById('inf-seg-result');
    if (card) card.style.display = 'none';
    if (segDiv) segDiv.style.display = 'block';

    const stats = document.getElementById('inf-seg-stats');
    if (stats) {
        stats.innerHTML = `
            <div class="status-detail-row"><span>Shape:</span><strong>${(res.shape || []).join(' × ')}</strong></div>
            <div class="status-detail-row"><span>Positive voxels:</span><strong>${res.voxels_positive?.toLocaleString() || 'N/A'}</strong></div>
            <div class="status-detail-row"><span>Volume fraction:</span><strong>${res.volume_fraction != null ? (res.volume_fraction * 100).toFixed(2) + '%' : 'N/A'}</strong></div>
        `;
    }
}
