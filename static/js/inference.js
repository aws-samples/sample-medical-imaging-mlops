// Inference JavaScript

registerTabInit('inference', function () {
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

    // Load from local deployments cache immediately (fast, no AWS call)
    try {
        const dep    = await apiCall('/api/deployment/deployments?local_only=true');
        const active = (dep.deployments || []).filter(d => d.status === 'InService');
        if (active.length > 0) {
            select.innerHTML = active.map(d =>
                `<option value="${escapeHtml(d.endpoint_name)}" data-type="${escapeHtml(d.endpoint_type || 'realtime')}" data-output-s3="${escapeHtml(d.output_s3_uri || '')}" data-task="${escapeHtml(d.task_type || 'classification')}">${escapeHtml(d.endpoint_name)}${d.endpoint_type === 'async' ? ' ⚡' : ''}</option>`
            ).join('');
            // Auto-fill output S3 for first selected endpoint
            onInfEndpointChange();
            return;
        }
    } catch (_) {}

    // Fallback: try live AWS endpoint list
    try {
        const res = await apiCall('/api/inference/endpoints');
        if (res.endpoints && res.endpoints.length > 0) {
            select.innerHTML = res.endpoints.map(e =>
                `<option value="${escapeHtml(e.name)}">${escapeHtml(e.name)} (${escapeHtml(e.status)})</option>`
            ).join('');
            return;
        }
    } catch (err) {
        console.warn('Could not load live endpoints:', err.message);
    }

    select.innerHTML = '<option value="" disabled selected>No InService endpoints found</option>';
}

function onInfEndpointChange() {
    const select = document.getElementById('inf-endpoint');
    const outputField = document.getElementById('inf-output-s3');
    if (!select || !outputField) return;

    const selected = select.options[select.selectedIndex];
    if (!selected) return;

    const endpointType = selected.dataset.type || 'realtime';
    const outputS3 = selected.dataset.outputS3 || '';

    // Auto-fill output S3 URI from deployment config for async endpoints
    if (endpointType === 'async' && outputS3) {
        outputField.value = outputS3;
        outputField.parentElement.querySelector('small').textContent = 'Auto-filled from deployment config (async endpoint)';
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
            res = await apiCall(`/api/inference/invoke-local?endpoint_name=${encodeURIComponent(fd.get('endpoint_name'))}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/dicom' },
                body: fileBytes,
            });
        } else {
            const payload = {
                endpoint_name: fd.get('endpoint_name'),
                input_type:    inputType,
                input_value:   inputValue,
                output_s3_uri: fd.get('output_s3_uri') || null,
                output_pattern: fd.get('output_pattern') || null,
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

        if (res.predicted_class || res.probabilities || res.predicted_class_index !== undefined) {
            displayClassificationResult(res);
        } else if (res.segmentation_mask_base64 || res.mask_nifti_b64 || res.segmented_voxels !== undefined) {
            displaySegmentationResult(res);
        } else if (res.async) {
            displayAsyncResult(res);
        } else {
            // Use task type from the endpoint's deployment record
            const epSelect = document.getElementById('inf-endpoint');
            const selectedOpt = epSelect?.options[epSelect.selectedIndex];
            const taskType = selectedOpt?.dataset?.task || 'classification';
            if (taskType === 'segmentation') {
                displaySegmentationResult(res);
            } else {
                displayClassificationResult(res);
            }
        }

        const savedTo = document.getElementById('inf-saved-to');
        if (savedTo) savedTo.textContent = res.saved_to ? `Saved to: ${res.saved_to}` : '';

    } catch (err) {
        if (banner) { banner.className = 'sm-status-banner sm-status-error'; banner.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`; }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Run Inference';
    }
}

function displayAsyncResult(res) {
    const resultDiv = document.getElementById('inference-result');
    if (!resultDiv) return;
    resultDiv.innerHTML = `
        <div style="padding:1rem; background:#e8f4fd; border-radius:6px; border:1px solid #1a6fa0;">
            <h4 style="margin:0 0 0.5rem;"><i class="fas fa-clock"></i> Async Inference Submitted</h4>
            <div style="font-size:0.85rem; display:grid; grid-template-columns:auto 1fr; gap:0.3rem 1rem;">
                <strong>Inference ID:</strong> <code>${escapeHtml(res.inference_id || 'N/A')}</code>
                <strong>Output:</strong> <code style="word-break:break-all;">${escapeHtml(res.output_location || res.message || 'Check S3')}</code>
                <strong>Status:</strong> <span>${escapeHtml(res.status || 'Submitted')}</span>
            </div>
            <p style="margin:0.75rem 0 0; font-size:0.8rem; color:#666;">Results will appear in the output S3 path. Check in 15-30 seconds.</p>
        </div>
    `;
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
                <span class="inf-prob-label">${escapeHtml(cls)}</span>
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
