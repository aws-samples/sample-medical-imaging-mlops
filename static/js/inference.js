// Inference JavaScript

registerTabInit('inference', function () {
    loadInferenceEndpoints();
    const form = document.getElementById('inference-form');
    if (form) form.addEventListener('submit', runInference);
});

// Fallback: also load on DOMContentLoaded in case tab init doesn't fire
document.addEventListener('DOMContentLoaded', function() {
    const select = document.getElementById('inf-endpoint');
    if (select && select.innerHTML.includes('Loading endpoints')) {
        loadInferenceEndpoints();
    }
    const form = document.getElementById('inference-form');
    if (form && !form._listenerAdded) {
        form.addEventListener('submit', runInference);
        form._listenerAdded = true;
    }
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

    // Try live AWS endpoint list first (most accurate)
    try {
        const res = await apiCall('/api/inference/endpoints');
        if (res.endpoints && res.endpoints.length > 0) {
            // Merge with local data for task_type and output_s3_uri
            let localData = {};
            try {
                const dep = await apiCall('/api/deployment/deployments?local_only=true');
                (dep.deployments || []).forEach(d => { localData[d.endpoint_name] = d; });
            } catch (_) {}

            select.innerHTML = res.endpoints.map(e => {
                const local = localData[e.name] || {};
                return `<option value="${escapeHtml(e.name)}" data-type="${escapeHtml(local.endpoint_type || 'realtime')}" data-output-s3="${escapeHtml(local.output_s3_uri || '')}" data-task="${escapeHtml(local.task_type || 'classification')}">${escapeHtml(e.name)}${local.endpoint_type === 'async' ? ' ⚡' : ''}</option>`;
            }).join('');
            onInfEndpointChange();
            return;
        }
    } catch (err) {
        console.warn('Could not load live endpoints:', err.message);
    }

    // Fallback: load from local deployments cache
    try {
        const dep = await apiCall('/api/deployment/deployments?local_only=true');
        const active = (dep.deployments || []).filter(d => d.status === 'InService');
        if (active.length > 0) {
            select.innerHTML = active.map(d =>
                `<option value="${escapeHtml(d.endpoint_name)}" data-type="${escapeHtml(d.endpoint_type || 'realtime')}" data-output-s3="${escapeHtml(d.output_s3_uri || '')}" data-task="${escapeHtml(d.task_type || 'classification')}">${escapeHtml(d.endpoint_name)}${d.endpoint_type === 'async' ? ' ⚡' : ''}</option>`
            ).join('');
            onInfEndpointChange();
            return;
        }
    } catch (_) {}

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

        if (placeholder) placeholder.style.display = 'none';
        if (result) result.style.display = 'block';

        if (res.async) {
            // Async endpoint — show submission card and start polling
            if (banner) { banner.className = 'sm-status-banner sm-status-submitting'; banner.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Async inference submitted — waiting for result...'; }
            displayAsyncResult(res);
        } else if (res.predicted_class || res.probabilities || res.predicted_class_index !== undefined) {
            if (banner) { banner.className = 'sm-status-banner sm-status-success'; banner.innerHTML = '<i class="fas fa-check-circle"></i> Inference complete'; }
            displayClassificationResult(res);
        } else if (res.segmentation_mask_base64 || res.mask_nifti_b64 || res.segmented_voxels !== undefined) {
            if (banner) { banner.className = 'sm-status-banner sm-status-success'; banner.innerHTML = '<i class="fas fa-check-circle"></i> Inference complete'; }
            displaySegmentationResult(res);
        } else {
            // Use task type from the endpoint's deployment record
            if (banner) { banner.className = 'sm-status-banner sm-status-success'; banner.innerHTML = '<i class="fas fa-check-circle"></i> Inference complete'; }
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
        <div id="async-result-card" style="padding:1rem; background:#e8f4fd; border-radius:6px; border:1px solid #1a6fa0;">
            <h4 style="margin:0 0 0.5rem;"><i class="fas fa-spinner fa-spin"></i> Async Inference In Progress</h4>
            <div style="font-size:0.85rem; display:grid; grid-template-columns:auto 1fr; gap:0.3rem 1rem;">
                <strong>Inference ID:</strong> <code>${escapeHtml(res.inference_id || 'N/A')}</code>
                <strong>Output:</strong> <code style="word-break:break-all;">${escapeHtml(res.output_location || res.message || 'Check S3')}</code>
                <strong>Status:</strong> <span id="async-poll-status">Submitted — polling for result...</span>
            </div>
            <div id="async-poll-progress" style="margin-top:0.75rem; font-size:0.8rem; color:#666;">
                <i class="fas fa-hourglass-half"></i> Waiting for result...
            </div>
        </div>
    `;

    // Poll for async result completion
    if (res.output_location) {
        pollAsyncResult(res.output_location, res.inference_id);
    }
}

async function pollAsyncResult(outputLocation, inferenceId) {
    const statusEl = document.getElementById('async-poll-status');
    const progressEl = document.getElementById('async-poll-progress');
    const card = document.getElementById('async-result-card');
    const maxWait = 300; // 5 minutes
    const interval = 5;  // poll every 5 seconds
    let elapsed = 0;

    while (elapsed < maxWait) {
        await new Promise(r => setTimeout(r, interval * 1000));
        elapsed += interval;

        if (progressEl) progressEl.innerHTML = `<i class="fas fa-hourglass-half"></i> Waiting... (${elapsed}s)`;

        try {
            // Check if result is ready via the backend
            const res = await apiCall(`/api/inference/async-status?output_location=${encodeURIComponent(outputLocation)}`);
            if (res.completed) {
                if (statusEl) statusEl.innerHTML = '<span style="color:#1e8449;"><i class="fas fa-check-circle"></i> Completed</span>';
                if (progressEl) progressEl.innerHTML = '';
                if (card) {
                    card.style.background = '#e8f8f0';
                    card.style.borderColor = '#1e8449';
                    card.querySelector('h4').innerHTML = '<i class="fas fa-check-circle" style="color:#1e8449;"></i> Async Inference Complete';

                    // Show the raw .out file contents
                    const rawOutputDiv = document.createElement('div');
                    rawOutputDiv.style.cssText = 'margin-top:0.75rem; font-size:0.8rem;';
                    const outputUri = res.result?.mask_s3_uri || outputLocation;
                    rawOutputDiv.innerHTML = `
                        <div style="margin-bottom:0.5rem;">
                            <strong>Output S3:</strong> <code style="word-break:break-all;">${escapeHtml(outputUri)}</code>
                        </div>
                        <details>
                            <summary style="cursor:pointer; color:#1a6fa0; font-weight:500;"><i class="fas fa-file-code"></i> Raw response (.out file)</summary>
                            <pre style="margin-top:0.5rem; padding:0.75rem; background:#f4f6f9; border-radius:4px; overflow-x:auto; font-size:0.75rem; max-height:300px; overflow-y:auto;">${escapeHtml(JSON.stringify(res.result, null, 2))}</pre>
                        </details>
                    `;
                    card.appendChild(rawOutputDiv);
                }
                // Update the top banner too
                const banner = document.getElementById('inf-status-banner');
                if (banner) {
                    banner.className = 'sm-status-banner sm-status-success';
                    banner.innerHTML = '<i class="fas fa-check-circle"></i> Async inference complete';
                }
                // Display the segmentation result
                if (res.result) {
                    const segDiv = document.getElementById('inf-seg-result');
                    if (segDiv) segDiv.style.display = 'block';
                    displaySegmentationResult(res.result);
                }
                return;
            }
        } catch (e) {
            // Not ready yet or API error, continue polling
            console.warn('Async poll error:', e.message);
        }
    }

    // Timed out
    if (statusEl) statusEl.innerHTML = '<span style="color:#c62828;">Timed out — check S3 manually</span>';
    if (progressEl) progressEl.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Result not available after ${maxWait}s. It may still be processing.`;
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

    // Show output file location and raw response
    const outputEl = document.getElementById('inf-saved-to');
    if (outputEl) {
        const savedHtml = res.saved_to
            ? `<div style="padding:0.5rem; background:#e8f8f0; border-radius:4px; font-size:0.8rem; margin-bottom:0.5rem;"><i class="fas fa-check-circle" style="color:#1e8449;"></i> Saved to: <code style="word-break:break-all;">${escapeHtml(res.saved_to)}</code></div>`
            : '';
        const rawHtml = `
            <details style="margin-top:0.5rem; font-size:0.8rem;">
                <summary style="cursor:pointer; color:#1a6fa0; font-weight:500;"><i class="fas fa-file-code"></i> Raw response</summary>
                <pre style="margin-top:0.5rem; padding:0.75rem; background:#f4f6f9; border-radius:4px; overflow-x:auto; font-size:0.75rem; max-height:300px; overflow-y:auto;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
            </details>
        `;
        outputEl.innerHTML = savedHtml + rawHtml;
    }
}

function displaySegmentationResult(res) {
    const card = document.getElementById('inf-class-result');
    const segDiv = document.getElementById('inf-seg-result');
    if (card) card.style.display = 'none';
    if (segDiv) segDiv.style.display = 'block';

    const stats = document.getElementById('inf-seg-stats');
    if (stats) {
        const maskUri = res.mask_s3_uri || res.saved_to || '';
        stats.innerHTML = `
            <div class="status-detail-row"><span>Model:</span><strong>${escapeHtml(res.model_name || 'N/A')}</strong></div>
            <div class="status-detail-row"><span>Original shape:</span><strong>${(res.original_shape || res.shape || []).join(' × ')}</strong></div>
            <div class="status-detail-row"><span>Segmentation shape:</span><strong>${(res.segmentation_shape || []).join(' × ')}</strong></div>
            <div class="status-detail-row"><span>Segmented voxels:</span><strong>${(res.segmented_voxels || res.voxels_positive || 0).toLocaleString()}</strong></div>
            <div class="status-detail-row"><span>Total voxels:</span><strong>${(res.total_voxels || 0).toLocaleString()}</strong></div>
            <div class="status-detail-row"><span>Segmentation %:</span><strong>${res.segmentation_percentage != null ? res.segmentation_percentage + '%' : (res.volume_fraction != null ? (res.volume_fraction * 100).toFixed(2) + '%' : 'N/A')}</strong></div>
            ${maskUri ? `<div style="margin-top:0.75rem; padding:0.5rem; background:#e8f8f0; border-radius:4px; font-size:0.8rem;"><i class="fas fa-check-circle" style="color:#1e8449;"></i> Mask saved to: <code style="word-break:break-all;">${escapeHtml(maskUri)}</code></div>` : ''}
        `;
    }
}
