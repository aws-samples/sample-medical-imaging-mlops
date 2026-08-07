// Evaluation JavaScript
// Depends on: escapeHtml() defined in main.js (loaded before this file)
if (typeof escapeHtml !== 'function') {
    throw new Error('escapeHtml() is required but not defined. Ensure main.js is loaded before evaluation.js.');
}

registerTabInit('evaluation', function () {
    loadEvalEndpoints();
    loadEvalJobs();
    const form = document.getElementById('evaluation-form');
    if (form) form.addEventListener('submit', submitEvaluation);
    setInterval(loadEvalJobs, 30000);
});

async function loadEvalEndpoints() {
    const select = document.getElementById('eval-endpoint');
    if (!select) return;
    try {
        const res = await apiCall('/api/inference/endpoints');
        const eps = res.endpoints || [];
        select.innerHTML = eps.length
            ? eps.map(e => `<option value="${escapeHtml(e.name)}">${escapeHtml(e.name)} (${escapeHtml(e.status)})</option>`).join('')
            : '<option value="" disabled selected>No InService endpoints found</option>';
    } catch (_) {
        select.innerHTML = '<option value="" disabled selected>Could not load endpoints</option>';
    }
}

async function submitEvaluation(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const endpointVal = document.getElementById('eval-endpoint-manual')?.value?.trim()
        || fd.get('endpoint_name');

    const payload = {
        task_type:     fd.get('task_type'),
        endpoint_name: endpointVal,
        data_s3_uri:   fd.get('data_s3_uri'),
        output_s3_uri: fd.get('output_s3_uri'),
        image_uri:     fd.get('image_uri'),
        role_arn:      fd.get('role_arn'),
        instance_type: fd.get('instance_type'),
    };

    const btn    = e.target.querySelector('button[type="submit"]');
    const banner = document.getElementById('eval-submit-banner');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    showBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Submitting evaluation job...');

    try {
        const result = await apiCall('/api/evaluation/submit', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });
        showBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Job <strong>${escapeHtml(result.job_name)}</strong> submitted`);
        loadEvalJobs();
    } catch (err) {
        showBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play-circle"></i> Submit Evaluation Job';
    }
}

async function loadEvalJobs() {
    const container = document.getElementById('eval-jobs-list');
    if (!container) return;
    try {
        const res = await apiCall('/api/evaluation/jobs');
        renderEvalJobs(res.jobs || []);
    } catch (err) {
        container.innerHTML = `<p style="color:#c62828;">Error loading jobs: ${escapeHtml(err.message)}</p>`;
    }
}

function renderEvalJobs(jobs) {
    const container = document.getElementById('eval-jobs-list');
    if (!jobs.length) {
        container.innerHTML = '<p style="color:#888;">No evaluation jobs yet.</p>';
        return;
    }

    const statusClass = {
        'InProgress': 'status-running',
        'Completed':  'status-completed',
        'Failed':     'status-failed',
        'Stopping':   'status-stopping',
        'Stopped':    'status-stopped',
    };

    container.innerHTML = jobs.map(j => `
        <div class="job-item">
            <div class="job-header">
                <div class="job-id">${escapeHtml(j.job_name)}</div>
                <div class="job-status ${statusClass[j.status] || 'status-unknown'}">${escapeHtml(j.status)}</div>
            </div>
            <div class="job-details">
                <div><strong>Endpoint:</strong> ${escapeHtml(j.endpoint_name)}</div>
                <div><strong>Task:</strong> ${escapeHtml(j.task_type || 'classification')}</div>
                <div><strong>Instance:</strong> ${escapeHtml(j.instance_type)}</div>
                <div><strong>Data:</strong> <code style="font-size:0.78rem;">${escapeHtml(j.data_s3_uri)}</code></div>
                <div><strong>Output:</strong> <code style="font-size:0.78rem;">${escapeHtml(j.output_s3_uri)}</code></div>
                <div><strong>Submitted:</strong> ${j.created_at ? escapeHtml(new Date(j.created_at).toLocaleString()) : 'N/A'}</div>
                ${j.completed_at ? `<div><strong>Completed:</strong> ${escapeHtml(new Date(j.completed_at).toLocaleString())}</div>` : ''}
                ${j.failure_reason ? `<div style="color:#c62828;"><strong>Failure:</strong> ${escapeHtml(j.failure_reason)}</div>` : ''}
                ${j.metrics ? renderInlineMetrics(j.metrics, j.task_type) : ''}
            </div>
            <div class="job-actions">
                ${j.status === 'InProgress' ? `
                <button class="btn btn-secondary" onclick="refreshEvalJob('${escapeHtml(j.job_name)}')">
                    <i class="fas fa-sync"></i> Refresh
                </button>` : ''}
                ${j.status === 'Completed' ? `
                <button class="btn btn-secondary" onclick="downloadEvalResults('${escapeHtml(j.job_name)}', this)">
                    <i class="fas fa-download"></i> ${j.cached_results ? 'Re-download' : 'Download Results'}
                </button>` : ''}
                ${j.cached_results ? `
                <button class="btn btn-secondary" onclick="clearEvalCache('${escapeHtml(j.job_name)}', this)" style="color:#c62828;" title="Remove locally cached files">
                    <i class="fas fa-trash"></i> Clear Cache
                </button>` : ''}
                <button class="btn btn-secondary" onclick="deleteEvalJob('${escapeHtml(j.job_name)}', this)" style="color:#c62828;" title="Delete this job record">
                    <i class="fas fa-times-circle"></i> Delete
                </button>
                ${j.metrics ? `
                <button class="btn btn-secondary" onclick="showEvalResults('${escapeHtml(j.job_name)}')">
                    <i class="fas fa-chart-bar"></i> View Results
                </button>` : ''}
            </div>
        </div>
    `).join('');
}

function renderInlineMetrics(m, taskType) {
    if (!m) return '';
    if (taskType === 'segmentation') {
        return `
        <div class="result-highlight" style="margin-top:0.5rem;">
            <div class="result-row"><span>Mean Dice</span><strong>${m.mean_dice ?? 'N/A'}</strong></div>
            <div class="result-row"><span>Mean IoU</span><strong>${m.mean_iou ?? 'N/A'}</strong></div>
            <div class="result-row"><span>Subjects</span><strong>${m.num_subjects ?? 'N/A'}</strong></div>
        </div>`;
    }
    return `
    <div class="result-highlight" style="margin-top:0.5rem;">
        <div class="result-row"><span>Accuracy</span><strong>${m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : 'N/A'}</strong></div>
        <div class="result-row"><span>Macro AUC</span><strong>${m.macro_auc ?? 'N/A'}</strong></div>
        <div class="result-row"><span>Samples</span><strong>${m.num_samples ?? 'N/A'}</strong></div>
    </div>`;
}

async function deleteEvalJob(jobName, btn) {
    if (!confirm(`Delete job "${jobName}" from tracking?\n\nThis removes the local record and cached files. S3 outputs are not affected.`)) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; }
    try {
        await apiCall(`/api/evaluation/jobs/${jobName}`, { method: 'DELETE' });
        loadEvalJobs();
    } catch (err) {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-times-circle"></i> Delete'; }
        alert(`Error: ${err.message}`);
    }
}

async function clearEvalCache(jobName, btn) {
    if (!confirm(`Remove locally cached files for "${jobName}"? The results remain in S3.`)) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...'; }
    try {
        await apiCall(`/api/evaluation/jobs/${jobName}/cache`, { method: 'DELETE' });
        loadEvalJobs();
    } catch (err) {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-trash"></i> Clear Cache'; }
        alert(`Error: ${err.message}`);
    }
}

async function downloadEvalResults(jobName, btn) {
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...'; }
    try {
        await apiCall(`/api/evaluation/jobs/${jobName}/download-results`, { method: 'POST' });
        if (btn) { btn.innerHTML = '<i class="fas fa-check"></i> Downloaded'; }
        loadEvalJobs();
    } catch (err) {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-download"></i> Download Results'; }
        alert(`Download failed: ${err.message}`);
    }
}

async function refreshEvalJob(jobName) {
    try {
        await apiCall(`/api/evaluation/jobs/${jobName}`);
        loadEvalJobs();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

function showEvalResults(jobName) {
    apiCall(`/api/evaluation/jobs/${jobName}`).then(job => {
        const panel = document.getElementById('eval-results-panel');
        const title = document.getElementById('eval-results-title');
        if (!panel || !job.metrics) return;

        title.textContent = `Results — ${jobName}`;
        panel.style.display = 'block';
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

        renderMetricCards(job.metrics, job.task_type);
        renderPerClassTable(job.metrics);
        renderOutputLinks(job.output_s3_uri, job.task_type);
        loadSubgroupMetrics(jobName);

        // If already cached, show plots immediately
        if (job.cached_results && Object.keys(job.cached_results).length > 0) {
            loadEvalPlots(jobName);
            return;
        }

        // Otherwise kick off plots-only download first (fast), render, then full download in background
        const grid = document.getElementById('eval-plots-grid');
        const wrap = document.getElementById('eval-plots-wrap');
        if (wrap) wrap.style.display = 'block';
        if (grid) grid.innerHTML = '<p style="color:#888;"><i class="fas fa-spinner fa-spin"></i> Fetching plots...</p>';

        apiCall(`/api/evaluation/jobs/${jobName}/download-results?plots_only=true`, { method: 'POST' })
            .then(res => {
                const hasPlots = res.cached && Object.keys(res.cached).some(f => f.endsWith('.png'));
                if (!hasPlots) {
                    if (grid) grid.innerHTML = `
                        <p style="color:#888;">
                            <i class="fas fa-exclamation-circle"></i>
                            No plot files found at <code>${escapeHtml(job.output_s3_uri)}</code>.<br>
                            The output S3 bucket must be writable by the Processing Job.
                        </p>`;
                    return;
                }
                loadEvalPlots(jobName);
                // Full download (CSV, JSON) in background
                apiCall(`/api/evaluation/jobs/${jobName}/download-results`, { method: 'POST' })
                    .then(() => loadEvalJobs())
                    .catch(() => {});
            })
            .catch(err => {
                if (grid) grid.innerHTML = `<p style="color:#c62828;">Download failed: ${escapeHtml(err.message)}</p>`;
            });

    }).catch(err => alert(`Error: ${err.message}`));
}

function renderMetricCards(m, taskType) {
    const container = document.getElementById('eval-metrics-cards');
    if (!container) return;

    let cards = [];
    if (taskType === 'segmentation') {
        cards = [
            { label: 'Mean Dice',   value: m.mean_dice   != null ? m.mean_dice.toFixed(4)   : 'N/A' },
            { label: 'Std Dice',    value: m.std_dice    != null ? m.std_dice.toFixed(4)    : 'N/A' },
            { label: 'Median Dice', value: m.median_dice != null ? m.median_dice.toFixed(4) : 'N/A' },
            { label: 'Mean IoU',    value: m.mean_iou    != null ? m.mean_iou.toFixed(4)    : 'N/A' },
            { label: 'Subjects',    value: m.num_subjects ?? 'N/A' },
        ];
    } else {
        cards = [
            { label: 'Accuracy',   value: m.accuracy  != null ? (m.accuracy * 100).toFixed(1) + '%' : 'N/A' },
            { label: 'Macro AUC',  value: m.macro_auc != null ? m.macro_auc.toFixed(4) : 'N/A' },
            { label: 'Samples',    value: m.num_samples ?? 'N/A' },
            { label: 'Classes',    value: (m.class_names || []).length || 'N/A' },
        ];
    }

    container.innerHTML = cards.map(c => `
        <div class="eval-metric-card">
            <div class="eval-metric-value">${escapeHtml(String(c.value))}</div>
            <div class="eval-metric-label">${escapeHtml(c.label)}</div>
        </div>
    `).join('');
}

function renderPerClassTable(m) {
    const wrap = document.getElementById('eval-per-class-wrap');
    const body = document.getElementById('eval-per-class-body');
    if (!wrap || !body || !m.per_class) { if (wrap) wrap.style.display = 'none'; return; }

    wrap.style.display = 'block';
    body.innerHTML = Object.entries(m.per_class).map(([cls, v]) => `
        <tr style="border-bottom:1px solid #f0f0f0;">
            <td style="padding:0.45rem 0.75rem;">${escapeHtml(cls)}</td>
            <td style="padding:0.45rem 0.75rem;">${v.precision?.toFixed(3) ?? 'N/A'}</td>
            <td style="padding:0.45rem 0.75rem;">${v.recall?.toFixed(3) ?? 'N/A'}</td>
            <td style="padding:0.45rem 0.75rem;">${v.f1?.toFixed(3) ?? 'N/A'}</td>
            <td style="padding:0.45rem 0.75rem;">${v.support ?? 'N/A'}</td>
        </tr>
    `).join('');
}

async function loadSubgroupMetrics(jobName) {
    const wrap   = document.getElementById('eval-subgroup-wrap');
    const tables = document.getElementById('eval-subgroup-tables');
    if (!wrap || !tables) return;

    try {
        // Fetch subgroup_metrics.json via a presigned URL from the plots endpoint
        // We piggyback on the job record's output_s3_uri
        const job = await apiCall(`/api/evaluation/jobs/${jobName}`);
        if (!job.output_s3_uri) return;

        const res = await fetch(`/api/evaluation/jobs/${jobName}/subgroup-metrics`);
        if (!res.ok) return;   // silently skip if not available
        const subgroups = await res.json();

        const attrLabels = { sex: 'Sex', age_bucket: 'Age Group', ethnicity: 'Ethnicity' };
        const attrs = Object.keys(subgroups).filter(a => Object.keys(subgroups[a]).length > 0);
        if (!attrs.length) return;

        wrap.style.display = 'block';
        tables.innerHTML = attrs.map(attr => {
            const groups = subgroups[attr];
            const rows   = Object.entries(groups)
                .sort((a, b) => b[1].accuracy - a[1].accuracy)
                .map(([grp, v]) => `
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:0.45rem 0.75rem;">${escapeHtml(grp)}</td>
                        <td style="padding:0.45rem 0.75rem;">${(v.accuracy * 100).toFixed(1)}%</td>
                        <td style="padding:0.45rem 0.75rem;">${v.macro_f1.toFixed(3)}</td>
                        <td style="padding:0.45rem 0.75rem;">${v.n}</td>
                    </tr>`).join('');
            return `
                <div style="margin-bottom:1.25rem;">
                    <h5 style="margin:0 0 0.5rem; color:#444;">${escapeHtml(attrLabels[attr] || attr)}</h5>
                    <div style="overflow-x:auto;">
                        <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                            <thead>
                                <tr style="background:#f5f5f5; text-align:left;">
                                    <th style="padding:0.45rem 0.75rem; border-bottom:2px solid #e0e0e0;">Group</th>
                                    <th style="padding:0.45rem 0.75rem; border-bottom:2px solid #e0e0e0;">Accuracy</th>
                                    <th style="padding:0.45rem 0.75rem; border-bottom:2px solid #e0e0e0;">Macro F1</th>
                                    <th style="padding:0.45rem 0.75rem; border-bottom:2px solid #e0e0e0;">N</th>
                                </tr>
                            </thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                </div>`;
        }).join('');
    } catch (_) {
        // Subgroup data is optional — fail silently
    }
}

function renderOutputLinks(outputS3Uri, taskType) {
    const el = document.getElementById('eval-output-links');
    if (!el || !outputS3Uri) return;

    const prefix = outputS3Uri.replace(/\/$/, '');
    const files = taskType === 'segmentation'
        ? ['metrics.json', 'predictions.csv', 'dice_distribution.png']
        : ['metrics.json', 'subgroup_metrics.json', 'predictions.csv', 'confusion_matrix.png', 'roc_auc.png', 'subgroup_accuracy.png'];

    el.innerHTML = `
        <strong>Output files at:</strong> <code>${escapeHtml(prefix)}/</code>
        <ul style="margin:0.5rem 0 0; padding-left:1.25rem;">
            ${files.map(f => `<li><code>${escapeHtml(f)}</code></li>`).join('')}
        </ul>
    `;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function loadEvalPlots(jobName) {
    const wrap = document.getElementById('eval-plots-wrap');
    const grid = document.getElementById('eval-plots-grid');
    if (!wrap || !grid) return;

    grid.innerHTML = '<p style="color:#888;"><i class="fas fa-spinner fa-spin"></i> Loading plots...</p>';
    wrap.style.display = 'block';

    try {
        const res = await apiCall(`/api/evaluation/jobs/${jobName}/plots`);
        const plots = res.plots || {};

        if (!Object.keys(plots).length) {
            grid.innerHTML = `<p style="color:#888;">No plots in cache yet — click "Download Results" to fetch from S3.</p>`;
            return;
        }

        const labels = {
            'confusion_matrix.png':   'Confusion Matrix',
            'roc_auc.png':            'ROC-AUC Curves',
            'subgroup_accuracy.png':  'Accuracy by Demographic Group',
            'dice_distribution.png':  'Dice Score Distribution',
        };

        grid.innerHTML = Object.entries(plots).map(([fname, url]) => {
            // Validate URL scheme to prevent javascript: URI injection
            const safeUrl = (url && (url.startsWith('/') || url.startsWith('http://') || url.startsWith('https://'))) ? url : '';
            return `
            <div style="background:#fff; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden;">
                <div style="padding:0.6rem 1rem; background:#f9f9f9; border-bottom:1px solid #e0e0e0; font-weight:600; font-size:0.9rem;">
                    ${escapeHtml(labels[fname] || fname)}
                </div>
                <div style="padding:1rem; text-align:center;">
                    <img src="${escapeHtml(safeUrl)}" alt="${escapeHtml(fname)}"
                         style="max-width:100%; height:auto; border-radius:4px; cursor:pointer;"
                         onclick="openPlotLightbox('${escapeHtml(safeUrl)}', '${escapeHtml(labels[fname] || fname)}')"
                         onerror="this.parentElement.innerHTML='<p style=color:#c62828>Image file missing from local cache — try re-downloading</p>'">
                </div>
                <div style="padding:0.4rem 1rem 0.75rem; text-align:right;">
                    <a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary" style="font-size:0.8rem; padding:0.3rem 0.75rem;">
                        <i class="fas fa-external-link-alt"></i> Open full size
                    </a>
                </div>
            </div>
        `}).join('');
    } catch (err) {
        grid.innerHTML = `<p style="color:#c62828;">Error loading plots: ${escapeHtml(err.message)}</p>`;
    }
}

function openPlotLightbox(url, title) {
    // Simple lightbox — create overlay on the fly
    const existing = document.getElementById('plot-lightbox');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'plot-lightbox';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:2000;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:zoom-out;';
    overlay.onclick = () => overlay.remove();
    overlay.innerHTML = `
        <div style="color:#fff; font-size:1rem; font-weight:600; margin-bottom:0.75rem;">${escapeHtml(title)}</div>
        <img src="${url}" style="max-width:90vw; max-height:85vh; border-radius:6px; box-shadow:0 8px 40px rgba(0,0,0,0.6);">
        <div style="color:#aaa; font-size:0.8rem; margin-top:0.6rem;">Click anywhere to close</div>
    `;
    document.body.appendChild(overlay);
}

function showBanner(el, type, html) {
    if (!el) return;
    const cls = { submitting: 'sm-status-submitting', success: 'sm-status-success', error: 'sm-status-error' };
    el.className = `sm-status-banner ${cls[type] || ''}`;
    el.innerHTML = html;
    el.style.display = 'flex';
}
