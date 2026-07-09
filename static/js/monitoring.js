// Model Monitoring JavaScript

registerTabInit('monitoring', function () {
    loadMonitorEndpoints();
    loadCvEndpoints();
    loadMonitorSchedules();
    loadBaselineJobs();
    refreshDQBaselinePicker();

    document.getElementById('data-capture-form')?.addEventListener('submit', enableDataCapture);
    document.getElementById('baseline-form')?.addEventListener('submit', runBaseline);
    document.getElementById('dq-schedule-form')?.addEventListener('submit', createDQSchedule);
    document.getElementById('mq-schedule-form')?.addEventListener('submit', createMQSchedule);

    setInterval(loadMonitorSchedules, 30000);
    setInterval(loadBaselineJobs, 20000);
});

// ── Capture Viewer ────────────────────────────────────────────────────────────

let _cvClassChart = null;
let _cvConfChart  = null;

async function loadCaptureViewer() {
    const select       = document.getElementById('cv-endpoint-select');
    const endpointName = select?.value?.trim();
    const captureS3    = document.getElementById('cv-capture-s3')?.value?.trim();
    const maxFiles     = document.getElementById('cv-max-files')?.value || '20';
    const banner       = document.getElementById('cv-status-banner');
    const results      = document.getElementById('cv-results');

    if (!endpointName || !captureS3) {
        showMonitorBanner(banner, 'error', '<i class="fas fa-exclamation-circle"></i> Select an endpoint and enter a capture S3 URI');
        return;
    }

    showMonitorBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Reading capture files from S3...');
    if (results) results.style.display = 'none';

    try {
        const data = await apiCall(
            `/api/monitoring/capture-summary?endpoint_name=${encodeURIComponent(endpointName)}&capture_s3_uri=${encodeURIComponent(captureS3)}&max_files=${maxFiles}`
        );

        if (!data.total_records) {
            showMonitorBanner(banner, 'error', `<i class="fas fa-info-circle"></i> ${escapeHtml(data.message || 'No capture records found')}`);
            return;
        }

        showMonitorBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Analysed <strong>${data.parsed_records}</strong> records from <strong>${data.files_scanned}</strong> files`
            + (data.errors ? ` (${data.errors} parse errors)` : ''));

        renderCaptureViewer(data);
        if (results) results.style.display = 'block';
    } catch (err) {
        showMonitorBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    }
}

function renderCaptureViewer(data) {
    // Summary cards
    const cards = document.getElementById('cv-summary-cards');
    if (cards) {
        cards.innerHTML = `
            <div class="eval-metric-card"><div class="eval-metric-value">${data.parsed_records}</div><div class="eval-metric-label">Records</div></div>
            <div class="eval-metric-card"><div class="eval-metric-value">${data.files_scanned}</div><div class="eval-metric-label">Files Scanned</div></div>
            <div class="eval-metric-card"><div class="eval-metric-value">${(data.avg_confidence * 100).toFixed(1)}%</div><div class="eval-metric-label">Avg Confidence</div></div>
            <div class="eval-metric-card"><div class="eval-metric-value">${Object.keys(data.class_distribution || {}).length}</div><div class="eval-metric-label">Classes Seen</div></div>
        `;
    }

    // Class distribution bar chart
    const classCtx = document.getElementById('cv-class-chart');
    if (classCtx && data.class_distribution) {
        if (_cvClassChart) _cvClassChart.destroy();
        const labels = Object.keys(data.class_distribution).sort((a, b) => data.class_distribution[b] - data.class_distribution[a]);
        const values = labels.map(l => data.class_distribution[l]);
        const total  = values.reduce((s, v) => s + v, 0);
        _cvClassChart = new Chart(classCtx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Predictions',
                    data: values,
                    backgroundColor: labels.map((_, i) => `hsl(${i * 45}, 65%, 55%)cc`),
                    borderColor:     labels.map((_, i) => `hsl(${i * 45}, 65%, 45%)`),
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${ctx.raw} (${((ctx.raw / total) * 100).toFixed(1)}%)`
                        }
                    }
                },
                scales: {
                    x: { ticks: { maxRotation: 30, font: { size: 10 } } },
                    y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                },
            },
        });
    }

    // Confidence histogram
    const confCtx = document.getElementById('cv-conf-chart');
    if (confCtx && data.confidence_histogram) {
        if (_cvConfChart) _cvConfChart.destroy();
        const hist = data.confidence_histogram;
        _cvConfChart = new Chart(confCtx, {
            type: 'bar',
            data: {
                labels: hist.buckets,
                datasets: [{
                    label: 'Count',
                    data: hist.counts,
                    backgroundColor: '#4B9CD3cc',
                    borderColor: '#4B9CD3',
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Confidence Range' }, ticks: { font: { size: 10 } } },
                    y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                },
            },
        });
    }

    // Recent records table
    const table = document.getElementById('cv-records-table');
    if (table && data.recent_records?.length) {
        const rows = data.recent_records.map(r => `
            <tr style="border-bottom:1px solid #f0f0f0;">
                <td style="padding:0.4rem 0.75rem; font-size:0.78rem; color:#888;">${r.time ? escapeHtml(new Date(r.time).toLocaleString()) : '—'}</td>
                <td style="padding:0.4rem 0.75rem; font-size:0.82rem;"><strong>${escapeHtml(String(r.predicted_class))}</strong></td>
                <td style="padding:0.4rem 0.75rem; font-size:0.82rem;">${(r.confidence * 100).toFixed(1)}%</td>
            </tr>`).join('');
        table.innerHTML = `
            <div style="overflow-x:auto; max-height:300px; overflow-y:auto;">
                <table style="width:100%; border-collapse:collapse;">
                    <thead style="position:sticky; top:0; background:#f9f9f9;">
                        <tr>
                            <th style="padding:0.4rem 0.75rem; text-align:left; font-size:0.8rem;">Time</th>
                            <th style="padding:0.4rem 0.75rem; text-align:left; font-size:0.8rem;">Predicted Class</th>
                            <th style="padding:0.4rem 0.75rem; text-align:left; font-size:0.8rem;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }
}

// ── Endpoint loader ───────────────────────────────────────────────────────────

async function loadMonitorEndpoints() {
    const select = document.getElementById('dc-endpoint');
    if (!select) return;
    try {
        const dep    = await apiCall('/api/deployment/deployments?local_only=true');
        const active = (dep.deployments || []).filter(d => d.status === 'InService');
        select.innerHTML = active.length
            ? active.map(d => `<option value="${escapeHtml(d.endpoint_name)}" data-capture="${escapeHtml(d.capture_s3_uri || '')}">${escapeHtml(d.endpoint_name)}${d.data_capture_enabled ? ' 📡' : ''}</option>`).join('')
            : '<option value="" disabled selected>No InService endpoints</option>';

        // Auto-fill capture viewer if any endpoint has capture enabled
        const withCapture = active.find(d => d.data_capture_enabled && d.capture_s3_uri);
        if (withCapture) {
            const cvEndpoint = document.getElementById('cv-endpoint');
            const cvS3       = document.getElementById('cv-capture-s3');
            if (cvEndpoint && !cvEndpoint.value) cvEndpoint.value = withCapture.endpoint_name;
            if (cvS3 && !cvS3.value)             cvS3.value       = withCapture.capture_s3_uri;
        }

        // Also populate the capture viewer dropdown
        loadCvEndpoints(active);
    } catch (_) {
        select.innerHTML = '<option value="" disabled selected>Could not load endpoints</option>';
    }
}

async function loadCvEndpoints(preloaded) {
    const select = document.getElementById('cv-endpoint-select');
    if (!select) return;
    try {
        const deployments = preloaded || (await apiCall('/api/deployment/deployments?local_only=true')).deployments || [];
        const all = deployments.filter(d => ['InService', 'Creating', 'Updating'].includes(d.status));
        if (!all.length) {
            select.innerHTML = '<option value="" disabled selected>No endpoints found</option>';
            return;
        }
        select.innerHTML = '<option value="" disabled selected>Select an endpoint…</option>'
            + all.map(d => {
                const captureFlag = d.data_capture_enabled ? ' 📡' : '';
                const captureUri  = d.capture_s3_uri || '';
                return `<option value="${escapeHtml(d.endpoint_name)}" data-capture="${escapeHtml(captureUri)}" data-status="${escapeHtml(d.status)}">`
                    + `${escapeHtml(d.endpoint_name)}${captureFlag} (${escapeHtml(d.status)})</option>`;
            }).join('');

        // Auto-select first endpoint with capture enabled
        const withCapture = all.find(d => d.data_capture_enabled && d.capture_s3_uri);
        if (withCapture) {
            select.value = withCapture.endpoint_name;
            onCvEndpointSelect();
        }
    } catch (_) {
        select.innerHTML = '<option value="" disabled selected>Could not load endpoints</option>';
    }
}

function onCvEndpointSelect() {
    const select = document.getElementById('cv-endpoint-select');
    const s3Input = document.getElementById('cv-capture-s3');
    if (!select || !s3Input) return;
    const selected = select.options[select.selectedIndex];
    const captureUri = selected?.dataset?.capture || '';
    if (captureUri) s3Input.value = captureUri;
}

// ── Data Capture ──────────────────────────────────────────────────────────────

async function enableDataCapture(e) {
    e.preventDefault();
    const endpointVal = document.getElementById('dc-endpoint-manual')?.value?.trim()
        || document.getElementById('dc-endpoint')?.value;
    const banner = document.getElementById('dc-status-banner');
    const btn    = e.target.querySelector('button[type="submit"]');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enabling...';
    showMonitorBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Enabling data capture...');

    try {
        const result = await apiCall('/api/monitoring/data-capture', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                endpoint_name:       endpointVal,
                capture_s3_uri:      document.getElementById('dc-capture-s3').value,
                sampling_percentage: parseInt(document.getElementById('dc-sampling').value || '100'),
            }),
        });
        showMonitorBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Data capture enabled on <strong>${result.endpoint_name}</strong> — endpoint updating`);
    } catch (err) {
        showMonitorBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-camera"></i> Enable Data Capture';
    }
}

// ── Baseline ──────────────────────────────────────────────────────────────────

async function runBaseline(e) {
    e.preventDefault();
    const banner = document.getElementById('bl-status-banner');
    const btn    = e.target.querySelector('button[type="submit"]');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    showMonitorBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Submitting baseline job...');

    try {
        const result = await apiCall('/api/monitoring/baseline/data-quality', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                baseline_data_s3_uri: document.getElementById('bl-data-s3').value,
                output_s3_uri:        document.getElementById('bl-output-s3').value,
                role_arn:             document.getElementById('bl-role-arn').value,
                instance_type:        document.getElementById('bl-instance-type').value,
            }),
        });
        showMonitorBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Baseline job <strong>${result.job_name}</strong> submitted`);
        loadBaselineJobs();
        refreshDQBaselinePicker();
    } catch (err) {
        showMonitorBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Run Baseline Job';
    }
}

// ── Schedules ─────────────────────────────────────────────────────────────────

async function createDQSchedule(e) {
    e.preventDefault();
    const banner = document.getElementById('dq-status-banner');
    const btn    = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
    showMonitorBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Creating schedule...');

    try {
        const result = await apiCall('/api/monitoring/schedules/data-quality', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                schedule_name:               document.getElementById('dq-schedule-name').value,
                endpoint_name:               document.getElementById('dq-endpoint').value,
                baseline_constraints_s3_uri: document.getElementById('dq-constraints-s3').value,
                baseline_statistics_s3_uri:  document.getElementById('dq-statistics-s3').value,
                output_s3_uri:               document.getElementById('dq-output-s3').value,
                role_arn:                    document.getElementById('dq-role-arn').value,
                schedule_expression:         document.getElementById('dq-schedule-expr').value,
            }),
        });
        showMonitorBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Schedule <strong>${result.schedule_name}</strong> created`);
        loadMonitorSchedules();
    } catch (err) {
        showMonitorBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-clock"></i> Create Schedule';
    }
}

async function createMQSchedule(e) {
    e.preventDefault();
    const banner = document.getElementById('mq-status-banner');
    const btn    = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
    showMonitorBanner(banner, 'submitting', '<i class="fas fa-spinner fa-spin"></i> Creating schedule...');

    try {
        const result = await apiCall('/api/monitoring/schedules/model-quality', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                schedule_name:         document.getElementById('mq-schedule-name').value,
                endpoint_name:         document.getElementById('mq-endpoint').value,
                ground_truth_s3_uri:   document.getElementById('mq-gt-s3').value,
                output_s3_uri:         document.getElementById('mq-output-s3').value,
                role_arn:              document.getElementById('mq-role-arn').value,
                problem_type:          document.getElementById('mq-problem-type').value,
                inference_attribute:   document.getElementById('mq-inference-attr').value,
                schedule_expression:   document.getElementById('mq-schedule-expr').value,
            }),
        });
        showMonitorBanner(banner, 'success',
            `<i class="fas fa-check-circle"></i> Schedule <strong>${result.schedule_name}</strong> created`);
        loadMonitorSchedules();
    } catch (err) {
        showMonitorBanner(banner, 'error', `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-clock"></i> Create Schedule';
    }
}

// ── Schedule list ─────────────────────────────────────────────────────────────

async function loadMonitorSchedules() {
    const container = document.getElementById('monitor-schedules-list');
    if (!container) return;
    try {
        const res = await apiCall('/api/monitoring/schedules?local_only=true');
        // Only show schedules (not baselines) in this list
        renderMonitorSchedules((res.schedules || []).filter(s => s.type === 'schedule'));
    } catch (err) {
        container.innerHTML = `<p style="color:#c62828;">Error: ${escapeHtml(err.message)}</p>`;
    }
}

async function loadBaselineJobs() {
    const container = document.getElementById('monitor-baseline-list');
    if (!container) return;
    try {
        const res = await apiCall('/api/monitoring/baselines');
        renderBaselineJobs(res.baselines || []);
    } catch (err) {
        container.innerHTML = `<p style="color:#c62828;">Error: ${escapeHtml(err.message)}</p>`;
    }
}

function renderBaselineJobs(baselines) {
    const container = document.getElementById('monitor-baseline-list');
    if (!baselines.length) {
        container.innerHTML = '<p style="color:#888;">No baseline jobs yet. Run a baseline job above to compute statistics and constraints.</p>';
        return;
    }

    const statusClass = {
        'InProgress': 'status-running',
        'Completed':  'status-completed',
        'Failed':     'status-failed',
        'Stopping':   'status-stopping',
        'Stopped':    'status-stopped',
    };

    container.innerHTML = baselines.map(b => `
        <div class="job-item">
            <div class="job-header">
                <div class="job-id">${escapeHtml(b.job_name)}</div>
                <div class="job-status ${statusClass[b.status] || 'status-unknown'}">${escapeHtml(b.status)}</div>
            </div>
            <div class="job-details">
                <div><strong>Type:</strong> Data Quality Baseline</div>
                <div><strong>Instance:</strong> ${escapeHtml(b.instance_type || 'N/A')}</div>
                <div><strong>Baseline data:</strong> <code style="font-size:0.78rem;">${escapeHtml(b.baseline_data_s3_uri || 'N/A')}</code></div>
                <div><strong>Output:</strong> <code style="font-size:0.78rem;">${escapeHtml(b.output_s3_uri || 'N/A')}</code></div>
                ${b.created_at ? `<div><strong>Submitted:</strong> ${escapeHtml(new Date(b.created_at).toLocaleString())}</div>` : ''}
                ${b.completed_at ? `<div><strong>Completed:</strong> ${escapeHtml(new Date(b.completed_at).toLocaleString())}</div>` : ''}
                ${b.failure_reason ? `<div style="color:#c62828; font-size:0.82rem;"><strong>Failure:</strong> ${escapeHtml(b.failure_reason)}</div>` : ''}
                ${b.status === 'Completed' ? `
                <div style="margin-top:0.5rem; font-size:0.82rem; color:#555;">
                    <strong>Use these URIs in your Data Quality Schedule:</strong><br>
                    Constraints: <code style="font-size:0.75rem;">${escapeHtml((b.output_s3_uri || '').replace(/\/$/, ''))}/constraints.json</code><br>
                    Statistics: <code style="font-size:0.75rem;">${escapeHtml((b.output_s3_uri || '').replace(/\/$/, ''))}/statistics.json</code>
                </div>` : ''}
            </div>
            <div class="job-actions">
                <button class="btn btn-secondary" onclick="deleteBaselineJob('${escapeHtml(b.job_name)}')">
                    <i class="fas fa-times"></i> Remove
                </button>
                ${b.status === 'Completed' ? `
                <button class="btn btn-secondary" onclick="fillScheduleFromBaseline('${escapeHtml(b.output_s3_uri)}')">
                    <i class="fas fa-arrow-right"></i> Use in Schedule
                </button>` : ''}
            </div>
        </div>
    `).join('');
}

async function deleteBaselineJob(jobName) {
    if (!confirm(`Remove baseline job "${jobName}" from tracking?`)) return;
    try {
        await apiCall(`/api/monitoring/baselines/${jobName}`, { method: 'DELETE' });
        loadBaselineJobs();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

function fillScheduleFromBaseline(outputS3Uri) {
    // Auto-fill the DQ schedule form with baseline output paths
    const base = outputS3Uri?.replace(/\/$/, '');
    const constraintsEl = document.getElementById('dq-constraints-s3');
    const statisticsEl  = document.getElementById('dq-statistics-s3');
    if (constraintsEl) constraintsEl.value = `${base}/constraints.json`;
    if (statisticsEl)  statisticsEl.value  = `${base}/statistics.json`;
    // Scroll to the DQ schedule form
    document.getElementById('dq-schedule-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── DQ baseline picker ────────────────────────────────────────────────────────

async function refreshDQBaselinePicker() {
    const select = document.getElementById('dq-baseline-picker');
    if (!select) return;
    try {
        const res      = await apiCall('/api/monitoring/baselines');
        const completed = (res.baselines || []).filter(b => b.status === 'Completed');
        if (!completed.length) {
            select.innerHTML = '<option value="" disabled selected>No completed baselines</option>';
            return;
        }
        select.innerHTML = '<option value="" disabled selected>Select a baseline…</option>'
            + completed.map(b => {
                const base = (b.output_s3_uri || '').replace(/\/$/, '');
                return `<option value="${escapeHtml(base)}" data-job="${escapeHtml(b.job_name)}" data-data="${escapeHtml(b.baseline_data_s3_uri || '')}">`
                    + `${escapeHtml(b.job_name)} — ${escapeHtml(b.baseline_data_s3_uri || 'unknown data')}</option>`;
            }).join('');
    } catch (_) {
        select.innerHTML = '<option value="" disabled selected>Could not load baselines</option>';
    }
}

function fillDQFromBaseline() {
    const select = document.getElementById('dq-baseline-picker');
    if (!select || !select.value) return;
    const base    = select.value;
    const opt     = select.options[select.selectedIndex];
    const dataUri = opt.dataset.data || '';

    // Auto-fill constraints, statistics, and suggest output URI
    const constraintsEl = document.getElementById('dq-constraints-s3');
    const statisticsEl  = document.getElementById('dq-statistics-s3');
    const outputEl      = document.getElementById('dq-output-s3');
    const nameEl        = document.getElementById('dq-schedule-name');

    if (constraintsEl) constraintsEl.value = `${base}/constraints.json`;
    if (statisticsEl)  statisticsEl.value  = `${base}/statistics.json`;

    // Suggest output URI alongside baseline output
    if (outputEl && !outputEl.value) {
        outputEl.value = base.replace(/\/[^/]+$/, '/monitor-output/');
    }

    // Suggest a schedule name from the baseline job name
    if (nameEl && !nameEl.value) {
        const jobName = opt.dataset.job || '';
        nameEl.value  = ('dq-schedule-' + jobName).slice(0, 63);
    }
}

function renderMonitorSchedules(schedules) {
    const container = document.getElementById('monitor-schedules-list');
    if (!schedules.length) {
        container.innerHTML = '<p style="color:#888;">No monitoring schedules yet.</p>';
        return;
    }

    const statusClass = {
        'Scheduled': 'status-running',
        'Pending':   'status-running',
        'Stopped':   'status-stopped',
        'Failed':    'status-failed',
        'InProgress':'status-running',
        'Completed': 'status-completed',
    };

    const typeLabel = {
        'data_quality':  '<span style="background:#e8f4fd; color:#1a6fa0; padding:2px 8px; border-radius:10px; font-size:0.78rem;">Data Quality</span>',
        'model_quality': '<span style="background:#fef0e7; color:#e67e22; padding:2px 8px; border-radius:10px; font-size:0.78rem;">Model Quality</span>',
        'baseline':      '<span style="background:#f4ecf7; color:#8e44ad; padding:2px 8px; border-radius:10px; font-size:0.78rem;">Baseline</span>',
    };

    container.innerHTML = schedules.map(s => {
        const name    = s.schedule_name || s.job_name;
        const status  = s.status || 'Unknown';
        const execStatus = s.last_execution_status;
        const execTime   = s.last_execution_time ? new Date(s.last_execution_time).toLocaleString() : null;

        return `
        <div class="job-item">
            <div class="job-header">
                <div class="job-id">${escapeHtml(name)} ${typeLabel[s.monitor_type] || ''}</div>
                <div class="job-status ${statusClass[status] || 'status-unknown'}">${escapeHtml(status)}</div>
            </div>
            <div class="job-details">
                ${s.endpoint_name ? `<div><strong>Endpoint:</strong> ${escapeHtml(s.endpoint_name)}</div>` : ''}
                ${s.schedule_expression ? `<div><strong>Schedule:</strong> ${escapeHtml(s.schedule_expression)}</div>` : ''}
                ${s.output_s3_uri ? `<div><strong>Output:</strong> <code style="font-size:0.78rem;">${escapeHtml(s.output_s3_uri)}</code></div>` : ''}
                ${s.created_at ? `<div><strong>Created:</strong> ${escapeHtml(new Date(s.created_at).toLocaleString())}</div>` : ''}
                ${execStatus ? `<div><strong>Last execution:</strong> <span class="${statusClass[execStatus] || ''}" style="font-size:0.82rem;">${escapeHtml(execStatus)}</span>${execTime ? ` at ${escapeHtml(execTime)}` : ''}</div>` : ''}
                ${s.last_failure_reason ? `<div style="color:#c62828; font-size:0.82rem;"><strong>Failure:</strong> ${escapeHtml(s.last_failure_reason)}</div>` : ''}
            </div>
            <div class="job-actions">
                ${s.type === 'schedule' ? `
                <button class="btn btn-secondary" onclick="refreshMonitorSchedule('${escapeHtml(name)}')">
                    <i class="fas fa-sync"></i> Refresh
                </button>
                <button class="btn btn-secondary" onclick="deleteMonitorSchedule('${escapeHtml(name)}')" style="color:#c62828;">
                    <i class="fas fa-trash"></i> Delete
                </button>` : ''}
            </div>
        </div>`;
    }).join('');
}

async function refreshMonitorSchedule(name) {
    try {
        await apiCall(`/api/monitoring/schedules/${name}`);
        loadMonitorSchedules();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function deleteMonitorSchedule(name) {
    if (!confirm(`Delete monitoring schedule "${name}"?`)) return;
    try {
        await apiCall(`/api/monitoring/schedules/${name}`, { method: 'DELETE' });
        loadMonitorSchedules();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

function showMonitorBanner(el, type, html) {
    if (!el) return;
    const cls = { submitting: 'sm-status-submitting', success: 'sm-status-success', error: 'sm-status-error' };
    el.className = `sm-status-banner ${cls[type] || ''}`;
    el.innerHTML = html;
    el.style.display = 'flex';
}
