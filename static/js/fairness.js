// Fairness / Demographic Analysis

const FAIRNESS_COLORS = [
    '#4B9CD3', '#FF9900', '#16A085', '#E74C3C',
    '#8E44AD', '#2ECC71', '#F39C12', '#1ABC9C',
    '#E67E22', '#3498DB',
];

let _fairnessCharts = {};

registerTabInit('fairness', () => {
    const form = document.getElementById('fairness-form');
    if (form) form.addEventListener('submit', runFairnessScan);
});

async function runFairnessScan(e) {
    e.preventDefault();
    const s3Uri   = document.getElementById('fairness-s3-uri').value.trim();
    const maxFiles = parseInt(document.getElementById('fairness-max-files').value || '200');
    const btn     = e.target.querySelector('button[type="submit"]');
    const status  = document.getElementById('fairness-status');
    const results = document.getElementById('fairness-results');

    if (!s3Uri) return;

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
    status.style.display = 'flex';
    status.className = 'sm-status-banner sm-status-submitting';
    status.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Scanning DICOM files in <code>${s3Uri}</code>...`;
    results.style.display = 'none';

    // Destroy old charts
    Object.values(_fairnessCharts).forEach(c => c.destroy());
    _fairnessCharts = {};

    try {
        const data = await apiCall('/api/fairness/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ s3_uri: s3Uri, max_files: maxFiles }),
        });

        status.className = 'sm-status-banner sm-status-success';
        status.innerHTML = `<i class="fas fa-check-circle"></i> Scanned <strong>${data.total_scanned}</strong> of <strong>${data.total_found}</strong> DICOM files${data.errors ? ` (${data.errors} unreadable)` : ''}`;

        renderFairnessResults(data);
        results.style.display = 'block';
    } catch (err) {
        status.className = 'sm-status-banner sm-status-error';
        status.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-search"></i> Scan Demographics';
    }
}

function renderFairnessResults(data) {
    renderSummaryCards(data);
    renderAttributeChart('fairness-sex-chart',       data.attributes.PatientSex,      'Sex Distribution');
    renderAttributeChart('fairness-ethnicity-chart', data.attributes.EthnicGroup,     'Ethnic Group Distribution');
    renderAttributeChart('fairness-age-chart',       data.age_distribution,           'Age Group Distribution', true);
    renderAttributeChart('fairness-modality-chart',  data.attributes.Modality,        'Modality');
    renderMissingRates(data.missing_rates, data.total_scanned);
    renderSampleTable(data.files_sample);
}

function renderSummaryCards(data) {
    const container = document.getElementById('fairness-summary-cards');
    if (!container) return;
    const attrs = data.attributes;

    const sexCounts    = Object.entries(attrs.PatientSex    || {}).filter(([k]) => k !== 'Unknown');
    const ethCounts    = Object.entries(attrs.EthnicGroup   || {}).filter(([k]) => k !== 'Unknown');
    const ageCounts    = Object.entries(data.age_distribution || {}).filter(([k]) => k !== 'Unknown');

    container.innerHTML = `
        <div class="fairness-card">
            <div class="fairness-card-value">${data.total_scanned}</div>
            <div class="fairness-card-label">Files Scanned</div>
        </div>
        <div class="fairness-card">
            <div class="fairness-card-value">${sexCounts.length}</div>
            <div class="fairness-card-label">Sex Categories</div>
        </div>
        <div class="fairness-card">
            <div class="fairness-card-value">${ethCounts.length}</div>
            <div class="fairness-card-label">Ethnic Groups</div>
        </div>
        <div class="fairness-card">
            <div class="fairness-card-value">${ageCounts.length}</div>
            <div class="fairness-card-label">Age Groups</div>
        </div>
    `;
}

function renderAttributeChart(canvasId, counts, title, sortByKey = false) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !counts) return;

    let entries = Object.entries(counts);
    if (sortByKey) {
        const order = ['0-17','18-29','30-44','45-59','60-74','75+','Unknown'];
        entries.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
    } else {
        entries.sort((a, b) => b[1] - a[1]);
    }

    const labels = entries.map(([k]) => k || 'Unknown');
    const values = entries.map(([, v]) => v);
    const total  = values.reduce((s, v) => s + v, 0);

    _fairnessCharts[canvasId] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Count',
                data: values,
                backgroundColor: labels.map((_, i) => FAIRNESS_COLORS[i % FAIRNESS_COLORS.length] + 'cc'),
                borderColor:     labels.map((_, i) => FAIRNESS_COLORS[i % FAIRNESS_COLORS.length]),
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: title, font: { size: 13 } },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const pct = total ? ((ctx.raw / total) * 100).toFixed(1) : 0;
                            return ` ${ctx.raw} (${pct}%)`;
                        }
                    }
                }
            },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Count' } },
            },
        },
    });
}

function renderMissingRates(missingRates, total) {
    const container = document.getElementById('fairness-missing-rates');
    if (!container) return;
    const rows = Object.entries(missingRates)
        .filter(([k]) => !['Modality','BodyPartExamined','PatientRace'].includes(k))
        .map(([attr, pct]) => `
            <tr>
                <td>${attr}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <div style="flex:1; background:#eee; border-radius:4px; height:8px;">
                            <div style="width:${pct}%; background:${pct > 50 ? '#E74C3C' : pct > 20 ? '#FF9900' : '#2ECC71'}; height:8px; border-radius:4px;"></div>
                        </div>
                        <span style="font-size:0.82rem; min-width:40px;">${pct}%</span>
                    </div>
                </td>
            </tr>`).join('');

    container.innerHTML = `
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
            <thead><tr style="border-bottom:1px solid #eee;">
                <th style="text-align:left; padding:4px 8px;">Attribute</th>
                <th style="text-align:left; padding:4px 8px;">Missing Rate</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function renderSampleTable(sample) {
    const container = document.getElementById('fairness-sample-table');
    if (!container || !sample?.length) return;

    const cols = ['key', 'PatientSex', 'AgeBucket', 'EthnicGroup', 'Modality'];
    const headers = cols.map(c => `<th style="padding:4px 8px; text-align:left; font-size:0.8rem;">${c}</th>`).join('');
    const rows = sample.map(row =>
        `<tr style="border-bottom:1px solid #f5f5f5;">${
            cols.map(c => `<td style="padding:4px 8px; font-size:0.78rem; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${row[c] || ''}">${row[c] || '—'}</td>`).join('')
        }</tr>`
    ).join('');

    container.innerHTML = `
        <div style="overflow-x:auto; max-height:300px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse;">
                <thead style="position:sticky; top:0; background:#f9f9f9;"><tr>${headers}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}
