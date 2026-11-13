// Preprocessing tab — MONAI pipeline builder + local preview + SageMaker batch job.
//
// One spec, two runners: the step list below is serialized to a pipeline spec
// and sent to /api/preprocessing/preview (local, one image) and, after a
// successful preview, /api/preprocessing/submit (batch over an S3 dataset).
//
// The backend routes (/api/preprocessing/*) are defined in the preprocessing
// spec (specs/preprocessing/). Until they are implemented this tab degrades
// gracefully: it falls back to a built-in catalog and shows a clear banner if
// an API call fails.

(function () {
    'use strict';

    // ── Fallback step catalog (used if /catalog is unavailable) ──────────────
    // Mirrors specs/preprocessing/design.md. Each entry: default params.
    const FALLBACK_CATALOG = {
        load:                 { label: 'Load image (DICOM/NIfTI)', params: { reader: 'auto' } },
        ensure_channel_first: { label: 'Ensure channel first',     params: {} },
        orientation:          { label: 'Orientation (RAS)',        params: { axcodes: 'RAS' } },
        resample:             { label: 'Resample spacing',         params: { pixdim: [1.5, 1.5, 2.0], mode: ['bilinear', 'nearest'] } },
        intensity_window:     { label: 'Intensity window (CT HU)', params: { a_min: -175, a_max: 250, b_min: 0.0, b_max: 1.0, clip: true } },
        scale_intensity:      { label: 'Scale intensity (min–max)', params: { minv: 0.0, maxv: 1.0 } },
        scale_intensity_percentiles: { label: 'Scale intensity (percentiles)', params: { lower: 1.0, upper: 99.0, b_min: 0.0, b_max: 1.0, clip: true } },
        clip_percentiles:     { label: 'Clip intensity (percentiles)', params: { lower: 1.0, upper: 99.0 } },
        threshold_intensity:  { label: 'Threshold intensity',      params: { threshold: 0.0, above: true, cval: 0.0 } },
        histogram_normalize:  { label: 'Histogram normalize',      params: { num_bins: 256, min: 0.0, max: 1.0 } },
        normalize:            { label: 'Normalize intensity',      params: {} },
        gaussian_smooth:      { label: 'Gaussian smooth',          params: { sigma: 1.0 } },
        median_smooth:        { label: 'Median smooth',            params: { radius: 1 } },
        crop_foreground:      { label: 'Crop to foreground',       params: { source_key: 'image' } },
        center_crop:          { label: 'Center spatial crop',      params: { roi_size: [128, 128, 128] } },
        spatial_pad:          { label: 'Spatial pad',              params: { spatial_size: [128, 128, 128], mode: 'constant' } },
        divisible_pad:        { label: 'Divisible pad',            params: { k: 16, mode: 'constant' } },
        resize:               { label: 'Resize',                   params: { spatial_size: [128, 128, 128], mode: ['area', 'nearest'] } },
        zoom:                 { label: 'Zoom',                     params: { zoom: 1.0, mode: ['area', 'nearest'] } },
        flip:                 { label: 'Flip',                     params: { spatial_axis: 0 } },
        rotate90:             { label: 'Rotate 90°',               params: { k: 1, spatial_axes: [0, 1] } },
        cast:                 { label: 'Cast dtype',               params: { dtype: 'float32' } },
        ensure_type:          { label: 'Ensure type (tensor)',     params: { data_type: 'tensor' } },
        save:                 { label: 'Save (.nii.gz)',           params: { output_postfix: 'proc', output_ext: '.nii.gz' } }
    };

    const DEFAULT_STEPS = ['load', 'ensure_channel_first', 'orientation', 'resample', 'intensity_window', 'crop_foreground', 'resize', 'save'];

    let catalog = FALLBACK_CATALOG;
    let steps = [];            // [{ type, params }]
    let previewOk = false;     // gates the submit button

    // ── Spec assembly ────────────────────────────────────────────────────────
    // Task type is derived from the dataset's dataset.json (via Resolve dataset),
    // not a manual selector. For the local preview it is key-agnostic (one image,
    // no label), so we default to segmentation keys which are a superset.
    let resolvedTaskType = null;   // set by ppResolveDataset()

    function taskType() {
        return resolvedTaskType || 'segmentation';
    }

    function buildSpec() {
        const tt = taskType();
        return {
            task_type: tt,
            keys: tt === 'segmentation' ? ['image', 'label'] : ['image'],
            steps: steps.map(s => ({ type: s.type, params: s.params })),
            dataset_prep: {
                enabled: !!document.getElementById('pp-dataset-prep')?.checked,
                split: { train: 0.7, val: 0.15, test: 0.15 },
                seed: 42
            }
        };
    }

    function refreshSpecView() {
        // Keep the JSON editor in sync only when the user isn't actively editing it.
        const ta = document.getElementById('pp-spec-editor');
        if (ta && document.activeElement !== ta) {
            ta.value = JSON.stringify(buildSpec(), null, 2);
        }
    }

    // Load a validated spec object into the builder state.
    function loadSpecIntoBuilder(spec) {
        if (spec && (spec.task_type === 'classification' || spec.task_type === 'segmentation')) {
            resolvedTaskType = spec.task_type;
        }
        if (spec && spec.dataset_prep && typeof spec.dataset_prep.enabled === 'boolean') {
            const dp = document.getElementById('pp-dataset-prep');
            if (dp) dp.checked = spec.dataset_prep.enabled;
        }
        steps = (spec.steps || []).map(s => ({ type: s.type, params: s.params || {} }));
        renderSteps();
    }

    // Validate a spec object against the backend, then load it. Falls back to a
    // client-side sanity check if the /validate endpoint is unavailable.
    async function validateAndLoad(spec, statusEl) {
        try {
            const res = await apiCall('/api/preprocessing/validate', {
                method: 'POST',
                body: JSON.stringify(spec)
            });
            loadSpecIntoBuilder(res.spec || spec);
            setImportStatus(statusEl, 'Spec validated and loaded.', 'success');
        } catch (err) {
            setImportStatus(statusEl, 'Invalid spec: ' + err.message, 'error');
        }
    }

    function setImportStatus(el, msg, type) {
        if (!el) return;
        el.style.display = 'block';
        el.textContent = msg;
        el.style.color = type === 'error' ? '#b71c1c' : (type === 'success' ? '#1b5e20' : '#555');
    }

    // ── JSON import (exposed on window for inline onclick / file input) ─────────
    window.ppImportFromEditor = function () {
        const ta = document.getElementById('pp-spec-editor');
        const status = document.getElementById('pp-import-status');
        let spec;
        try {
            spec = JSON.parse(ta.value || '{}');
        } catch (e) {
            setImportStatus(status, 'Not valid JSON: ' + e.message, 'error');
            return;
        }
        validateAndLoad(spec, status);
    };

    window.ppSyncEditorFromSteps = function () {
        const ta = document.getElementById('pp-spec-editor');
        if (ta) ta.value = JSON.stringify(buildSpec(), null, 2);
    };

    function handleImportFile(evt) {
        const status = document.getElementById('pp-import-status');
        const file = evt.target.files && evt.target.files[0];
        if (!file) return;
        if (file.size > 1024 * 1024) {   // 1 MB cap
            setImportStatus(status, 'File too large (max 1 MB).', 'error');
            evt.target.value = '';
            return;
        }
        const reader = new FileReader();
        reader.onload = function () {
            let spec;
            try {
                spec = JSON.parse(reader.result);
            } catch (e) {
                setImportStatus(status, 'Not valid JSON: ' + e.message, 'error');
                return;
            }
            const ta = document.getElementById('pp-spec-editor');
            if (ta) ta.value = JSON.stringify(spec, null, 2);
            validateAndLoad(spec, status);
        };
        reader.onerror = function () { setImportStatus(status, 'Could not read file.', 'error'); };
        reader.readAsText(file);
        evt.target.value = '';   // allow re-selecting the same file
    }

    // ── Step list rendering ────────────────────────────────────────────────────
    function renderStepPicker() {
        const sel = document.getElementById('pp-step-picker');
        if (!sel) return;
        sel.replaceChildren();
        Object.keys(catalog).forEach(type => {
            const opt = document.createElement('option');
            opt.value = type;
            opt.textContent = catalog[type].label || type;   // textContent = safe
            sel.appendChild(opt);
        });
    }

    function renderSteps() {
        const box = document.getElementById('pp-steps-list');
        if (!box) return;
        box.replaceChildren();
        if (!steps.length) {
            setHTML(box, html`<p style="color:#888; font-size:0.85rem; margin:0;">No steps yet — add one above, or click “Load Default”.</p>`);
            refreshSpecView();
            return;
        }
        steps.forEach((step, i) => {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex; align-items:center; gap:8px; padding:6px 8px; border-bottom:1px solid #eee;';

            const idx = document.createElement('span');
            idx.style.cssText = 'color:#888; font-size:0.8rem; width:20px;';
            idx.textContent = (i + 1) + '.';

            const name = document.createElement('span');
            name.style.cssText = 'flex:0 0 200px; font-weight:600; font-size:0.85rem;';
            name.textContent = (catalog[step.type] && catalog[step.type].label) || step.type;

            const params = document.createElement('input');
            params.type = 'text';
            params.style.cssText = 'flex:1; font-family:monospace; font-size:0.78rem;';
            params.value = JSON.stringify(step.params || {});
            params.addEventListener('change', () => {
                try {
                    step.params = JSON.parse(params.value || '{}');
                    params.style.borderColor = '';
                } catch (e) {
                    params.style.borderColor = '#d9534f';   // invalid JSON
                }
                refreshSpecView();
            });

            const up = mkIconBtn('fa-arrow-up', () => moveStep(i, -1));
            const down = mkIconBtn('fa-arrow-down', () => moveStep(i, 1));
            const del = mkIconBtn('fa-trash', () => { steps.splice(i, 1); renderSteps(); }, '#d9534f');

            row.append(idx, name, params, up, down, del);
            box.appendChild(row);
        });
        refreshSpecView();
    }

    function mkIconBtn(icon, onClick, color) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-secondary';
        b.style.cssText = 'padding:2px 8px; font-size:0.75rem;' + (color ? ('color:' + color + ';') : '');
        setHTML(b, html`<i class="fas ${icon}"></i>`);   // static icon class only
        b.addEventListener('click', onClick);
        return b;
    }

    function moveStep(i, dir) {
        const j = i + dir;
        if (j < 0 || j >= steps.length) return;
        [steps[i], steps[j]] = [steps[j], steps[i]];
        renderSteps();
    }

    // ── Toolbar actions (exposed on window for inline onclick) ─────────────────
    window.ppAddStep = function () {
        const sel = document.getElementById('pp-step-picker');
        if (!sel || !sel.value) return;
        const type = sel.value;
        const defaults = (catalog[type] && catalog[type].params) || {};
        steps.push({ type, params: JSON.parse(JSON.stringify(defaults)) });
        renderSteps();
    };

    window.ppLoadDefault = function () {
        steps = DEFAULT_STEPS
            .filter(t => catalog[t])
            .map(t => ({ type: t, params: structuredClone(catalog[t].params || {}) }));
        renderSteps();
    };

    window.ppRenderKeys = function () { refreshSpecView(); };

    // ── Resolve dataset (dry-run: read only dataset.json) ──────────────────────
    window.ppResolveDataset = async function () {
        const input = document.getElementById('pp-input-s3').value.trim();
        const status = document.getElementById('pp-resolve-status');
        const box = document.getElementById('pp-resolve-results');
        if (box) box.style.display = 'none';
        if (!input) { setBanner('pp-resolve-status', 'Enter the input S3 URI first.', 'error'); return; }
        setBanner('pp-resolve-status', 'Reading dataset.json…', 'info');
        try {
            // Fetch directly (not via apiCall) so we can render structured errors,
            // e.g. the "dataset.json missing" case that carries an example file.
            const resp = await fetch('/api/preprocessing/resolve-dataset?input_s3_uri=' + encodeURIComponent(input));
            const body = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                const detail = body && body.detail;
                if (detail && typeof detail === 'object' && detail.error === 'dataset_json_missing') {
                    renderMissingDatasetJson(detail);
                    if (status) status.style.display = 'none';
                    return;
                }
                const msg = typeof detail === 'string' ? detail
                    : Array.isArray(detail) ? detail.map(e => e.msg || JSON.stringify(e)).join(', ')
                    : (detail && detail.message) || 'Resolve failed';
                setBanner('pp-resolve-status', 'Resolve failed: ' + msg, 'error');
                return;
            }
            resolvedTaskType = body.task_type || null;   // derive task type
            renderResolved(body);
            if (status) status.style.display = 'none';
        } catch (err) {
            setBanner('pp-resolve-status', 'Resolve failed: ' + err.message, 'error');
        }
    };

    // Actionable panel for the "no dataset.json" case: message + example + copy.
    function renderMissingDatasetJson(detail) {
        const box = document.getElementById('pp-resolve-results');
        if (!box) return;
        const example = detail.example || '';
        const expected = detail.expected_location || '';
        let _markup = html`<div class="card" style="border-left:4px solid #e65100;"><div class="card-body">`;
        _markup = html`${_markup}${html`<div style="display:flex; align-items:flex-start; gap:0.6rem; margin-bottom:0.75rem;"><i class="fas fa-triangle-exclamation" style="color:#e65100; margin-top:3px;"></i><div><strong>No <code>dataset.json</code> found.</strong><br><span style="font-size:0.85rem; color:#555;">${detail.message || ''}</span></div></div>`}`;
        if (expected) {
            _markup = html`${_markup}${html`<div style="font-size:0.82rem; margin-bottom:0.5rem;"><strong>Expected location:</strong> <code>${expected}</code></div>`}`;
        }
        _markup = html`${_markup}${html`<div style="font-size:0.82rem; margin-bottom:0.35rem; display:flex; justify-content:space-between; align-items:center;"><strong>Example <code>dataset.json</code> (MSD/Decathlon format):</strong><button type="button" class="btn btn-secondary" style="padding:2px 10px; font-size:0.75rem;" onclick="ppCopyExample()"><i class="fas fa-copy"></i> Copy</button></div>`}`;
        _markup = html`${_markup}${html`<pre id="pp-example-json" style="background:#1e1e1e; color:#d4d4d4; padding:0.75rem; border-radius:6px; font-size:0.76rem; overflow:auto; max-height:340px; margin:0;">${example}</pre>`}`;
        _markup = html`${_markup}${html`<small style="color:#888; display:block; margin-top:0.5rem;">Paths are relative to the dataset root. Entries with a <code>label</code> are treated as segmentation; image-only entries as classification.</small>`}`;
        _markup = html`${_markup}${html`</div></div>`}`;
        setHTML(box, _markup);
        box.style.display = 'block';
    }

    window.ppCopyExample = function () {
        const pre = document.getElementById('pp-example-json');
        if (!pre) return;
        const text = pre.textContent || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).catch(() => {});
        }
    };

    function renderResolved(res) {
        const box = document.getElementById('pp-resolve-results');
        if (!box) return;
        const c = res.counts || {};
        const labels = res.labels || {};
        let _markup = html`<div class="card"><div class="card-body">`;
        _markup = html`${_markup}${html`<div style="display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.85rem; margin-bottom:0.5rem;">`}`;
        _markup = html`${_markup}${html`<div><strong>Dataset:</strong> ${res.name || '—'}</div>`}`;
        _markup = html`${_markup}${html`<div><strong>Task:</strong> <span style="color:#00695c; font-weight:600;">${res.task_type || '—'}</span> (derived)</div>`}`;
        if (res.source === 'imagefolder') {
            _markup = html`${_markup}${html`<div><strong>Source:</strong> ImageFolder (class = subfolder)</div>`}`;
        }
        _markup = html`${_markup}${html`<div><strong>Train:</strong> ${String(c.training ?? 0)}</div>`}`;
        _markup = html`${_markup}${html`<div><strong>Val:</strong> ${String(c.validation ?? 0)}</div>`}`;
        _markup = html`${_markup}${html`<div><strong>Test:</strong> ${String(c.test ?? 0)}</div>`}`;
        _markup = html`${_markup}${html`<div><strong>Total:</strong> ${String(res.total_pairs ?? 0)}</div>`}`;
        _markup = html`${_markup}${html`</div>`}`;

        // ImageFolder: show discovered classes and per-class counts.
        const classes = res.classes || {};
        const classNames = Object.keys(classes);
        if (classNames.length) {
            _markup = html`${_markup}${html`<div style="font-size:0.82rem; margin-bottom:0.5rem;"><strong>Classes (${classNames.length}):</strong> ${joinHtml(classNames.map(k => html`${k} (${String(classes[k])})`), ', ')}</div>`}`;
        }

        const labelKeys = Object.keys(labels);
        if (labelKeys.length) {
            _markup = html`${_markup}${html`<div style="font-size:0.82rem; margin-bottom:0.5rem;"><strong>Labels:</strong> ${joinHtml(labelKeys.map(k => html`${k}=${String(labels[k])}`), ', ')}</div>`}`;
        }

        const pairs = res.sample_pairs || [];
        if (pairs.length) {
            _markup = html`${_markup}${html`<table style="width:100%; font-size:0.8rem; border-collapse:collapse;"><tr style="text-align:left; color:#666; border-bottom:1px solid #ddd;"><th style="padding:4px;">#</th><th style="padding:4px;">split</th><th style="padding:4px;">image</th><th style="padding:4px;">label</th></tr>`}`;
            pairs.forEach((p, i) => {
                _markup = html`${_markup}${html`<tr style="border-bottom:1px solid #f2f2f2;"><td style="padding:4px; color:#999;">${i + 1}</td><td style="padding:4px;">${p.split || ''}</td><td style="padding:4px; font-family:monospace;">${p.image || ''}</td><td style="padding:4px; font-family:monospace;">${p.label || '—'}</td></tr>`}`;
            });
            _markup = html`${_markup}${html`</table>`}`;
            _markup = html`${_markup}${html`<small style="color:#888;">Showing first ${pairs.length} of ${String(res.total_pairs ?? pairs.length)} entries.</small>`}`;
        }
        _markup = html`${_markup}${html`</div></div>`}`;
        setHTML(box, _markup);
        box.style.display = 'block';
    }

    // ── Preview (local, one image) ─────────────────────────────────────────────
    async function runPreview(e) {
        e.preventDefault();
        const input = document.getElementById('pp-preview-input').value.trim();
        if (!input) return;
        setBanner('pp-preview-status', 'Running preview…', 'info');
        previewOk = false;
        updateSubmitGate();

        try {
            const res = await apiCall('/api/preprocessing/preview', {
                method: 'POST',
                body: JSON.stringify({ input_ref: input, spec: buildSpec() })
            });
            renderPreview(res);
            setBanner('pp-preview-status', 'Preview complete. You can now approve and run on the dataset.', 'success');
            previewOk = true;
            updateSubmitGate();
        } catch (err) {
            setBanner('pp-preview-status', 'Preview failed: ' + err.message, 'error');
        }
    }

    function renderPreview(res) {
        const wrap = document.getElementById('pp-preview-results');
        const statsBox = document.getElementById('pp-preview-stats');
        if (wrap) wrap.style.display = 'block';

        const inp = (res && res.input) || {};
        const out = (res && res.output) || {};
        const rows = [
            ['Shape', fmt(inp.shape), fmt(out.shape)],
            ['Spacing', fmt(inp.spacing), fmt(out.spacing)],
            ['Intensity min', fmt(inp.intensity_min), fmt(out.intensity_min)],
            ['Intensity max', fmt(inp.intensity_max), fmt(out.intensity_max)]
        ];
        let _markup = html`<table style="width:100%; font-size:0.85rem; border-collapse:collapse;"><tr style="text-align:left; color:#666;"><th></th><th>Before</th><th>After</th></tr>`;
        rows.forEach(r => {
            _markup = html`${_markup}${html`<tr><td style="padding:3px 6px; color:#555;">${r[0]}</td><td style="padding:3px 6px;">${r[1]}</td><td style="padding:3px 6px; font-weight:600;">${r[2]}</td></tr>`}`;
        });
        _markup = html`${_markup}${html`</table>`}`;

        if (Array.isArray(res && res.timings) && res.timings.length) {
            _markup = html`${_markup}${html`<h4 style="margin:0.75rem 0 0.25rem; font-size:0.82rem; color:#555;">Per-step timing</h4><ul style="margin:0; padding-left:1.1rem; font-size:0.8rem;">`}`;
            res.timings.forEach(t => {
                _markup = html`${_markup}${html`<li>${t.step}: ${fmt(t.seconds)}s</li>`}`;
            });
            _markup = html`${_markup}${html`</ul>`}`;
        }
        if (statsBox) setHTML(statsBox, _markup);

        const img = document.getElementById('pp-preview-image');
        const noimg = document.getElementById('pp-preview-noimage');
        const src = safeUrl((res && res.preview_image_url) || '');
        if (img && src) {
            img.src = src; img.style.display = 'inline-block';
            if (noimg) noimg.style.display = 'none';
        } else if (img && res && res.preview_image_base64) {
            img.src = 'data:image/png;base64,' + res.preview_image_base64;
            img.style.display = 'inline-block';
            if (noimg) noimg.style.display = 'none';
        } else {
            if (img) img.style.display = 'none';
            if (noimg) noimg.style.display = 'block';
        }
    }

    function fmt(v) {
        if (v == null) return '—';
        if (Array.isArray(v)) return '[' + v.map(x => (typeof x === 'number' ? +x.toFixed(3) : x)).join(', ') + ']';
        if (typeof v === 'number') return String(+v.toFixed(4));
        return String(v);
    }

    // ── Submit batch job ────────────────────────────────────────────────────────
    function updateSubmitGate() {
        const btn = document.getElementById('pp-submit-btn');
        const hint = document.getElementById('pp-submit-hint');
        if (!btn) return;
        btn.disabled = !previewOk;
        if (hint) hint.style.display = previewOk ? 'none' : 'block';
    }

    async function submitBatch(e) {
        if (e) e.preventDefault();
        console.log('[preprocessing] submit clicked; previewOk=', previewOk);
        if (!previewOk) {
            setBanner('pp-submit-status', 'Run a successful preview first, then submit.', 'error');
            return;
        }
        const body = {
            input_s3_uri: (document.getElementById('pp-input-s3').value || '').trim(),
            output_s3_uri: (document.getElementById('pp-output-s3').value || '').trim(),
            image_uri: (document.getElementById('pp-image-uri').value || '').trim(),
            role_arn: (document.getElementById('pp-role-arn').value || '').trim(),
            instance_type: (document.getElementById('pp-instance-type').value || '').trim() || 'ml.m5.2xlarge',
            spec: buildSpec()
        };
        // Basic client-side validation with a visible reason.
        const missing = [];
        if (!body.input_s3_uri) missing.push('Input S3 URI');
        if (!body.output_s3_uri) missing.push('Output S3 URI');
        if (!body.image_uri) missing.push('Processing Image URI');
        if (!body.role_arn) missing.push('SageMaker Role ARN');
        if (missing.length) {
            setBanner('pp-submit-status', 'Missing required field(s): ' + missing.join(', '), 'error');
            return;
        }
        setBanner('pp-submit-status', 'Submitting batch job…', 'info');
        console.log('[preprocessing] POST /api/preprocessing/submit', body);
        try {
            const res = await apiCall('/api/preprocessing/submit', {
                method: 'POST',
                body: JSON.stringify(body)
            });
            console.log('[preprocessing] submit response', res);
            setBanner('pp-submit-status', 'Submitted job: ' + (res.job_name || '(unknown)'), 'success');
            ppLoadJobs();
        } catch (err) {
            console.error('[preprocessing] submit failed', err);
            setBanner('pp-submit-status', 'Submit failed: ' + (err && err.message ? err.message : String(err)), 'error');
        }
    }
    // Expose so a direct button click can call it too (belt and suspenders).
    window.ppSubmitBatch = submitBatch;

    // ── Jobs table ──────────────────────────────────────────────────────────────
    let _jobsPollTimer = null;

    window.ppLoadJobs = async function () {
        const box = document.getElementById('pp-jobs-table');
        if (!box) return;
        try {
            const res = await apiCall('/api/preprocessing/jobs');
            const jobs = res.jobs || [];
            renderJobs(jobs);
            // Auto-poll while any job is still running so status updates to
            // Completed/Failed on its own; stop polling once none are active.
            const anyActive = jobs.some(j => j.status === 'InProgress' || j.status === 'Stopping');
            if (anyActive && !_jobsPollTimer) {
                _jobsPollTimer = setInterval(() => window.ppLoadJobs(), 15000);
            } else if (!anyActive && _jobsPollTimer) {
                clearInterval(_jobsPollTimer);
                _jobsPollTimer = null;
            }
        } catch (err) {
            setHTML(box, html`<p style="color:#888; font-size:0.85rem;">Jobs unavailable: ${err.message}</p>`);
        }
    };

    window.ppSyncJobs = async function () {
        try {
            await apiCall('/api/preprocessing/sync', { method: 'POST' });
            ppLoadJobs();
        } catch (err) {
            setBanner('pp-submit-status', 'Sync failed: ' + err.message, 'error');
        }
    };

    window.ppDeleteJob = async function (jobName) {
        if (!(await confirmAction('Delete job record ' + jobName + '? (S3 outputs are kept.)', { danger: true, confirmLabel: 'Delete' }))) return;
        try {
            await apiCall('/api/preprocessing/jobs/' + encodeURIComponent(jobName), { method: 'DELETE' });
            ppLoadJobs();
        } catch (err) {
            setBanner('pp-submit-status', 'Delete failed: ' + err.message, 'error');
        }
    };

    function renderJobs(jobs) {
        const box = document.getElementById('pp-jobs-table');
        if (!box) return;
        if (!jobs.length) {
            setHTML(box, html`<p style="color:#888; font-size:0.85rem;">No jobs yet.</p>`);
            return;
        }
        let _markup = html`<table style="width:100%; font-size:0.82rem; border-collapse:collapse;"><tr style="text-align:left; color:#666; border-bottom:1px solid #ddd;"><th style="padding:6px;">Job</th><th style="padding:6px;">Status</th><th style="padding:6px;">Output</th><th style="padding:6px;">Created</th><th></th></tr>`;
        jobs.forEach(j => {
            const name = j.job_name || '';
            const status = j.status || '';
            const color = j.status === 'Completed' ? '#2e7d32'
                : j.status === 'Failed' ? '#c62828'
                : j.status === 'InProgress' ? '#1565c0' : '#666';
            // Failed jobs: show the reason on hover; completed: show finish time.
            const statusCell = j.status === 'Failed' && j.failure_reason
                ? html`<span title="${j.failure_reason}">${status} ⓘ</span>`
                : (j.status === 'Completed' && j.completed_at)
                ? html`<span title="Completed ${formatDate(j.completed_at)}">${status}</span>`
                : status;
            _markup = html`${_markup}${html`<tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:6px; font-family:monospace;">${name}</td><td style="padding:6px; color:${color}; font-weight:600;">${statusCell}</td><td style="padding:6px; font-family:monospace; word-break:break-all;">${j.output_s3_uri || '—'}</td><td style="padding:6px;">${j.created_at ? formatDate(j.created_at) : '—'}</td><td style="padding:6px;"><button type="button" class="btn btn-secondary" style="padding:2px 8px; font-size:0.75rem; color:#d9534f;" onclick="ppDeleteJob('${escapeJsString(j.job_name || '')}')"><i class="fas fa-trash"></i></button></td></tr>`}`;
        });
        _markup = html`${_markup}${html`</table>`}`;
        setHTML(box, _markup);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────
    function setBanner(id, msg, type) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = msg;
        el.style.display = 'block';
        el.className = 'sm-status-banner ' + (type || 'info');
        el.style.cssText += type === 'error'
            ? 'background:#fdecea; color:#b71c1c; border:1px solid #f5c6cb; padding:0.6rem 0.9rem; border-radius:6px;'
            : type === 'success'
            ? 'background:#e8f5e9; color:#1b5e20; border:1px solid #c8e6c9; padding:0.6rem 0.9rem; border-radius:6px;'
            : 'background:#e3f2fd; color:#0d47a1; border:1px solid #bbdefb; padding:0.6rem 0.9rem; border-radius:6px;';
    }

    function prefillFromConfig() {
        const cfg = window.__appConfig || {};
        const img = document.getElementById('pp-image-uri');
        const role = document.getElementById('pp-role-arn');
        // Reuse the MONAI bundle / processing image + role if config exposes them.
        if (img && !img.value) img.value = cfg.MONAI_BUNDLE_IMAGE_URI || cfg.PROCESSING_IMAGE_URI || cfg.INFERENCE_IMAGE_URI || '';
        if (role && !role.value) role.value = cfg.SAGEMAKER_ROLE_ARN || cfg.ROLE_ARN || '';
    }

    // ── Tab init (runs once, on first visit) ───────────────────────────────────
    async function initPreprocessing() {
        renderStepPicker();

        // Try to load the live catalog; fall back silently to the built-in one.
        try {
            const res = await apiCall('/api/preprocessing/catalog');
            if (res && res.catalog && Object.keys(res.catalog).length) {
                catalog = res.catalog;
                renderStepPicker();
            }
        } catch (e) { /* fallback catalog already set */ }

        window.ppLoadDefault();          // seed a sensible default pipeline
        prefillFromConfig();
        updateSubmitGate();

        document.getElementById('pp-preview-form')?.addEventListener('submit', runPreview);
        document.getElementById('pp-submit-form')?.addEventListener('submit', submitBatch);
        document.getElementById('pp-import-file')?.addEventListener('change', handleImportFile);

        window.ppLoadJobs();
    }

    if (typeof registerTabInit === 'function') {
        registerTabInit('preprocessing', initPreprocessing);
    }
})();
