// Standalone NIfTI prediction viewer page (NiiVue) — /viewer.
// Research use only — rendered predictions are not clinical diagnoses.
// Loads volumes through the same-origin proxy /api/viewer/fetch (no bucket
// CORS needed). Deep-linkable: /viewer?image=s3://...&mask=s3://...
// Self-contained: this page does not load main.js; the shared escaping and
// safe-DOM helpers come from safe_html.js (loaded first in viewer.html).

let _nvViewer = null;

function viewerInit() {
    if (_nvViewer) return _nvViewer;
    _nvViewer = new niivue.Niivue({
        show3Dcrosshair: true,
        backColor: [0.08, 0.09, 0.11, 1],
    });
    _nvViewer.attachTo('viewer-canvas');
    _nvViewer.setSliceType(_nvViewer.sliceTypeMultiplanar);
    _nvViewer.onLocationChange = _viewerOnLocation;
    return _nvViewer;
}

// ── Segmentation legend (right sidebar) ──
// Built from the mask itself: the label values present, their voxel counts and
// a discrete color per label (applied to the overlay so legend == render).
// Names come from the endpoint's bundle (GET /api/viewer/labels); without them
// labels show as "Label N". Research use only — not a clinical diagnosis.
const _LEGEND_MAX_LABELS = 1024;   // beyond this it is not a label mask
const _legend = { vol: null, maskUri: '', labels: [], counts: {}, names: {},
                  hidden: new Set(), maxLabel: 0, voxelMl: 0 };

// Distinct, deterministic color per label value (golden-angle hue walk).
function _legendRgb(v) {
    const h = (v * 137.508) % 360 / 360;
    const s = [0.75, 0.6, 0.9][v % 3];
    const l = [0.5, 0.62, 0.42][Math.floor(v / 3) % 3];
    const hue2 = function (p, q, t) {
        if (t < 0) t += 1; if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    return [hue2(p, q, h + 1 / 3), hue2(p, q, h), hue2(p, q, h - 1 / 3)]
        .map(function (c) { return Math.round(c * 255); });
}

function _legendName(v) {
    return _legend.names[String(v)] || ('Label ' + v);
}

function _viewerHideLegend() {
    const el = document.getElementById('viewer-legend');
    if (el) el.style.display = 'none';
    _legend.vol = null;
}

function _viewerBuildLegend(vol, maskUri) {
    if (!vol || !vol.img) { _viewerHideLegend(); return; }
    if (maskUri !== _legend.maskUri) {   // new mask: reset toggles and names
        _legend.hidden = new Set();
        _legend.names = {};
        _legend.maskUri = maskUri;
    }
    const img = vol.img;
    const slope = (vol.hdr && vol.hdr.scl_slope) ? vol.hdr.scl_slope : 1;
    const inter = (vol.hdr && vol.hdr.scl_inter) ? vol.hdr.scl_inter : 0;
    const counts = {};
    let n = 0;
    for (let i = 0; i < img.length; i++) {
        const v = img[i] * slope + inter;
        if (v === 0) continue;
        if (!Number.isInteger(v) || v < 0 || v > 65535) { _viewerHideLegend(); return; }
        if (counts[v] === undefined) {
            if (++n > _LEGEND_MAX_LABELS) { _viewerHideLegend(); return; }
            counts[v] = 0;
        }
        counts[v]++;
    }
    const pd = (vol.hdr && vol.hdr.pixDims) || [];
    _legend.voxelMl = (pd[1] && pd[2] && pd[3]) ? Math.abs(pd[1] * pd[2] * pd[3]) / 1000 : 0;
    _legend.vol = vol;
    _legend.counts = counts;
    _legend.labels = Object.keys(counts).map(Number).sort(function (a, b) { return a - b; });
    _legend.maxLabel = _legend.labels.length ? _legend.labels[_legend.labels.length - 1] : 0;
    _viewerApplyLabelLut();
    document.getElementById('viewer-legend').style.display = 'flex';
    viewerRenderLegend();
    if (!Object.keys(_legend.names).length) viewerLoadLabelNames(true);
}

// Discrete label LUT on the mask volume; hidden labels get alpha 0.
function _viewerApplyLabelLut() {
    const vol = _legend.vol;
    if (!vol || !_nvViewer) return;
    const M = Math.max(_legend.maxLabel, 1);
    const R = [], G = [], B = [], A = [], I = [], labels = [];
    for (let v = 0; v <= M; v++) {
        const rgb = _legendRgb(v);
        R.push(rgb[0]); G.push(rgb[1]); B.push(rgb[2]); I.push(v);
        A.push(v === 0 || _legend.hidden.has(v) || _legend.counts[v] === undefined ? 0 : 255);
        labels.push(v === 0 ? '' : _legendName(v));
    }
    vol.setColormapLabel({ R: R, G: G, B: B, A: A, I: I, labels: labels });
    _nvViewer.updateGLVolume();
}

function viewerRenderLegend() {
    const list = document.getElementById('viewer-legend-list');
    if (!list) return;
    const filter = (document.getElementById('viewer-legend-filter').value || '').trim().toLowerCase();
    list.replaceChildren();
    let shown = 0;
    _legend.labels.forEach(function (v) {
        const name = _legendName(v);
        if (filter && name.toLowerCase().replace(/_/g, ' ').indexOf(filter.replace(/_/g, ' ')) < 0
                && String(v) !== filter) return;
        shown++;
        const on = !_legend.hidden.has(v);
        const li = document.createElement('li');
        li.className = 'viewer-legend-item' + (on ? '' : ' off');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = on;
        cb.setAttribute('aria-label', 'Show ' + name);
        const sw = document.createElement('span');
        sw.className = 'viewer-legend-swatch';
        sw.style.background = 'rgb(' + _legendRgb(v).join(',') + ')';
        const nm = document.createElement('span');
        nm.className = 'viewer-legend-name';
        nm.textContent = name.replace(/_/g, ' ');    // names come from bundle metadata — never innerHTML
        nm.title = name;
        const meta = document.createElement('span');
        meta.className = 'viewer-legend-meta';
        meta.textContent = _legend.voxelMl
            ? (function (ml) { return ml.toFixed(ml < 10 ? 2 : 1); })(_legend.counts[v] * _legend.voxelMl) + ' mL'
            : _legend.counts[v].toLocaleString() + ' vox';
        meta.title = 'Label value ' + v + ' · ' + _legend.counts[v].toLocaleString() + ' voxels';
        li.append(cb, sw, nm, meta);
        li.addEventListener('click', function (ev) {
            if (ev.target !== cb) cb.checked = !cb.checked;
            if (cb.checked) _legend.hidden.delete(v); else _legend.hidden.add(v);
            li.classList.toggle('off', !cb.checked);
            _viewerApplyLabelLut();
        });
        list.appendChild(li);
    });
    document.getElementById('viewer-legend-count').textContent =
        (filter ? shown + ' / ' : '') + _legend.labels.length + ' labels';
}

function viewerSetAllLabels(show) {
    _legend.hidden = show ? new Set() : new Set(_legend.labels);
    _viewerApplyLabelLut();
    viewerRenderLegend();
}

async function viewerLoadLabelNames(silent) {
    const epField = document.getElementById('viewer-legend-endpoint');
    const ep = epField.value.trim();
    const note = document.getElementById('viewer-legend-note');
    // No endpoint given: let the backend match the mask's S3 location to the
    // endpoint that writes there.
    const query = ep ? 'endpoint_name=' + encodeURIComponent(ep)
                     : (_legend.maskUri ? 'mask_s3_uri=' + encodeURIComponent(_legend.maskUri) : '');
    if (!query) { if (!silent) note.textContent = 'Enter the endpoint that produced this mask.'; return; }
    note.textContent = 'Looking up label names…';
    try {
        const r = await fetch('/api/viewer/labels?' + query);
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'label lookup failed');
        if (!ep && d.endpoint_name) epField.value = d.endpoint_name;
        const src = d.endpoint_name || ep;
        _legend.names = d.labels || {};
        const n = Object.keys(_legend.names).length;
        note.textContent = n ? ('Names from ' + src + ' (' + n + ' defined).')
                             : (d.note || 'No label names found — showing label values.');
        _viewerApplyLabelLut();
        viewerRenderLegend();
    } catch (e) {
        note.textContent = 'Label names unavailable: ' + e.message;
    }
}

// Status bar: label name under the crosshair (mask is the last volume).
function _viewerOnLocation(d) {
    const el = document.getElementById('viewer-location');
    if (!el || !d || !d.values) return;
    const parts = [];
    const idx = window._viewerMaskIndex;
    if (idx >= 0 && d.values[idx] && _legend.vol) {
        const v = Math.round(d.values[idx].value);
        parts.push(v > 0 ? 'Label: ' + _legendName(v) + ' (' + v + ')' : 'Label: —');
    }
    if (idx !== 0 && d.values[0] && Number.isFinite(d.values[0].value)) {
        parts.push('Intensity: ' + Math.round(d.values[0].value * 10) / 10);
    }
    el.textContent = parts.join(' · ');
}

// Same-origin proxy URL for a NIfTI object — the backend streams it from S3.
function _viewerUrl(s3Uri) {
    return '/api/viewer/fetch?s3_uri=' + encodeURIComponent(s3Uri);
}

// Volume name for NiiVue — it picks its parser from the extension.
// The backend converts .dcm to gzipped NIfTI, so name those .nii.gz.
function _viewerName(s3Uri) {
    const base = s3Uri.split('/').pop();
    return /\.dcm$/i.test(base) ? base + '.nii.gz' : base;
}

async function viewerLoad(silent) {
    const imgUri = document.getElementById('viewer-image-uri').value.trim();
    const maskUri = document.getElementById('viewer-mask-uri').value.trim();
    const status = document.getElementById('viewer-status');
    if (!imgUri && !maskUri) {
        if (!silent) notify('Provide an image and/or mask S3 URI', 'info');
        return;
    }
    status.textContent = 'Loading volumes…';
    try {
        const nv = viewerInit();
        const volumes = [];
        if (imgUri) {
            volumes.push({ url: _viewerUrl(imgUri),
                           name: _viewerName(imgUri) });
        }
        if (maskUri) {
            volumes.push({ url: _viewerUrl(maskUri),
                           name: _viewerName(maskUri),
                           colormap: 'nih',            // multi-label segmentation colors
                           opacity: 0.55 });
        }
        await nv.loadVolumes(volumes);
        // Single-slice (2D) volumes: multiplanar/3D render adds nothing and
        // can trip NiiVue's render-object math — switch to axial.
        const dims = nv.volumes[0] && nv.volumes[0].dims; // [ndim, x, y, z, ...]
        if (dims && dims[3] === 1) {
            nv.setSliceType(nv.sliceTypeAxial);
            const sel = document.getElementById('viewer-slice-type');
            if (sel) sel.value = 'axial';
        }
        // mask is the last volume when both are loaded
        window._viewerMaskIndex = volumes.length > 1 ? 1 : (maskUri ? 0 : -1);
        if (maskUri) {
            _viewerBuildLegend(nv.volumes[window._viewerMaskIndex], maskUri);
        } else {
            _viewerHideLegend();
        }
        status.textContent = volumes.length === 2
            ? 'Loaded image + prediction overlay.'
            : 'Loaded 1 volume.';
    } catch (e) {
        status.textContent = 'Load failed: ' + e.message;
        if (!silent) notify('Viewer load failed: ' + e.message, 'error');
    }
}

function viewerSetOpacity(val) {
    document.getElementById('viewer-opacity-label').textContent =
        Math.round(Number(val) * 100) + '%';
    if (!_nvViewer || window._viewerMaskIndex < 0) return;
    _nvViewer.setOpacity(window._viewerMaskIndex, Number(val));
}

function viewerSetSliceType(kind) {
    const nv = viewerInit();
    const map = {
        axial: nv.sliceTypeAxial,
        coronal: nv.sliceTypeCoronal,
        sagittal: nv.sliceTypeSagittal,
        render: nv.sliceTypeRender,
        multiplanar: nv.sliceTypeMultiplanar,
    };
    nv.setSliceType(map[kind] !== undefined ? map[kind] : nv.sliceTypeMultiplanar);
}

async function viewerLoadRecent() {
    const list = document.getElementById('viewer-recent-list');
    if (!list) return;
    setHTML(list, html`<p style="color:#888; padding:0.6rem;">Loading…</p>`);
    try {
        const r = await fetch('/api/viewer/recent-masks');
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'recent-masks failed');
        const masks = d.masks || [];
        if (!masks.length) {
            setHTML(list, html`<p style="color:#888; padding:0.6rem; font-size:0.8rem;">${d.note || 'No prediction masks found yet.'}</p>`);
            return;
        }
        setHTML(list, joinHtml(masks.map(function (m) {
            return html`<div style="padding:0.45rem 0.6rem; border-bottom:1px solid #f0f0f0; cursor:pointer; display:flex; justify-content:space-between; gap:0.5rem;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background=''" onclick="viewerPickMask('${escapeJsString(m.s3_uri)}')"><span style="font-size:0.8rem; overflow-wrap:anywhere;">${m.name}</span><span style="font-size:0.72rem; color:#888; white-space:nowrap;">${(m.last_modified || '').slice(0, 16).replace('T', ' ')}</span></div>`;
        }), ''));
    } catch (e) {
        setHTML(list, html`<p style="color:#c62828; padding:0.6rem; font-size:0.8rem;">${e.message}</p>`);
    }
}

function viewerPickMask(s3Uri) {
    document.getElementById('viewer-mask-uri').value = s3Uri;
    // The endpoint field may belong to a different mask — don't apply its names.
    document.getElementById('viewer-legend-endpoint').value = '';
    // If no background image chosen, view the mask alone; user can add the image.
    viewerLoad();
}

// ── Classification result panel (right sidebar) ──
// Shown when the displayed source image matches the last inference run
// remembered by the inference tab. Research use only — not a diagnosis.
function viewerShowClassification(cls) {
    const panel = document.getElementById('viewer-classification');
    const body = document.getElementById('viewer-classification-body');
    if (!panel || !body || !cls || cls.predicted_class == null) return;
    let _markup = html`<div style="font-size:1.05rem; font-weight:700; color:#232f3e;">${cls.predicted_class}</div>`;
    if (cls.confidence != null) {
        _markup = html`${_markup}${html`<div style="font-size:0.8rem; color:#666; margin-bottom:0.5rem;">Confidence: ${(Number(cls.confidence) * 100).toFixed(1)}%</div>`}`;
    }
    if (cls.probabilities) {
        const sorted = Object.entries(cls.probabilities).sort((a, b) => b[1] - a[1]);
        _markup = html`${_markup}${html`<div class="inf-prob-bars">${joinHtml(sorted.map(function ([c, p]) {
            const pct = (Number(p) * 100).toFixed(1);
            return html`<div class="inf-prob-row"><span class="inf-prob-label">${c}</span><div class="inf-prob-bar-wrap"><div class="inf-prob-bar" style="width:${pct}%"></div></div><span class="inf-prob-value">${pct}%</span></div>`;
        }), '')}</div>`}`;
    }
    _markup = html`${_markup}${html`<p style="font-size:0.68rem; color:#999; margin:0.6rem 0 0;">Research use only — not a clinical diagnosis.</p>`}`;
    setHTML(body, _markup);
    panel.style.display = 'block';
}

// ── Page init: read ?image=&mask= deep-link params and auto-load. Without
// params, fall back to the last segmentation run saved by the inference tab.
document.addEventListener('DOMContentLoaded', function () {
    viewerInit();
    viewerLoadRecent();
    const params = new URLSearchParams(window.location.search);
    let image = params.get('image') || '';
    let mask = params.get('mask') || '';
    let last = null;
    try {
        last = JSON.parse(localStorage.getItem('medanvil_viewer_last') || 'null');
    } catch (_) { /* corrupt entry — ignore */ }
    if (!image && !mask && last) {
        image = last.image || '';
        mask = last.mask || '';
    }
    if (image) document.getElementById('viewer-image-uri').value = image;
    if (mask) document.getElementById('viewer-mask-uri').value = mask;
    // Endpoint for legend label names: deep-link param, else the last run's
    // endpoint when it produced the mask being shown.
    const ep = params.get('endpoint') ||
        (last && last.endpoint && mask && last.mask === mask ? last.endpoint : '');
    if (ep) document.getElementById('viewer-legend-endpoint').value = ep;
    if (image || mask) viewerLoad(true);   // silent — no alert on auto-load
    // Show the classification result when it belongs to the displayed image
    if (last && last.classification && image && last.image === image) {
        viewerShowClassification(last.classification);
    }
});
