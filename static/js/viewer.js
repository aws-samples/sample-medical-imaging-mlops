// Standalone NIfTI prediction viewer page (NiiVue) — /viewer.
// Research use only — rendered predictions are not clinical diagnoses.
// Loads volumes through the same-origin proxy /api/viewer/fetch (no bucket
// CORS needed). Deep-linkable: /viewer?image=s3://...&mask=s3://...
// Self-contained: this page does not load main.js, so helpers live here.

let _nvViewer = null;

function _viewerEscapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function _viewerEscapeJs(str) {
    return String(str).replace(/\\/g, '\\\\').replace(/'/g, "\\'")
                      .replace(/"/g, '\\"').replace(/</g, '\\x3c');
}

function viewerInit() {
    if (_nvViewer) return _nvViewer;
    _nvViewer = new niivue.Niivue({
        show3Dcrosshair: true,
        backColor: [0.08, 0.09, 0.11, 1],
    });
    _nvViewer.attachTo('viewer-canvas');
    _nvViewer.setSliceType(_nvViewer.sliceTypeMultiplanar);
    return _nvViewer;
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
        if (!silent) alert('Provide an image and/or mask S3 URI');
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
        status.textContent = volumes.length === 2
            ? 'Loaded image + prediction overlay.'
            : 'Loaded 1 volume.';
    } catch (e) {
        status.textContent = 'Load failed: ' + e.message;
        if (!silent) alert('Viewer load failed: ' + e.message);
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
    list.innerHTML = '<p style="color:#888; padding:0.6rem;">Loading…</p>';
    try {
        const r = await fetch('/api/viewer/recent-masks');
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'recent-masks failed');
        const masks = d.masks || [];
        if (!masks.length) {
            list.innerHTML = '<p style="color:#888; padding:0.6rem; font-size:0.8rem;">' +
                _viewerEscapeHtml(d.note || 'No prediction masks found yet.') + '</p>';
            return;
        }
        list.innerHTML = masks.map(function (m) {
            return '<div style="padding:0.45rem 0.6rem; border-bottom:1px solid #f0f0f0; cursor:pointer; display:flex; justify-content:space-between; gap:0.5rem;"' +
                ' onmouseover="this.style.background=\'#f5f5f5\'" onmouseout="this.style.background=\'\'"' +
                ' onclick="viewerPickMask(\'' + _viewerEscapeJs(m.s3_uri) + '\')">' +
                '<span style="font-size:0.8rem; overflow-wrap:anywhere;">' + _viewerEscapeHtml(m.name) + '</span>' +
                '<span style="font-size:0.72rem; color:#888; white-space:nowrap;">' +
                    _viewerEscapeHtml((m.last_modified || '').slice(0, 16).replace('T', ' ')) + '</span>' +
            '</div>';
        }).join('');
    } catch (e) {
        list.innerHTML = '<p style="color:#c62828; padding:0.6rem; font-size:0.8rem;">' +
            _viewerEscapeHtml(e.message) + '</p>';
    }
}

function viewerPickMask(s3Uri) {
    document.getElementById('viewer-mask-uri').value = s3Uri;
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
    let html = '<div style="font-size:1.05rem; font-weight:700; color:#232f3e;">' +
        _viewerEscapeHtml(cls.predicted_class) + '</div>';
    if (cls.confidence != null) {
        html += '<div style="font-size:0.8rem; color:#666; margin-bottom:0.5rem;">Confidence: ' +
            (Number(cls.confidence) * 100).toFixed(1) + '%</div>';
    }
    if (cls.probabilities) {
        const sorted = Object.entries(cls.probabilities).sort((a, b) => b[1] - a[1]);
        html += '<div class="inf-prob-bars">' + sorted.map(function ([c, p]) {
            const pct = (Number(p) * 100).toFixed(1);
            return '<div class="inf-prob-row">' +
                '<span class="inf-prob-label">' + _viewerEscapeHtml(c) + '</span>' +
                '<div class="inf-prob-bar-wrap"><div class="inf-prob-bar" style="width:' + pct + '%"></div></div>' +
                '<span class="inf-prob-value">' + pct + '%</span></div>';
        }).join('') + '</div>';
    }
    html += '<p style="font-size:0.68rem; color:#999; margin:0.6rem 0 0;">Research use only — not a clinical diagnosis.</p>';
    body.innerHTML = html;
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
    if (image || mask) viewerLoad(true);   // silent — no alert on auto-load
    // Show the classification result when it belongs to the displayed image
    if (last && last.classification && image && last.image === image) {
        viewerShowClassification(last.classification);
    }
});
