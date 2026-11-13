// Safe DOM helpers for MedAnvil — loaded before every other script
// (templates/base.html and templates/viewer.html).
//
// Policy: never assign strings to innerHTML / outerHTML / insertAdjacentHTML or
// call document.write directly. Build markup with the `html` tagged template,
// which HTML-escapes EVERY interpolated value by default, and insert it with
// setHTML() / appendHTML(). Plain (untagged) strings passed to those sinks are
// treated as text and escaped, so a missed conversion fails safe (visible
// escaped markup) instead of becoming an XSS hole.
//
// Contexts:
//   element text / quoted attribute   html`<td title="${v}">${v}</td>`
//   inline JS handler argument        html`<button onclick="fn(${jsArg(v)})">`
//   href / src URL                    html`<a href="${safeUrl(v)}">`
//   nested fragments / lists          html`<ul>${items.map(i => html`<li>${i}</li>`)}</ul>`
//   static trusted markup chunk       html`<b>literal</b>` (no raw-string escape hatch)

'use strict';

// ── Escaping primitives ──────────────────────────────────────────────────────
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// URL allowlist for href/src: same-origin paths and http(s) only. Blocks
// javascript:, data:, vbscript: and protocol-relative (//host) URLs.
function safeUrl(url) {
    if (!url) return '';
    const s = String(url).trim();
    if (s.startsWith('//')) return '';
    if (s.startsWith('/') || s.startsWith('http://') || s.startsWith('https://')) return s;
    return '';
}

// JS string escaping for legacy inline-handler code: `fn('${escapeJsString(v)}')`.
// Prefer jsArg() in new code.
function escapeJsString(str) {
    if (str == null) return '';
    return String(str)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/"/g, '\\"')
        .replace(/</g, '\\x3c')
        .replace(/>/g, '\\x3e')
        .replace(/&/g, '\\x26')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029');
}

// ── SafeHTML value type + `html` tagged template ─────────────────────────────
class SafeHTML {
    constructor(value) { this.__html = value; Object.freeze(this); }
    toString() { return this.__html; }
}

const _MARKUP_RE = /<\/?[a-z][^>]*>/i;

function _htmlValue(v) {
    if (v instanceof SafeHTML) return v.__html;
    if (Array.isArray(v)) return v.map(_htmlValue).join('');
    if (v == null || v === false) return '';
    if (typeof v === 'string' && _MARKUP_RE.test(v)) {
        // A markup-looking plain string was interpolated: it is escaped (safe),
        // but the author probably meant a nested html`` fragment.
        console.debug('[safe_html] escaped markup-like string; wrap static markup in html``:', v.slice(0, 80));
    }
    return escapeHtml(v);
}

// Tagged template: literal parts are trusted (authored in source); every ${}
// value is escaped unless it is itself a SafeHTML produced by html``.
function html(strings, ...values) {
    let out = strings[0];
    for (let i = 0; i < values.length; i++) {
        out += _htmlValue(values[i]) + strings[i + 1];
    }
    return new SafeHTML(out);
}

// Join a list of fragments with a (static) separator fragment. Returns '' for
// an empty result so `list ? ... : ''` truthiness matches Array.join('').
function joinHtml(items, sep) {
    const s = sep instanceof SafeHTML ? sep.__html : escapeHtml(sep == null ? '' : sep);
    const out = (items || []).map(_htmlValue).join(s);
    return out === '' ? '' : new SafeHTML(out);
}

// Encode a value as a JS literal for an inline event-handler attribute.
// JSON.stringify yields a valid JS literal; HTML-escaping it survives the
// browser's attribute decoding, so the handler sees exactly that literal.
function jsArg(v) {
    const json = JSON.stringify(v === undefined ? null : v)
        .replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
    return new SafeHTML(escapeHtml(json));
}

function _toSafe(content) {
    if (content instanceof SafeHTML) return content;
    if (Array.isArray(content)) return new SafeHTML(_htmlValue(content));
    return new SafeHTML(escapeHtml(content == null ? '' : content));
}

// Parse SafeHTML into an inert fragment. DOMParser builds a separate, script-
// disabled document; wrapping in <template> keeps context-sensitive markup
// (<tr>, <td>, <option>) intact. The fragment is then imported into this
// document. This is the single HTML-parsing path in the app; it only ever
// receives SafeHTML built by html`` (whose values are escaped) or escaped text.
const _parser = new DOMParser();

function _fragment(content) {
    const doc = _parser.parseFromString('<template>' + _toSafe(content).__html + '</template>', 'text/html');
    const tpl = doc.querySelector('template');
    return tpl ? document.importNode(tpl.content, true) : document.createDocumentFragment();
}

// Replace el's children with the given SafeHTML (or escaped text).
function setHTML(el, content) {
    if (!el) return;
    el.replaceChildren(_fragment(content));
}

// Append SafeHTML (or escaped text) to the end of el.
function appendHTML(el, content) {
    if (!el) return;
    el.append(_fragment(content));
}

// Insert SafeHTML relative to el, same positions as insertAdjacentHTML.
function insertHTML(el, position, content) {
    if (!el) return;
    const frag = _fragment(content);
    switch (position) {
        case 'beforebegin': el.before(frag); break;
        case 'afterbegin': el.prepend(frag); break;
        case 'beforeend': el.append(frag); break;
        case 'afterend': el.after(frag); break;
        default: throw new Error('insertHTML: bad position ' + position);
    }
}

// ── Non-blocking notifications (replace window.alert) ────────────────────────
function _toastHost() {
    let host = document.getElementById('app-toast-host');
    if (!host) {
        host = document.createElement('div');
        host.id = 'app-toast-host';
        host.className = 'app-toast-host';
        host.setAttribute('role', 'status');
        host.setAttribute('aria-live', 'polite');
        document.body.appendChild(host);
    }
    return host;
}

// type: 'info' | 'success' | 'warning' | 'error'
function notify(message, type = 'info', timeoutMs) {
    const host = _toastHost();
    const toast = document.createElement('div');
    toast.className = 'app-toast app-toast-' + type;
    if (type === 'error') toast.setAttribute('role', 'alert');
    const text = document.createElement('span');
    text.className = 'app-toast-text';
    text.textContent = message == null ? '' : String(message);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'app-toast-close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.textContent = '×';
    close.addEventListener('click', () => toast.remove());
    toast.append(text, close);
    host.appendChild(toast);
    const ms = timeoutMs != null ? timeoutMs : (type === 'error' ? 10000 : 5000);
    if (ms > 0) setTimeout(() => toast.remove(), ms);
    return toast;
}

// ── In-page confirmation (replaces window.confirm) ───────────────────────────
// Returns Promise<boolean>. Uses a modal <dialog> rather than window.confirm(),
// which embedded browsers may suppress (auto-returning false).
function confirmAction(message, { confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false } = {}) {
    return new Promise(resolve => {
        const dlg = document.createElement('dialog');
        dlg.className = 'app-confirm';
        dlg.setAttribute('aria-modal', 'true');
        const msg = document.createElement('p');
        msg.className = 'app-confirm-message';
        msg.textContent = message == null ? '' : String(message);
        const row = document.createElement('div');
        row.className = 'app-confirm-actions';
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'btn btn-secondary';
        cancel.textContent = cancelLabel;
        const ok = document.createElement('button');
        ok.type = 'button';
        ok.className = danger ? 'btn btn-danger' : 'btn btn-primary';
        ok.textContent = confirmLabel;
        row.append(cancel, ok);
        dlg.append(msg, row);
        document.body.appendChild(dlg);
        let settled = false;
        const finish = (val) => {
            if (settled) return;
            settled = true;
            if (dlg.open) dlg.close();
            dlg.remove();
            resolve(val);
        };
        cancel.addEventListener('click', () => finish(false));
        ok.addEventListener('click', () => finish(true));
        dlg.addEventListener('cancel', (e) => { e.preventDefault(); finish(false); });
        dlg.addEventListener('close', () => finish(false));
        if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
        cancel.focus();
    });
}
