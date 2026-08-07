// Main JavaScript file for MedAnvil

// ── HTML escaping utility (prevents XSS in template strings) ─────────────────
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ── URL sanitization (prevents javascript: URI injection in href attributes) ──
function safeUrl(url) {
    if (!url) return '';
    const s = String(url).trim();
    if (s.startsWith('/') || s.startsWith('http://') || s.startsWith('https://')) return s;
    return '';
}

// ── JavaScript string escaping (for use inside inline onclick handlers) ───────
// HTML-escaping alone is insufficient in JS string contexts because browsers
// HTML-decode attribute values before executing them as JavaScript.
function escapeJsString(str) {
    if (str == null) return '';
    return String(str)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/"/g, '\\"')
        .replace(/</g, '\\x3c')
        .replace(/>/g, '\\x3e')
        .replace(/&/g, '\\x26');
}

// ── Tab system ────────────────────────────────────────────────────────────────
const _tabInits   = {};
const _tabVisited = {};

function registerTabInit(tabId, fn) { _tabInits[tabId] = fn; }

function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById('tab-' + tabId);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
    localStorage.setItem('activeTab', tabId);
    if (_tabInits[tabId] && !_tabVisited[tabId]) {
        _tabVisited[tabId] = true;
        _tabInits[tabId]();
    }
}

document.addEventListener('DOMContentLoaded', function () {
    // Activate saved tab (default: aws)
    const saved = localStorage.getItem('activeTab') || 'home';
    const btn   = document.querySelector(`.sidebar-item[data-tab="${saved}"]`);
    switchTab(saved, btn);

    // Load local-cache data immediately for key sections
    apiCall('/api/training/sagemaker/jobs?local_only=true').then(r => { if(typeof displaySageMakerJobs==='function') displaySageMakerJobs(r.jobs||[]); }).catch(()=>{});
    apiCall('/api/training/models').then(r => { if(typeof displayModels==='function'){ displayModels(r.models||[]); if(typeof populateSageMakerModelDropdown==='function') populateSageMakerModelDropdown(r.models||[]); }}).catch(()=>{});
    apiCall('/api/deployment/deployments?local_only=true').then(r => { if(typeof displayDeployments==='function') displayDeployments(r.deployments||[]); }).catch(()=>{});
    apiCall('/api/evaluation/jobs?local_only=true').then(r => { if(typeof renderEvalJobs==='function') renderEvalJobs(r.jobs||[]); }).catch(()=>{});
});

// Global state
window.MedMLApp = {
    awsConnected: false,
    selectedBucket: null,
    selectedModel: null,
    selectedSteps: [],
    currentJobId: null
};

// Utility functions
function showStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.className = `status-indicator ${type}`;
        element.style.display = 'block';
        
        // Auto-hide after 5 seconds for success messages
        if (type === 'success') {
            setTimeout(() => {
                element.style.display = 'none';
            }, 5000);
        }
    }
}

function hideStatus(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

// API helper functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json();
            const detail = error.detail;
            const msg = typeof detail === 'string' ? detail
                : Array.isArray(detail) ? detail.map(e => e.msg || JSON.stringify(e)).join(', ')
                : JSON.stringify(detail) || 'API call failed';
            throw new Error(msg);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}