// Main JavaScript file for MedAnvil

// escapeHtml / safeUrl / escapeJsString / html`` / setHTML / notify /
// confirmAction live in safe_html.js (loaded before this file).

// ── Tab system ────────────────────────────────────────────────────────────────
// Map (not a plain object) so a tab id read from localStorage can only ever
// resolve to an init function a module explicitly registered — never to an
// inherited Object.prototype member.
const _tabInits   = new Map();
const _tabVisited = new Set();

function registerTabInit(tabId, fn) {
    if (typeof fn !== 'function') return;
    // Chain: multiple modules may register for the same tab (e.g. inference +
    // the prediction viewer). Overwriting would silently drop earlier inits.
    const prev = _tabInits.get(tabId);
    _tabInits.set(tabId, prev ? function () { prev(); fn(); } : fn);
}

function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById('tab-' + tabId);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
    localStorage.setItem('activeTab', tabId);
    const init = _tabInits.get(tabId);
    if (typeof init === 'function' && !_tabVisited.has(tabId)) {
        _tabVisited.add(tabId);
        init();
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