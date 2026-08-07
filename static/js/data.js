// Data - S3 Bucket Browser
// Depends on: escapeHtml() defined in main.js (loaded before this file)
if (typeof escapeHtml !== 'function') {
    throw new Error('escapeHtml() is required but not defined. Ensure main.js is loaded before data.js.');
}

registerTabInit('data', function() {
    const loadBucketsBtn = document.getElementById('load-buckets');
    if (loadBucketsBtn) loadBucketsBtn.addEventListener('click', loadS3Buckets);
    const loadUriBtn = document.getElementById('load-s3-uri');
    if (loadUriBtn) loadUriBtn.addEventListener('click', browseS3Uri);
});

async function browseS3Uri() {
    const uriInput = document.getElementById('s3-uri');
    const uri = uriInput ? uriInput.value.trim() : '';

    if (!uri) {
        showStatus('s3-uri-status', 'Please enter an S3 URI', 'error');
        return;
    }

    // Parse s3://bucket/prefix
    const match = uri.match(/^s3:\/\/([^/]+)\/?(.*)$/);
    if (!match) {
        showStatus('s3-uri-status', 'Invalid S3 URI format. Use s3://bucket-name/prefix/', 'error');
        return;
    }

    const bucket = match[1];
    const prefix = match[2] || '';

    const btn = document.getElementById('load-s3-uri');
    if (btn) { btn.disabled = true; btn.innerHTML = '<span class="btn-spinner"></span> Loading...'; }

    try {
        const response = await apiCall(`/api/aws/buckets/${bucket}/objects?prefix=${encodeURIComponent(prefix)}`);
        const container = document.getElementById('buckets-list');
        if (container) {
            if (response.objects.length === 0) {
                container.innerHTML = '<p>No objects found at that path.</p>';
            } else {
                container.innerHTML = response.objects.map(obj => `
                    <div class="bucket-item">
                        <div class="bucket-name">${escapeHtml(obj.key)}</div>
                        <div class="bucket-info">
                            <div>Size: ${(obj.size / 1024).toFixed(1)} KB</div>
                            <div>Modified: ${formatDate(obj.last_modified)}</div>
                        </div>
                    </div>
                `).join('');
            }
        }
        showStatus('s3-uri-status', `Found ${response.objects.length} object(s)`, 'success');
    } catch (error) {
        showStatus('s3-uri-status', `Error: ${error.message}`, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-search"></i> Browse'; }
    }
}

async function loadS3Buckets() {
    if (!window.MedMLApp.awsConnected) {
        showStatus('s3-uri-status', 'Please configure AWS credentials first', 'error');
        return;
    }

    const btn = document.getElementById('load-buckets');
    const bucketsContainer = document.getElementById('all-buckets-list');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-spinner"></span> Loading...';
    }

    if (bucketsContainer) {
        bucketsContainer.innerHTML = Array(3).fill(0).map(() => `
            <div class="bucket-item bucket-skeleton">
                <div class="skeleton-line skeleton-title"></div>
                <div class="skeleton-line skeleton-text"></div>
            </div>
        `).join('');
    }

    try {
        const buckets = await apiCall('/api/aws/buckets');
        displayBuckets(buckets);
    } catch (error) {
        if (bucketsContainer) bucketsContainer.innerHTML = '';
        showStatus('s3-uri-status', `Error loading buckets: ${error.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sync-alt"></i> Load S3 Buckets';
        }
    }
}

function displayBuckets(buckets) {
    const bucketsContainer = document.getElementById('all-buckets-list');
    if (!bucketsContainer) return;

    bucketsContainer.innerHTML = '';

    if (buckets.length === 0) {
        bucketsContainer.innerHTML = '<p>No S3 buckets found.</p>';
        return;
    }

    buckets.forEach(bucket => {
        const bucketElement = document.createElement('div');
        bucketElement.className = 'bucket-item';
        bucketElement.innerHTML = `
            <div class="bucket-name">${escapeHtml(bucket.name)}</div>
            <div class="bucket-info">
                <div>Region: ${escapeHtml(bucket.region)}</div>
                <div>Created: ${formatDate(bucket.creation_date)}</div>
            </div>
        `;
        // Clicking a bucket fills the URI field
        bucketElement.addEventListener('click', () => {
            const uriInput = document.getElementById('s3-uri');
            if (uriInput) uriInput.value = `s3://${bucket.name}/`;
        });
        bucketsContainer.appendChild(bucketElement);
    });
}
