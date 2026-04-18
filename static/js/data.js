// Data - S3 Bucket Browser

document.addEventListener('DOMContentLoaded', function() {
    const loadBucketsBtn = document.getElementById('load-buckets');
    if (loadBucketsBtn) {
        loadBucketsBtn.addEventListener('click', loadS3Buckets);
    }
});

async function loadS3Buckets() {
    if (!window.MedMLApp.awsConnected) {
        showStatus('buckets-status', 'Please configure AWS credentials first', 'error');
        return;
    }

    const btn = document.getElementById('load-buckets');
    const bucketsContainer = document.getElementById('buckets-list');

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
        showStatus('buckets-status', `Error loading buckets: ${error.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sync-alt"></i> Load S3 Buckets';
        }
    }
}

function displayBuckets(buckets) {
    const bucketsContainer = document.getElementById('buckets-list');
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
            <div class="bucket-name">${bucket.name}</div>
            <div class="bucket-info">
                <div>Region: ${bucket.region}</div>
                <div>Created: ${formatDate(bucket.creation_date)}</div>
            </div>
        `;
        bucketsContainer.appendChild(bucketElement);
    });
}
