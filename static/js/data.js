// Data Import JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const loadBucketsBtn = document.getElementById('load-buckets');
    const importForm = document.getElementById('import-form');
    
    if (loadBucketsBtn) {
        loadBucketsBtn.addEventListener('click', loadS3Buckets);
    }
    
    if (importForm) {
        importForm.addEventListener('submit', importDataFromS3);
    }
});

async function loadS3Buckets() {
    if (!window.MedMLApp.awsConnected) {
        showStatus('import-status', 'Please configure AWS credentials first', 'error');
        return;
    }

    const btn = document.getElementById('load-buckets');
    const bucketsContainer = document.getElementById('buckets-list');

    // Show button loading state
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-spinner"></span> Loading...';
    }

    // Show skeleton placeholders
    if (bucketsContainer) {
        bucketsContainer.innerHTML = Array(3).fill(0).map(() => `
            <div class="bucket-item bucket-skeleton">
                <div class="skeleton-line skeleton-title"></div>
                <div class="skeleton-line skeleton-text"></div>
                <div class="skeleton-line skeleton-text short"></div>
            </div>
        `).join('');
    }

    try {
        const buckets = await apiCall('/api/aws/buckets');
        displayBuckets(buckets);
        hideStatus('import-status');
    } catch (error) {
        if (bucketsContainer) bucketsContainer.innerHTML = '';
        showStatus('import-status', `Error loading buckets: ${error.message}`, 'error');
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
        
        bucketElement.addEventListener('click', () => selectBucket(bucket, bucketElement));
        bucketsContainer.appendChild(bucketElement);
    });
}

function selectBucket(bucket, element) {
    // Remove previous selection
    document.querySelectorAll('.bucket-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Select current bucket
    element.classList.add('selected');
    window.MedMLApp.selectedBucket = bucket;
    
    // Load bucket objects
    loadBucketObjects(bucket.name);
}

async function loadBucketObjects(bucketName) {
    try {
        const response = await apiCall(`/api/aws/buckets/${bucketName}/objects`);
        console.log(`Loaded ${response.objects.length} objects from bucket ${bucketName}`);
        
        // Enable import form
        const importForm = document.getElementById('import-form');
        if (importForm) {
            importForm.style.display = 'block';
        }
        
    } catch (error) {
        showStatus('import-status', `Error loading bucket contents: ${error.message}`, 'error');
    }
}

async function importDataFromS3(e) {
    e.preventDefault();
    
    if (!window.MedMLApp.selectedBucket) {
        showStatus('import-status', 'Please select an S3 bucket first', 'error');
        return;
    }
    
    const formData = new FormData(e.target);
    const prefix = formData.get('prefix') || '';
    
    try {
        showStatus('import-status', 'Importing data from S3...', 'info');
        
        const importRequest = {
            bucket_name: window.MedMLApp.selectedBucket.name,
            prefix: prefix
        };
        
        const response = await apiCall('/api/data/import', {
            method: 'POST',
            body: JSON.stringify(importRequest)
        });
        
        showStatus('import-status', response.message, 'success');
        enableSection('preprocessing');
        
    } catch (error) {
        showStatus('import-status', `Import failed: ${error.message}`, 'error');
    }
}