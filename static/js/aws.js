// AWS Configuration JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const awsForm = document.getElementById('aws-config-form');
    
    if (awsForm) {
        awsForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const credentials = {
                access_key_id: formData.get('access_key_id'),
                secret_access_key: formData.get('secret_access_key'),
                region: formData.get('region')
            };
            
            try {
                showStatus('aws-status', 'Connecting to AWS...', 'info');
                
                const response = await apiCall('/api/aws/configure', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(credentials)
                });
                
                if (response.status === 'success') {
                    showStatus('aws-status', 'Successfully connected to AWS!', 'success');
                    window.MedMLApp.awsConnected = true;
                    const indicator = document.getElementById('aws-connection-indicator');
                    if (indicator) {
                        indicator.textContent = '● Connected';
                        indicator.className = 'aws-connected';
                    }
                    
                    // Enable other sections
                    enableSection('data-import');
                    
                    // Clear sensitive form data
                    document.getElementById('access-key').value = '';
                    document.getElementById('secret-key').value = '';
                } else {
                    throw new Error('Failed to configure AWS credentials');
                }
                
            } catch (error) {
                showStatus('aws-status', `Error: ${error.message}`, 'error');
                window.MedMLApp.awsConnected = false;
            }
        });
    }
});

function enableSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.opacity = '1';
        section.style.pointerEvents = 'auto';
        
        // Remove any disabled styling
        const inputs = section.querySelectorAll('input, button, select');
        inputs.forEach(input => {
            input.disabled = false;
        });
    }
}

function disableSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.opacity = '0.5';
        section.style.pointerEvents = 'none';
        
        // Disable all inputs
        const inputs = section.querySelectorAll('input, button, select');
        inputs.forEach(input => {
            input.disabled = true;
        });
    }
}

// Initialize sections as disabled
document.addEventListener('DOMContentLoaded', function() {
    disableSection('data-import');
    disableSection('preprocessing');
    disableSection('training');
    disableSection('deployment');

    // Check if already connected via environment credentials
    checkAwsStatus();
});

async function checkAwsStatus() {
    try {
        const response = await apiCall('/api/aws/status');
        const indicator = document.getElementById('aws-connection-indicator');
        if (response.configured) {
            window.MedMLApp.awsConnected = true;
            if (indicator) {
                indicator.textContent = '● Connected';
                indicator.className = 'aws-connected';
            }
            showStatus('aws-status', `Connected to AWS (${response.source} credentials)`, 'success');
            enableSection('data-import');
            enableSection('preprocessing');
            enableSection('training');
            enableSection('deployment');
        }
    } catch (error) {
        console.log('AWS not configured from environment');
    }
}