// Deployment JavaScript

document.addEventListener('DOMContentLoaded', function() {
    loadDeployments();
    
    const deploymentForm = document.getElementById('deployment-form');
    if (deploymentForm) {
        deploymentForm.addEventListener('submit', deployModel);
    }
    
    // Refresh deployments every 15 seconds
    setInterval(loadDeployments, 15000);
});

async function deployModel(e) {
    e.preventDefault();
    
    if (!window.MedMLApp.currentJobId) {
        alert('Please complete a training job first');
        return;
    }
    
    const formData = new FormData(e.target);
    const config = {
        endpoint_name: formData.get('endpoint_name'),
        instance_type: formData.get('instance_type'),
        initial_instance_count: parseInt(formData.get('initial_instance_count')),
        auto_scaling: formData.get('auto_scaling') === 'on'
    };
    
    if (!config.endpoint_name) {
        alert('Please enter an endpoint name');
        return;
    }
    
    try {
        const response = await apiCall('/api/deployment/deploy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `model_job_id=${window.MedMLApp.currentJobId}&${new URLSearchParams(config)}`
        });
        
        alert(`Model deployment started! Deployment ID: ${response.deployment_id}`);
        
        // Clear form
        e.target.reset();
        
        // Refresh deployments
        loadDeployments();
        
    } catch (error) {
        alert(`Error deploying model: ${error.message}`);
    }
}

async function loadDeployments() {
    try {
        const response = await apiCall('/api/deployment/deployments');
        displayDeployments(response.deployments);
    } catch (error) {
        console.error('Error loading deployments:', error);
    }
}

function displayDeployments(deployments) {
    const container = document.getElementById('deployments-list');
    if (!container) return;
    
    if (deployments.length === 0) {
        container.innerHTML = '<p>No deployments found.</p>';
        return;
    }
    
    container.innerHTML = '';
    
    deployments.forEach(deployment => {
        const deploymentElement = document.createElement('div');
        deploymentElement.className = 'deployment-item';
        
        const statusClass = deployment.status === 'active' ? 'success' : 
                           deployment.status === 'deploying' ? 'info' : 'error';
        
        deploymentElement.innerHTML = `
            <div class="deployment-header">
                <h4>${deployment.endpoint_name}</h4>
                <span class="status-badge ${statusClass}">${deployment.status.toUpperCase()}</span>
            </div>
            <div class="deployment-details">
                <div><strong>Deployment ID:</strong> ${deployment.deployment_id.substring(0, 8)}...</div>
                <div><strong>Instance Type:</strong> ${deployment.instance_type}</div>
                <div><strong>Instance Count:</strong> ${deployment.instance_count}</div>
                <div><strong>Auto Scaling:</strong> ${deployment.auto_scaling ? 'Enabled' : 'Disabled'}</div>
                ${deployment.status === 'active' ? 
                    `<div><strong>Endpoint URL:</strong> <code>${deployment.endpoint_url}</code></div>` : ''
                }
            </div>
            <div class="deployment-actions">
                ${deployment.status === 'active' ? 
                    `<button onclick="testEndpoint('${deployment.deployment_id}')" class="btn btn-primary">
                        <i class="fas fa-vial"></i> Test Endpoint
                    </button>` : ''
                }
                <button onclick="viewDeploymentDetails('${deployment.deployment_id}')" class="btn btn-secondary">
                    <i class="fas fa-info-circle"></i> Details
                </button>
                <button onclick="deleteDeployment('${deployment.deployment_id}')" class="btn btn-secondary" style="background-color: #e74c3c;">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        `;
        
        container.appendChild(deploymentElement);
    });
}

async function testEndpoint(deploymentId) {
    try {
        const response = await apiCall(`/api/deployment/deployments/${deploymentId}/test`, {
            method: 'POST'
        });
        
        const resultWindow = window.open('', '_blank', 'width=600,height=400');
        resultWindow.document.write(`
            <html>
                <head>
                    <title>Endpoint Test Results</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        .result { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
                        .success { background-color: #d4edda; color: #155724; }
                        .prediction { background-color: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 15px; }
                    </style>
                </head>
                <body>
                    <h1>Endpoint Test Results</h1>
                    <div class="result success">
                        <strong>Status:</strong> ${response.test_status}
                    </div>
                    <div class="result">
                        <strong>Response Time:</strong> ${response.response_time_ms}ms
                    </div>
                    <div class="prediction">
                        <h3>Prediction Results</h3>
                        <div><strong>Predicted Class:</strong> ${response.prediction.class}</div>
                        <div><strong>Confidence:</strong> ${(response.prediction.confidence * 100).toFixed(2)}%</div>
                        <h4>Class Probabilities:</h4>
                        ${Object.entries(response.prediction.probabilities).map(([cls, prob]) => 
                            `<div>${cls}: ${(prob * 100).toFixed(2)}%</div>`
                        ).join('')}
                    </div>
                </body>
            </html>
        `);
        
    } catch (error) {
        alert(`Error testing endpoint: ${error.message}`);
    }
}

async function viewDeploymentDetails(deploymentId) {
    try {
        const deployment = await apiCall(`/api/deployment/deployments/${deploymentId}`);
        
        const detailsWindow = window.open('', '_blank', 'width=600,height=500');
        detailsWindow.document.write(`
            <html>
                <head>
                    <title>Deployment Details</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        .detail { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
                        .status { padding: 5px 10px; border-radius: 15px; color: white; }
                        .active { background-color: #28a745; }
                        .deploying { background-color: #17a2b8; }
                    </style>
                </head>
                <body>
                    <h1>Deployment Details</h1>
                    <div class="detail"><strong>Deployment ID:</strong> ${deployment.deployment_id}</div>
                    <div class="detail"><strong>Model Job ID:</strong> ${deployment.model_job_id}</div>
                    <div class="detail"><strong>Endpoint Name:</strong> ${deployment.endpoint_name}</div>
                    <div class="detail">
                        <strong>Status:</strong> 
                        <span class="status ${deployment.status}">${deployment.status.toUpperCase()}</span>
                    </div>
                    <div class="detail"><strong>Instance Type:</strong> ${deployment.instance_type}</div>
                    <div class="detail"><strong>Instance Count:</strong> ${deployment.instance_count}</div>
                    <div class="detail"><strong>Auto Scaling:</strong> ${deployment.auto_scaling ? 'Enabled' : 'Disabled'}</div>
                    ${deployment.endpoint_url ? 
                        `<div class="detail"><strong>Endpoint URL:</strong><br><code>${deployment.endpoint_url}</code></div>` : ''
                    }
                </body>
            </html>
        `);
        
    } catch (error) {
        alert(`Error loading deployment details: ${error.message}`);
    }
}

async function deleteDeployment(deploymentId) {
    if (!confirm('Are you sure you want to delete this deployment? This action cannot be undone.')) {
        return;
    }
    
    try {
        await apiCall(`/api/deployment/deployments/${deploymentId}`, {
            method: 'DELETE'
        });
        
        alert('Deployment deleted successfully');
        loadDeployments();
        
    } catch (error) {
        alert(`Error deleting deployment: ${error.message}`);
    }
}