// Training JavaScript

document.addEventListener('DOMContentLoaded', function() {
    loadAvailableModels();
    loadTrainingJobs();
    loadImportedModels();
    loadSageMakerJobs();
    
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', startTraining);
    }

    const importForm = document.getElementById('model-import-form');
    if (importForm) {
        importForm.addEventListener('submit', importModel);
    }

    const smForm = document.getElementById('sagemaker-training-form');
    if (smForm) {
        smForm.addEventListener('submit', startSageMakerTraining);
    }
    
    // Refresh training jobs every 10 seconds
    setInterval(loadTrainingJobs, 10000);
    setInterval(loadSageMakerJobs, 15000);
});

async function loadAvailableModels() {
    try {
        const response = await apiCall('/api/training/models');
        displayModels(response.models);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function displayModels(models) {
    const container = document.getElementById('models-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    models.forEach(model => {
        const modelElement = document.createElement('div');
        modelElement.className = 'model-item';
        modelElement.dataset.modelType = model.name;
        
        modelElement.innerHTML = `
            <div class="model-name">${model.name.toUpperCase()}</div>
            <div class="model-description">${model.description}</div>
            <div class="model-specs">
                <span>Parameters: ${model.parameters}</span>
                <span>Input: ${model.input_size}</span>
                <span>${model.pretrained ? 'Pre-trained' : 'Custom'}</span>
            </div>
        `;
        
        modelElement.addEventListener('click', () => selectModel(model, modelElement));
        container.appendChild(modelElement);
    });
}

function selectModel(model, element) {
    // Remove previous selection
    document.querySelectorAll('.model-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Select current model
    element.classList.add('selected');
    window.MedMLApp.selectedModel = model;
}

async function startTraining(e) {
    e.preventDefault();
    
    if (!window.MedMLApp.selectedModel) {
        alert('Please select a model first');
        return;
    }
    
    const formData = new FormData(e.target);
    const config = {
        model_type: window.MedMLApp.selectedModel.name,
        epochs: parseInt(formData.get('epochs')),
        batch_size: parseInt(formData.get('batch_size')),
        learning_rate: parseFloat(formData.get('learning_rate')),
        optimizer: formData.get('optimizer'),
        loss_function: 'cross_entropy'
    };
    
    try {
        const response = await apiCall('/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `dataset_path=/path/to/preprocessed/data&${new URLSearchParams(config)}`
        });
        
        window.MedMLApp.currentJobId = response.job_id;
        
        alert(`Training started successfully! Job ID: ${response.job_id}`);
        
        // Enable deployment section
        enableSection('deployment');
        
        // Refresh jobs list
        loadTrainingJobs();
        
    } catch (error) {
        alert(`Error starting training: ${error.message}`);
    }
}

async function loadTrainingJobs() {
    try {
        const jobs = await apiCall('/api/training/jobs');
        displayTrainingJobs(jobs);
    } catch (error) {
        console.error('Error loading training jobs:', error);
    }
}

function displayTrainingJobs(jobs) {
    const container = document.getElementById('training-jobs');
    if (!container) return;
    
    if (jobs.length === 0) {
        container.innerHTML = '<p>No training jobs found.</p>';
        return;
    }
    
    container.innerHTML = '';
    
    jobs.forEach(job => {
        const jobElement = document.createElement('div');
        jobElement.className = 'job-item';
        
        jobElement.innerHTML = `
            <div class="job-header">
                <div class="job-id">Job: ${job.job_id.substring(0, 8)}...</div>
                <div class="job-status ${job.status}">${job.status.toUpperCase()}</div>
            </div>
            <div class="job-details">
                <div><strong>Model:</strong> ${job.model_type}</div>
                <div><strong>Created:</strong> ${formatDate(job.created_at)}</div>
                ${job.completed_at ? `<div><strong>Completed:</strong> ${formatDate(job.completed_at)}</div>` : ''}
            </div>
            <div class="job-actions">
                <button onclick="viewJobMetrics('${job.job_id}')" class="btn btn-secondary">
                    <i class="fas fa-chart-line"></i> View Metrics
                </button>
                ${job.status === 'running' ? 
                    `<button onclick="stopTrainingJob('${job.job_id}')" class="btn btn-secondary">
                        <i class="fas fa-stop"></i> Stop
                    </button>` : ''
                }
            </div>
        `;
        
        container.appendChild(jobElement);
    });
}

async function viewJobMetrics(jobId) {
    try {
        const metrics = await apiCall(`/api/training/jobs/${jobId}/metrics`);
        
        const metricsWindow = window.open('', '_blank', 'width=800,height=600');
        metricsWindow.document.write(`
            <html>
                <head>
                    <title>Training Metrics - ${jobId}</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
                        .links { margin-top: 20px; }
                        .links a { display: inline-block; margin-right: 10px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
                    </style>
                </head>
                <body>
                    <h1>Training Metrics</h1>
                    <h2>Job ID: ${jobId}</h2>
                    <div class="metric"><strong>Epoch:</strong> ${metrics.metrics.epoch}</div>
                    <div class="metric"><strong>Training Loss:</strong> ${metrics.metrics.train_loss}</div>
                    <div class="metric"><strong>Validation Loss:</strong> ${metrics.metrics.val_loss}</div>
                    <div class="metric"><strong>Training Accuracy:</strong> ${(metrics.metrics.train_accuracy * 100).toFixed(2)}%</div>
                    <div class="metric"><strong>Validation Accuracy:</strong> ${(metrics.metrics.val_accuracy * 100).toFixed(2)}%</div>
                    <div class="metric"><strong>Learning Rate:</strong> ${metrics.metrics.learning_rate}</div>
                    
                    <div class="links">
                        <a href="${metrics.metrics.tensorboard_url}" target="_blank">Open TensorBoard</a>
                        <a href="${metrics.metrics.mlflow_url}" target="_blank">Open MLflow</a>
                    </div>
                </body>
            </html>
        `);
        
    } catch (error) {
        alert(`Error loading metrics: ${error.message}`);
    }
}

async function stopTrainingJob(jobId) {
    if (!confirm('Are you sure you want to stop this training job?')) {
        return;
    }
    
    try {
        await apiCall(`/api/training/jobs/${jobId}/stop`, {
            method: 'POST'
        });
        
        alert('Training job stopped successfully');
        loadTrainingJobs();
        
    } catch (error) {
        alert(`Error stopping training job: ${error.message}`);
    }
}

// --- Model Import ---

async function importModel(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const payload = {
        model_name: formData.get('model_name'),
        s3_bucket: formData.get('s3_bucket'),
        s3_key: formData.get('s3_key'),
        model_format: formData.get('model_format') || 'pytorch',
        num_classes: parseInt(formData.get('num_classes') || '2')
    };

    try {
        const response = await apiCall('/api/training/models/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        alert(`Model imported successfully! ID: ${response.model_id}`);
        e.target.reset();
        loadImportedModels();
    } catch (error) {
        alert(`Error importing model: ${error.message}`);
    }
}

async function loadImportedModels() {
    try {
        const response = await apiCall('/api/training/models/imported');
        displayImportedModels(response.models);
    } catch (error) {
        console.error('Error loading imported models:', error);
    }
}

function displayImportedModels(models) {
    const container = document.getElementById('imported-models-list');
    if (!container) return;

    if (!models || models.length === 0) {
        container.innerHTML = '<p>No imported models yet.</p>';
        return;
    }

    container.innerHTML = '';
    models.forEach(model => {
        const el = document.createElement('div');
        el.className = 'model-item';
        el.dataset.modelId = model.model_id;
        el.innerHTML = `
            <div class="model-name">${model.model_name}</div>
            <div class="model-description">Format: ${model.model_format} | Classes: ${model.num_classes}</div>
            <div class="model-specs">
                <span>Source: ${model.s3_source}</span>
                <span>Status: ${model.status}</span>
            </div>
        `;
        el.addEventListener('click', () => {
            document.querySelectorAll('.model-item').forEach(i => i.classList.remove('selected'));
            el.classList.add('selected');
            window.MedMLApp.selectedModel = { name: 'custom', imported_model_id: model.model_id };
        });
        container.appendChild(el);
    });
}


// --- SageMaker BYOC Training ---

async function startSageMakerTraining(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const payload = {
        model_name: fd.get('sm_model_name'),
        epochs: parseInt(fd.get('sm_epochs')),
        batch_size: parseInt(fd.get('sm_batch_size')),
        learning_rate: parseFloat(fd.get('sm_learning_rate')),
        num_classes: parseInt(fd.get('sm_num_classes')),
        val_interval: parseInt(fd.get('sm_val_interval') || '1'),
        instance_type: fd.get('sm_instance_type'),
        instance_count: parseInt(fd.get('sm_instance_count') || '1'),
        image_uri: fd.get('sm_image_uri'),
        train_s3_uri: fd.get('sm_train_s3_uri'),
        test_s3_uri: fd.get('sm_test_s3_uri'),
        role_arn: fd.get('sm_role_arn'),
        entry_point: fd.get('sm_entry_point') || 'train_mlflow.py',
        source_dir: fd.get('sm_source_dir') || 'src',
        mlflow_tracking_uri: fd.get('sm_mlflow_tracking_uri') || null,
    };

    try {
        const response = await apiCall('/api/training/sagemaker/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        alert(`SageMaker training started! Job: ${response.job_name}`);
        loadSageMakerJobs();
    } catch (error) {
        alert(`Error starting SageMaker training: ${error.message}`);
    }
}

async function loadSageMakerJobs() {
    try {
        const response = await apiCall('/api/training/sagemaker/jobs');
        displaySageMakerJobs(response.jobs);
    } catch (error) {
        console.error('Error loading SageMaker jobs:', error);
    }
}

function displaySageMakerJobs(jobs) {
    const container = document.getElementById('sagemaker-jobs');
    if (!container) return;

    if (!jobs || jobs.length === 0) {
        container.innerHTML = '<p>No SageMaker training jobs.</p>';
        return;
    }

    container.innerHTML = '';
    jobs.forEach(job => {
        const el = document.createElement('div');
        el.className = 'job-item';
        el.innerHTML = `
            <div class="job-header">
                <div class="job-id">Job: ${job.job_name}</div>
                <div class="job-status ${job.status}">${job.status}</div>
            </div>
            <div class="job-details">
                <div><strong>Model:</strong> ${job.config?.model_name || 'N/A'}</div>
                <div><strong>Instance:</strong> ${job.config?.instance_type || 'N/A'}</div>
            </div>
            <div class="job-actions">
                <button onclick="checkSageMakerStatus('${job.job_name}')" class="btn btn-secondary">
                    <i class="fas fa-sync"></i> Refresh Status
                </button>
                <button onclick="stopSageMakerJob('${job.job_name}')" class="btn btn-secondary">
                    <i class="fas fa-stop"></i> Stop
                </button>
            </div>
        `;
        container.appendChild(el);
    });
}

async function checkSageMakerStatus(jobName) {
    try {
        const status = await apiCall(`/api/training/sagemaker/jobs/${jobName}/status`);
        let msg = `Job: ${status.job_name}\nStatus: ${status.status}`;
        if (status.model_artifacts) msg += `\nArtifacts: ${status.model_artifacts}`;
        if (status.failure_reason) msg += `\nFailure: ${status.failure_reason}`;
        alert(msg);
        loadSageMakerJobs();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function stopSageMakerJob(jobName) {
    if (!confirm(`Stop SageMaker job ${jobName}?`)) return;
    try {
        await apiCall(`/api/training/sagemaker/jobs/${jobName}/stop`, { method: 'POST' });
        alert('SageMaker job stop requested');
        loadSageMakerJobs();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}
