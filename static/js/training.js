// Training JavaScript

document.addEventListener('DOMContentLoaded', function() {
    loadAvailableModels();
    loadTrainingJobs();
    loadImportedModels();
    loadSageMakerJobs();
    
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
        populateSageMakerModelDropdown(response.models);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function populateSageMakerModelDropdown(models) {
    const select = document.getElementById('sm_model_name');
    if (!select) return;
    select.innerHTML = models.map(m =>
        `<option value="${m.name}">${m.name.toUpperCase()}</option>`
    ).join('');
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

const MODEL_INFO = {
    resnet50: {
        blurb: "ResNet-50 is a 50-layer deep residual network that introduced skip connections to solve the vanishing gradient problem, enabling training of much deeper networks. It remains a strong baseline for medical image classification tasks due to its balance of accuracy and computational efficiency.",
        specs: { Parameters: "25.6M", "Input Size": "224×224", Pretrained: "ImageNet" },
        dimensions: ["2D"],
        papers: [
            { title: "Deep Residual Learning for Image Recognition", authors: "He et al., 2016", url: "https://arxiv.org/abs/1512.03385" },
            { title: "Identity Mappings in Deep Residual Networks", authors: "He et al., 2016", url: "https://arxiv.org/abs/1603.05027" },
        ]
    },
    vgg16: {
        blurb: "VGG-16 is a 16-layer convolutional network known for its simplicity — using only 3×3 convolutions throughout. Despite its large parameter count, it has been widely used as a feature extractor in medical imaging, particularly for transfer learning on small datasets.",
        specs: { Parameters: "138M", "Input Size": "224×224", Pretrained: "ImageNet" },
        dimensions: ["2D"],
        papers: [
            { title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", authors: "Simonyan & Zisserman, 2015", url: "https://arxiv.org/abs/1409.1556" },
            { title: "How transferable are features in deep neural networks?", authors: "Yosinski et al., 2014", url: "https://arxiv.org/abs/1411.1792" },
        ]
    },
    efficientnet: {
        blurb: "EfficientNet-B0 uses compound scaling to uniformly scale network depth, width, and resolution. It achieves state-of-the-art accuracy with significantly fewer parameters than comparable models, making it well-suited for resource-constrained medical imaging deployments.",
        specs: { Parameters: "5.3M", "Input Size": "224×224", Pretrained: "ImageNet" },
        dimensions: ["2D"],
        papers: [
            { title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", authors: "Tan & Le, 2019", url: "https://arxiv.org/abs/1905.11946" },
            { title: "EfficientNetV2: Smaller Models and Faster Training", authors: "Tan & Le, 2021", url: "https://arxiv.org/abs/2104.00298" },
        ]
    },
    DenseNet121: {
        blurb: "DenseNet-121 connects each layer to every other layer in a feed-forward fashion, encouraging feature reuse and reducing the number of parameters. It has become a standard architecture in medical imaging, most notably used in CheXNet for chest X-ray pathology detection.",
        specs: { Parameters: "8M", "Input Size": "224×224", Pretrained: "ImageNet" },
        dimensions: ["2D", "3D"],
        papers: [
            { title: "Densely Connected Convolutional Networks", authors: "Huang et al., 2017", url: "https://arxiv.org/abs/1608.06993" },
            { title: "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays", authors: "Rajpurkar et al., 2017", url: "https://arxiv.org/abs/1711.05225" },
        ]
    },
    ViT: {
        blurb: "Vision Transformer (ViT) applies the transformer architecture — originally designed for NLP — directly to image patches. It excels on large datasets and has shown strong performance in medical imaging tasks such as histopathology and radiology when fine-tuned from large pretrained checkpoints.",
        specs: { Parameters: "86M (ViT-B/16)", "Input Size": "224×224", Pretrained: "ImageNet-21k" },
        dimensions: ["2D", "3D"],
        papers: [
            { title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", authors: "Dosovitskiy et al., 2021", url: "https://arxiv.org/abs/2010.11929" },
            { title: "Transformers in Medical Imaging: A Survey", authors: "Shamshad et al., 2023", url: "https://arxiv.org/abs/2201.09873" },
        ]
    },
    custom: {
        blurb: "Upload your own model architecture and weights. Custom models can be imported from S3 in PyTorch, ONNX, or TensorFlow formats and registered for training or deployment.",
        specs: { Parameters: "Variable", "Input Size": "Configurable", Pretrained: "User-defined" },
        dimensions: ["2D", "3D"],
        papers: []
    }
};

function selectModel(model, element) {
    document.querySelectorAll('.model-item').forEach(item => item.classList.remove('selected'));
    element.classList.add('selected');
    window.MedMLApp.selectedModel = model;
    showModelInfo(model);
}

function showModelInfo(model) {
    const key = Object.keys(MODEL_INFO).find(
        k => k.toLowerCase() === (model.name || '').toLowerCase()
    );
    const info = key ? MODEL_INFO[key] : null;

    const placeholder = document.getElementById('model-info-placeholder');
    const content = document.getElementById('model-info-content');

    if (!info) {
        console.warn('No MODEL_INFO entry for model name:', model.name);
        return;
    }

    placeholder.style.display = 'none';
    content.style.display = 'block';

    document.getElementById('model-info-title').innerHTML = `<i class="fas fa-info-circle"></i> ${model.name.toUpperCase()}`;
    document.getElementById('model-info-blurb').textContent = info.blurb;

    const meta = document.getElementById('model-info-meta');
    meta.innerHTML = Object.entries(info.specs).map(([k, v]) =>
        `<span class="model-meta-tag"><strong>${k}:</strong> ${v}</span>`
    ).join('');

    const dimBadges = (info.dimensions || []).map(d =>
        `<span class="model-dim-badge model-dim-${d.toLowerCase()}">${d}</span>`
    ).join('');
    meta.innerHTML += `<div class="model-dim-row">${dimBadges}</div>`;

    const papersList = document.getElementById('model-papers-list');
    if (info.papers.length === 0) {
        papersList.innerHTML = '<li>No papers listed for custom models.</li>';
    } else {
        papersList.innerHTML = info.papers.map(p =>
            `<li><a href="${p.url}" target="_blank" rel="noopener">
                <i class="fas fa-external-link-alt"></i> ${p.title}
            </a><span class="paper-authors">${p.authors}</span></li>`
        ).join('');
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

async function validateSageMakerConfig() {
    const form = document.getElementById('sagemaker-training-form');
    const fd = new FormData(form);
    const btn = document.getElementById('sm-validate-btn');
    const resultsEl = document.getElementById('sm-validation-results');

    // Build same payload as startSageMakerTraining
    const payload = {
        model_name: fd.get('sm_model_name') || 'densenet121',
        image_uri: fd.get('sm_image_uri') || '',
        data_s3_uri: fd.get('sm_data_s3_uri') || '',
        role_arn: fd.get('sm_role_arn') || '',
        task_type: fd.get('sm_task_type') || 'classification',
        dimension: fd.get('sm_dimension') || '2D',
        epochs: parseInt(fd.get('sm_epochs') || '1'),
        batch_size: parseInt(fd.get('sm_batch_size') || '16'),
        learning_rate: parseFloat(fd.get('sm_learning_rate') || '0.001'),
        num_classes: parseInt(fd.get('sm_num_classes') || '8'),
        val_interval: 1,
        instance_type: fd.get('sm_instance_type') || 'ml.g5.xlarge',
        instance_count: 1,
        entry_point: fd.get('sm_entry_point') || 'train.py',
        source_dir: 'src/' + (fd.get('sm_model_name') || 'densenet121'),
        mlflow_tracking_uri: fd.get('sm_mlflow_tracking_uri') || null,
    };

    btn.disabled = true;
    btn.innerHTML = '<span class="btn-spinner"></span> Validating...';
    resultsEl.style.display = 'none';

    try {
        const result = await apiCall('/api/training/sagemaker/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        renderValidationResults(result, resultsEl);
    } catch (error) {
        resultsEl.style.display = 'block';
        resultsEl.innerHTML = `<div class="validation-panel validation-failed">
            <h4><i class="fas fa-times-circle"></i> Validation Error</h4>
            <p>${error.message}</p>
        </div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-check-circle"></i> Validate';
    }
}

function renderValidationResults(result, container) {
    const overallClass = result.passed ? 'validation-passed' : 'validation-failed';
    const overallIcon = result.passed ? 'fa-check-circle' : 'fa-times-circle';
    const overallText = result.passed ? 'All checks passed — ready to train' : 'Validation failed — fix issues before training';

    let checksHtml = (result.checks || []).map(c => `
        <div class="validation-check ${c.passed ? 'check-ok' : 'check-fail'}">
            <i class="fas ${c.passed ? 'fa-check' : 'fa-times'}"></i>
            <div>
                <strong>${c.name}</strong>
                <span>${c.message}</span>
            </div>
        </div>
    `).join('');

    let statsHtml = '';
    const stats = result.statistics || {};
    if (stats.splits) {
        statsHtml = '<div class="validation-stats"><h5><i class="fas fa-chart-bar"></i> Dataset Statistics</h5>';
        for (const [split, s] of Object.entries(stats.splits)) {
            statsHtml += `<div class="stat-split">
                <strong>${split}</strong> — ${s.num_classes} classes, ${s.total_files} files, ${s.total_size_mb} MB
                <div class="stat-classes">${Object.entries(s.classes).map(([cls, n]) =>
                    `<span class="stat-class-tag">${cls}: ${n}</span>`).join('')}
                </div>
            </div>`;
        }
        statsHtml += '</div>';
    } else if (stats.total_files !== undefined) {
        statsHtml = `<div class="validation-stats"><h5><i class="fas fa-chart-bar"></i> Dataset Statistics</h5>
            <p>${stats.total_files} files &nbsp;|&nbsp; ${stats.total_size_mb} MB</p>
        </div>`;
    }

    container.style.display = 'block';
    container.innerHTML = `
        <div class="validation-panel ${overallClass}">
            <h4><i class="fas ${overallIcon}"></i> ${overallText}</h4>
            <div class="validation-checks">${checksHtml}</div>
            ${statsHtml}
        </div>`;
}

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
        data_s3_uri: fd.get('sm_data_s3_uri'),
        role_arn: fd.get('sm_role_arn'),
        task_type: fd.get('sm_task_type'),
        dimension: fd.get('sm_dimension'),
        entry_point: fd.get('sm_entry_point') || 'train.py',
        source_dir: fd.get('sm_source_dir') || 'src/densenet121',
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
