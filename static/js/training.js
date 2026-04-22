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

    // Poll SageMaker jobs every 15s, stop when all are in terminal state
    const TERMINAL = new Set(["Completed", "Failed", "Stopped"]);
    const smPollInterval = setInterval(async () => {
        const response = await apiCall('/api/training/sagemaker/jobs').catch(() => null);
        if (!response) return;
        displaySageMakerJobs(response.jobs);
        const allDone = response.jobs.length > 0 &&
            response.jobs.every(j => TERMINAL.has(j.status));
        if (allDone) clearInterval(smPollInterval);
    }, 15000);
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

function displayModels(models) {
    const container = document.getElementById('models-list');
    if (!container) return;

    container.innerHTML = '';

    models.forEach(model => {
        const task = (model.task || 'classification').toLowerCase();
        const taskClass = task === 'segmentation' ? 'model-tag-segmentation' : 'model-tag-classification';
        const taskLabel = task === 'segmentation' ? 'Segmentation' : 'Classification';

        const modelElement = document.createElement('div');
        modelElement.className = `model-item model-item-${task}`;
        modelElement.dataset.modelType = model.name;

        modelElement.innerHTML = `
            <div class="model-item-header">
                <div class="model-name">${getDisplayName(model.name)}</div>
                <span class="model-tag ${taskClass}">${taskLabel}</span>
            </div>
            <div class="model-description">${model.description}</div>
            <div class="model-specs">
                <span>Parameters: ${model.parameters}</span>
                <span>Input: ${model.input_size}</span>
            </div>
        `;

        modelElement.addEventListener('click', () => selectModel(model, modelElement));
        container.appendChild(modelElement);
    });
}

function populateSageMakerModelDropdown(models) {
    const select = document.getElementById('sm_model_name');
    if (!select) return;
    select.innerHTML = models.map(m => {
        const task = (m.task || 'classification').toLowerCase();
        const label = task === 'segmentation' ? '🔬' : '🏷️';
        return `<option value="${m.name}" data-task="${task}">${label} ${getDisplayName(m.name)} (${task})</option>`;
    }).join('');
}

const MODEL_INFO = {
    resnet50: {
        blurb: "ResNet-50 is a 50-layer deep residual network that introduced skip connections to solve the vanishing gradient problem, enabling training of much deeper networks. It remains a strong baseline for medical image classification tasks due to its balance of accuracy and computational efficiency.",
        specs: { Parameters: "25.6M", "Input Size": "224×224" },
        dimensions: ["2D"],
        papers: [
            { title: "Deep Residual Learning for Image Recognition", authors: "He et al., 2016", url: "https://arxiv.org/abs/1512.03385" },
            { title: "Identity Mappings in Deep Residual Networks", authors: "He et al., 2016", url: "https://arxiv.org/abs/1603.05027" },
        ]
    },
    vgg16: {
        blurb: "VGG-16 is a 16-layer convolutional network known for its simplicity — using only 3×3 convolutions throughout. Despite its large parameter count, it has been widely used as a feature extractor in medical imaging, particularly for transfer learning on small datasets.",
        specs: { Parameters: "138M", "Input Size": "224×224" },
        dimensions: ["2D"],
        papers: [
            { title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", authors: "Simonyan & Zisserman, 2015", url: "https://arxiv.org/abs/1409.1556" },
            { title: "How transferable are features in deep neural networks?", authors: "Yosinski et al., 2014", url: "https://arxiv.org/abs/1411.1792" },
        ]
    },
    efficientnet: {
        blurb: "EfficientNet-B0 uses compound scaling to uniformly scale network depth, width, and resolution. It achieves state-of-the-art accuracy with significantly fewer parameters than comparable models, making it well-suited for resource-constrained medical imaging deployments.",
        specs: { Parameters: "5.3M", "Input Size": "224×224" },
        dimensions: ["2D"],
        papers: [
            { title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", authors: "Tan & Le, 2019", url: "https://arxiv.org/abs/1905.11946" },
            { title: "EfficientNetV2: Smaller Models and Faster Training", authors: "Tan & Le, 2021", url: "https://arxiv.org/abs/2104.00298" },
        ]
    },
    DenseNet121: {
        blurb: "DenseNet-121 connects each layer to every other layer in a feed-forward fashion, encouraging feature reuse and reducing the number of parameters. It has become a standard architecture in medical imaging, most notably used in CheXNet for chest X-ray pathology detection.",
        specs: { Parameters: "8M", "Input Size": "224×224" },
        dimensions: ["2D", "3D"],
        papers: [
            { title: "Densely Connected Convolutional Networks", authors: "Huang et al., 2017", url: "https://arxiv.org/abs/1608.06993" },
            { title: "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays", authors: "Rajpurkar et al., 2017", url: "https://arxiv.org/abs/1711.05225" },
        ]
    },
    ViT: {
        blurb: "Vision Transformer (ViT) applies the transformer architecture — originally designed for NLP — directly to image patches. It excels on large datasets and has shown strong performance in medical imaging tasks such as histopathology and radiology when fine-tuned from large pretrained checkpoints.",
        specs: { Parameters: "86M (ViT-B/16)", "Input Size": "224×224" },
        dimensions: ["2D", "3D"],
        papers: [
            { title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", authors: "Dosovitskiy et al., 2021", url: "https://arxiv.org/abs/2010.11929" },
            { title: "Transformers in Medical Imaging: A Survey", authors: "Shamshad et al., 2023", url: "https://arxiv.org/abs/2201.09873" },
        ]
    },
    custom: {
        blurb: "Upload your own model architecture and weights. Custom models can be imported from S3 in PyTorch, ONNX, or TensorFlow formats and registered for training or deployment.",
        specs: { Parameters: "Variable", "Input Size": "Configurable" },
        dimensions: ["2D", "3D"],
        papers: []
    },
    segresnet: {
        blurb: "SegResNet is a residual encoder-decoder network designed for volumetric medical image segmentation. It uses residual blocks throughout with deep supervision, achieving strong Dice scores with a compact ~5M parameter footprint — ideal for prototyping and quick iterations on 3D CT/MRI data.",
        specs: { Parameters: "5M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        papers: [
            { title: "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization", authors: "Myronenko, 2018", url: "https://arxiv.org/abs/1810.11654" },
        ]
    },
    swinunetr: {
        blurb: "SwinUNETR combines Swin Transformers with a U-shaped encoder-decoder for 3D medical image segmentation. It captures long-range dependencies via shifted window attention, achieving state-of-the-art results on volumetric tasks at the cost of higher GPU memory (~20GB).",
        specs: { Parameters: "62M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        papers: [
            { title: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", authors: "Hatamizadeh et al., 2022", url: "https://arxiv.org/abs/2201.01266" },
            { title: "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis", authors: "Tang et al., 2022", url: "https://arxiv.org/abs/2111.14791" },
        ]
    },
    unet: {
        blurb: "UNet is the classic encoder-decoder architecture with skip connections for medical image segmentation. The 3D variant extends this to volumetric data and remains a strong, well-understood baseline for organ and lesion segmentation tasks.",
        specs: { Parameters: "31M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        papers: [
            { title: "U-Net: Convolutional Networks for Biomedical Image Segmentation", authors: "Ronneberger et al., 2015", url: "https://arxiv.org/abs/1505.04597" },
            { title: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", authors: "Çiçek et al., 2016", url: "https://arxiv.org/abs/1606.06650" },
        ]
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
        // Fallback — show basic info from the model object itself
        placeholder.style.display = 'none';
        content.style.display = 'block';
        document.getElementById('model-info-title').innerHTML = `<i class="fas fa-info-circle"></i> ${getDisplayName(model.name)}`;
        document.getElementById('model-info-blurb').textContent = model.description || 'No description available.';
        const meta = document.getElementById('model-info-meta');
        meta.innerHTML = [
            model.parameters ? `<span class="model-meta-tag"><strong>Parameters:</strong> ${model.parameters}</span>` : '',
            model.input_size  ? `<span class="model-meta-tag"><strong>Input:</strong> ${model.input_size}</span>` : '',
            model.task        ? `<span class="model-meta-tag"><strong>Task:</strong> ${model.task}</span>` : '',
        ].join('');
        document.getElementById('model-papers-list').innerHTML = '<li>No papers listed.</li>';
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
    resultsEl.style.display = 'block';
    resultsEl.innerHTML = `
        <div class="validation-loading">
            <div class="walker-scene">
                <div class="walker-track">
                    <div class="walker-track-fill" id="walker-track-fill"></div>
                </div>
                <div class="walker" id="walker-figure">
                    <div class="walker-figure">
                        <div class="walker-head"></div>
                        <div class="walker-body"></div>
                        <div class="walker-arm-left"></div>
                        <div class="walker-arm-right"></div>
                        <div class="walker-leg-left"></div>
                        <div class="walker-leg-right"></div>
                    </div>
                </div>
            </div>
            <p>Validating your configuration...</p>
        </div>`;

    // Animate progress bar while waiting
    let progress = 0;
    const progressInterval = setInterval(() => {
        // Ease toward 90% but never reach 100% until done
        progress += (90 - progress) * 0.03;
        const fill = document.getElementById('walker-track-fill');
        const walker = document.getElementById('walker-figure');
        if (fill) fill.style.width = progress + '%';
        if (walker) walker.style.left = `calc(${progress}% - 18px)`;
    }, 80);

    try {
        const result = await apiCall('/api/training/sagemaker/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        clearInterval(progressInterval);
        // Snap to 100% before rendering results
        const fill = document.getElementById('walker-track-fill');
        const walker = document.getElementById('walker-figure');
        if (fill) fill.style.width = '100%';
        if (walker) walker.style.left = 'calc(100% - 18px)';
        setTimeout(() => renderValidationResults(result, resultsEl), 300);
    } catch (error) {
        clearInterval(progressInterval);
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
        const chartId = 'validation-dist-chart';
        statsHtml = `
            <div class="validation-stats">
                <h5><i class="fas fa-chart-bar"></i> Dataset Statistics</h5>
                <div class="validation-chart-wrap">
                    <canvas id="${chartId}"></canvas>
                </div>
            </div>`;

        container.style.display = 'block';
        container.innerHTML = `
            <div class="validation-panel ${overallClass}">
                <h4><i class="fas ${overallIcon}"></i> ${overallText}</h4>
                <div class="validation-checks">${checksHtml}</div>
                ${statsHtml}
            </div>`;

        // Build chart data
        const splits = Object.keys(stats.splits);
        const allClasses = [...new Set(splits.flatMap(s => Object.keys(stats.splits[s].classes || {})))].sort();
        const colors = [
            '#4B9CD3', '#FF9900', '#16A085', '#E74C3C',
            '#8E44AD', '#2ECC71', '#F39C12', '#1ABC9C',
            '#E67E22', '#3498DB', '#E91E63', '#00BCD4'
        ];

        const datasets = splits.map((split, i) => ({
            label: split,
            data: allClasses.map(cls => stats.splits[split].classes?.[cls] || 0),
            backgroundColor: colors[i % colors.length] + 'cc',
            borderColor: colors[i % colors.length],
            borderWidth: 1,
        }));

        new Chart(document.getElementById(chartId), {
            type: 'bar',
            data: { labels: allClasses, datasets },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    title: {
                        display: true,
                        text: 'Class Distribution per Split',
                        font: { size: 14 }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Class' } },
                    y: { title: { display: true, text: 'Number of Files' }, beginAtZero: true }
                }
            }
        });
        return;

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
        val_interval: parseInt(fd.get('sm_val_interval') || '1'),
        instance_type: fd.get('sm_instance_type'),
        instance_count: parseInt(fd.get('sm_instance_count') || '1'),
        image_uri: fd.get('sm_image_uri'),
        data_s3_uri: fd.get('sm_data_s3_uri'),
        role_arn: fd.get('sm_role_arn'),
        task_type: fd.get('sm_task_type'),
        dimension: fd.get('sm_dimension'),
        entry_point: fd.get('sm_entry_point') || 'train.py',
        source_dir: fd.get('sm_source_dir') || ('src/' + (fd.get('sm_task_type') || 'classification')),
        mlflow_tracking_uri: fd.get('sm_mlflow_tracking_uri') || null,
        output_s3_uri: fd.get('sm_output_s3_uri') || null,
    };

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const statusBanner = document.getElementById('sm-submission-status');

    // Show submitting state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    if (statusBanner) {
        statusBanner.className = 'sm-status-banner sm-status-submitting';
        statusBanner.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting training job to SageMaker...';
        statusBanner.style.display = 'flex';
    }

    try {
        const response = await apiCall('/api/training/sagemaker/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (statusBanner) {
            statusBanner.className = 'sm-status-banner sm-status-success';
            statusBanner.innerHTML = `<i class="fas fa-check-circle"></i> Job submitted: <strong>${response.job_name}</strong> — Status: <strong>${response.status || 'InProgress'}</strong>`;
        }
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Start SageMaker Training';
        loadSageMakerJobs();
    } catch (error) {
        if (statusBanner) {
            statusBanner.className = 'sm-status-banner sm-status-error';
            statusBanner.innerHTML = `<i class="fas fa-exclamation-circle"></i> Failed: ${error.message}`;
        }
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Start SageMaker Training';
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
        const statusClass = {
            'InProgress': 'status-running',
            'Completed': 'status-completed',
            'Failed': 'status-failed',
            'Stopping': 'status-stopping',
            'Stopped': 'status-stopped',
        }[job.status] || 'status-unknown';

        const el = document.createElement('div');
        el.className = 'job-item';
        el.id = `sm-job-${job.job_name}`;
        el.innerHTML = `
            <div class="job-header">
                <div class="job-id">Job: ${job.job_name}</div>
                <div class="job-status ${statusClass}">${job.status}</div>
            </div>
            <div class="job-details">
                <div><strong>Model:</strong> ${job.config?.model_name || 'N/A'}</div>
                <div><strong>Instance:</strong> ${job.config?.instance_type || 'N/A'}</div>
            </div>
            <div id="sm-job-status-detail-${job.job_name}" class="job-status-detail" style="display:none;"></div>
            <div class="job-actions">
                <button onclick="checkSageMakerStatus('${job.job_name}')" class="btn btn-secondary" id="sm-status-btn-${job.job_name}" ${['Completed','Failed','Stopped'].includes(job.status) ? 'style="display:none"' : ''}>
                    <i class="fas fa-sync"></i> Fetch Status
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
    const btn = document.getElementById(`sm-status-btn-${jobName}`);
    const detail = document.getElementById(`sm-job-status-detail-${jobName}`);
    const jobEl = document.getElementById(`sm-job-${jobName}`);

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching...';
    }

    try {
        const status = await apiCall(`/api/training/sagemaker/jobs/${jobName}/status`);

        // Update status badge
        if (jobEl) {
            const badge = jobEl.querySelector('.job-status');
            const statusClass = {
                'InProgress': 'status-running',
                'Completed': 'status-completed',
                'Failed': 'status-failed',
                'Stopping': 'status-stopping',
                'Stopped': 'status-stopped',
            }[status.status] || 'status-unknown';
            if (badge) { badge.className = `job-status ${statusClass}`; badge.textContent = status.status; }
        }

        // Show detail panel
        if (detail) {
            let html = `<div class="status-detail-row"><span>Status:</span><strong>${status.status}</strong></div>`;
            if (status.creation_time) html += `<div class="status-detail-row"><span>Created:</span><strong>${status.creation_time}</strong></div>`;
            if (status.training_end_time) html += `<div class="status-detail-row"><span>Ended:</span><strong>${status.training_end_time}</strong></div>`;
            if (status.billable_seconds) html += `<div class="status-detail-row"><span>Billable seconds:</span><strong>${status.billable_seconds}s</strong></div>`;
            if (status.model_artifacts) html += `<div class="status-detail-row"><span>Artifacts:</span><strong>${status.model_artifacts}</strong></div>`;
            if (status.failure_reason) html += `<div class="status-detail-row status-error"><span>Failure:</span><strong>${status.failure_reason}</strong></div>`;
            detail.innerHTML = html;
            detail.style.display = 'block';
        }
    } catch (error) {
        if (detail) {
            detail.innerHTML = `<div class="status-detail-row status-error"><span>Error:</span><strong>${error.message}</strong></div>`;
            detail.style.display = 'block';
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sync"></i> Fetch Status';
        }
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

// ── Container Image Management ────────────────────────────────────────────────

const DEFAULT_CONTAINERS = [
    { label: "classification-latest", uri: "575108919340.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest" },
    { label: "segmentation-latest",   uri: "575108919340.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest" },
];

function loadContainers() {
    const stored = localStorage.getItem('sm_containers');
    return stored ? JSON.parse(stored) : DEFAULT_CONTAINERS;
}

function saveContainers(containers) {
    localStorage.setItem('sm_containers', JSON.stringify(containers));
}

function populateContainerDropdown() {
    const select = document.getElementById('sm_image_uri');
    if (!select) return;
    const containers = loadContainers();
    select.innerHTML = containers.map(c =>
        `<option value="${c.uri}">${c.label} — ${c.uri}</option>`
    ).join('');
}

function openAddContainerModal() {
    document.getElementById('new-container-label').value = '';
    document.getElementById('new-container-uri').value = '';
    const modal = document.getElementById('add-container-modal');
    modal.style.display = 'flex';
}

function closeAddContainerModal() {
    document.getElementById('add-container-modal').style.display = 'none';
}

function saveNewContainer() {
    const label = document.getElementById('new-container-label').value.trim();
    const uri   = document.getElementById('new-container-uri').value.trim();
    if (!uri) { alert('Image URI is required.'); return; }
    const containers = loadContainers();
    containers.push({ label: label || uri, uri });
    saveContainers(containers);
    populateContainerDropdown();
    // Select the newly added one
    const select = document.getElementById('sm_image_uri');
    select.value = uri;
    closeAddContainerModal();
}

function removeSelectedContainer() {
    const select = document.getElementById('sm_image_uri');
    if (!select || !select.value) return;
    if (!confirm(`Remove "${select.options[select.selectedIndex].text}"?`)) return;
    const containers = loadContainers().filter(c => c.uri !== select.value);
    saveContainers(containers);
    populateContainerDropdown();
}

// Initialise on load
document.addEventListener('DOMContentLoaded', function() {
    populateContainerDropdown();
});
