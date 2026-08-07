// Training JavaScript
// Depends on: escapeHtml() defined in main.js (loaded before this file)
if (typeof escapeHtml !== 'function') {
    throw new Error('escapeHtml() is required but not defined. Ensure main.js is loaded before training.js.');
}

registerTabInit('training', function() {
    loadAvailableModels();
    populateContainerDropdown();

    const smForm = document.getElementById('sagemaker-training-form');
    if (smForm) smForm.addEventListener('submit', startSageMakerTraining);

    document.querySelectorAll('input[name="sm_task_type"]')
        .forEach(r => r.addEventListener('change', onTaskTypeChange));

    // Refresh jobs every 15s for active ones
    setInterval(async () => {
        const r = await apiCall('/api/training/sagemaker/jobs').catch(() => null);
        if (r) displaySageMakerJobs(r.jobs);
    }, 15000);
});

document.addEventListener('DOMContentLoaded', function() {
    const importForm = document.getElementById('model-import-form');
    if (importForm) importForm.addEventListener('submit', importModel);
});

async function loadAvailableModels() {
    try {
        const response = await apiCall('/api/training/models');
        const apiModels    = response.models || [];
        const customModels = getCustomModels();
        // Merge: custom models appear after built-in ones, marked with a badge
        const allModels = [...apiModels, ...customModels];
        displayModels(allModels);
        populateSageMakerModelDropdown(allModels);
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
        const isCustom  = !!model.custom;

        const modelElement = document.createElement('div');
        modelElement.className = `model-item model-item-${task}`;
        modelElement.dataset.modelType = model.name;

        modelElement.innerHTML = `
            <div class="model-item-header">
                <div class="model-name">${escapeHtml(getDisplayName(model.name))}</div>
                <div style="display:flex; gap:4px; align-items:center;">
                    ${isCustom ? '<span class="model-tag" style="background:#f3eeff; color:#7b2ff7; border:1px solid #c9b0f5;">Custom</span>' : ''}
                    <span class="model-tag ${taskClass}">${taskLabel}</span>
                    ${isCustom ? `<button type="button" title="Remove" onclick="removeCustomModel('${escapeJsString(model.name)}', event)"
                        style="background:none; border:none; cursor:pointer; color:#c62828; font-size:0.75rem; padding:0 2px;">
                        <i class="fas fa-times"></i></button>` : ''}
                </div>
            </div>
            <div class="model-description">${escapeHtml(model.description)}</div>
            <div class="model-specs">
                <span>Parameters: ${escapeHtml(model.parameters)}</span>
                <span>Input: ${escapeHtml(model.input_size)}</span>
            </div>
        `;

        modelElement.addEventListener('click', () => selectModel(model, modelElement));
        container.appendChild(modelElement);
    });
}

function removeCustomModel(name, event) {
    event.stopPropagation();
    if (!confirm(`Remove custom model "${name}"?`)) return;
    const updated = loadCustomModels().filter(m => m.name !== name);
    saveCustomModels(updated);
    loadAvailableModels();
}

function populateSageMakerModelDropdown(models) {
    const select = document.getElementById('sm_model_name');
    if (!select) return;
    select.innerHTML = models.map(m => {
        const task = (m.task || 'classification').toLowerCase();
        const label = task === 'segmentation' ? '🔬' : '🏷️';
        return `<option value="${escapeHtml(m.name)}" data-task="${escapeHtml(task)}">${label} ${escapeHtml(getDisplayName(m.name))} (${escapeHtml(task)})</option>`;
    }).join('');
}

// Data folder structures per task type
const FOLDER_STRUCTURE = {
    classification: {
        label: "s3://your-bucket/dataset/",
        children: [
            { label: "train/", children: [
                { label: "class_a/", note: "e.g. No_finding", children: [{ label: "*.dcm / *.png / *.jpg" }] },
                { label: "class_b/", note: "e.g. Pneumonia", children: [{ label: "*.dcm / *.png / *.jpg" }] },
                { label: "class_c/  …", children: [] },
            ]},
            { label: "val/", children: [
                { label: "class_a/", children: [{ label: "*.dcm / *.png / *.jpg" }] },
                { label: "class_b/  …", children: [] },
            ]},
            { label: "test/", note: "optional", children: [
                { label: "class_a/", children: [{ label: "*.dcm / *.png / *.jpg" }] },
                { label: "class_b/  …", children: [] },
            ]},
        ]
    },
    segmentation: {
        label: "s3://your-bucket/dataset/",
        children: [
            { label: "train/", children: [
                { label: "subject_001/", children: [
                    { label: "img.nii.gz" },
                    { label: "label.nii.gz" },
                ]},
                { label: "subject_002/  ...", children: [] },
            ]},
            { label: "val/", children: [
                { label: "subject_N/", children: [
                    { label: "img.nii.gz" },
                    { label: "label.nii.gz" },
                ]},
            ]},
        ]
    },
    custom: {
        label: "s3://your-bucket/",
        children: [
            { label: "<weights>.pt / .pth", note: "PyTorch checkpoint" },
            { label: "<weights>.onnx", note: "ONNX export (optional)" },
            { label: "<weights>.pb / .h5", note: "TensorFlow (optional)" },
        ]
    }
};

const MODEL_INFO = {
    resnet50: {
        blurb: "ResNet-50 is a 50-layer deep residual network that introduced skip connections to solve the vanishing gradient problem, enabling training of much deeper networks. It remains a strong baseline for medical image classification tasks due to its balance of accuracy and computational efficiency.",
        specs: { Parameters: "25.6M", "Input Size": "224×224" },
        dimensions: ["2D"],
        folderStructure: FOLDER_STRUCTURE.classification,
        papers: [
            { title: "Deep Residual Learning for Image Recognition", authors: "He et al., 2016", url: "https://arxiv.org/abs/1512.03385" },
            { title: "Identity Mappings in Deep Residual Networks", authors: "He et al., 2016", url: "https://arxiv.org/abs/1603.05027" },
        ]
    },
    vgg16: {
        blurb: "VGG-16 is a 16-layer convolutional network known for its simplicity — using only 3×3 convolutions throughout. Despite its large parameter count, it has been widely used as a feature extractor in medical imaging, particularly for transfer learning on small datasets.",
        specs: { Parameters: "138M", "Input Size": "224×224" },
        dimensions: ["2D"],
        folderStructure: FOLDER_STRUCTURE.classification,
        papers: [
            { title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", authors: "Simonyan & Zisserman, 2015", url: "https://arxiv.org/abs/1409.1556" },
            { title: "How transferable are features in deep neural networks?", authors: "Yosinski et al., 2014", url: "https://arxiv.org/abs/1411.1792" },
        ]
    },
    efficientnet: {
        blurb: "EfficientNet-B0 uses compound scaling to uniformly scale network depth, width, and resolution. It achieves state-of-the-art accuracy with significantly fewer parameters than comparable models, making it well-suited for resource-constrained medical imaging deployments.",
        specs: { Parameters: "5.3M", "Input Size": "224×224" },
        dimensions: ["2D"],
        folderStructure: FOLDER_STRUCTURE.classification,
        papers: [
            { title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", authors: "Tan & Le, 2019", url: "https://arxiv.org/abs/1905.11946" },
            { title: "EfficientNetV2: Smaller Models and Faster Training", authors: "Tan & Le, 2021", url: "https://arxiv.org/abs/2104.00298" },
        ]
    },
    DenseNet121: {
        blurb: "DenseNet-121 connects each layer to every other layer in a feed-forward fashion, encouraging feature reuse and reducing the number of parameters. It has become a standard architecture in medical imaging, most notably used in CheXNet for chest X-ray pathology detection.",
        specs: { Parameters: "8M", "Input Size": "224×224" },
        dimensions: ["2D", "3D"],
        folderStructure: FOLDER_STRUCTURE.classification,
        papers: [
            { title: "Densely Connected Convolutional Networks", authors: "Huang et al., 2017", url: "https://arxiv.org/abs/1608.06993" },
            { title: "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays", authors: "Rajpurkar et al., 2017", url: "https://arxiv.org/abs/1711.05225" },
        ]
    },
    ViT: {
        blurb: "Vision Transformer (ViT) applies the transformer architecture — originally designed for NLP — directly to image patches. It excels on large datasets and has shown strong performance in medical imaging tasks such as histopathology and radiology when fine-tuned from large pretrained checkpoints.",
        specs: { Parameters: "86M (ViT-B/16)", "Input Size": "224×224" },
        dimensions: ["2D", "3D"],
        folderStructure: FOLDER_STRUCTURE.classification,
        papers: [
            { title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", authors: "Dosovitskiy et al., 2021", url: "https://arxiv.org/abs/2010.11929" },
            { title: "Transformers in Medical Imaging: A Survey", authors: "Shamshad et al., 2023", url: "https://arxiv.org/abs/2201.09873" },
        ]
    },
    custom: {
        blurb: "Upload your own model architecture and weights. Custom models can be imported from S3 in PyTorch, ONNX, or TensorFlow formats and registered for training or deployment.",
        specs: { Parameters: "Variable", "Input Size": "Configurable" },
        dimensions: ["2D", "3D"],
        folderStructure: FOLDER_STRUCTURE.custom,
        papers: []
    },
    segresnet: {
        blurb: "SegResNet is a residual encoder-decoder network designed for volumetric medical image segmentation. It uses residual blocks throughout with deep supervision, achieving strong Dice scores with a compact ~5M parameter footprint — ideal for prototyping and quick iterations on 3D CT/MRI data.",
        specs: { Parameters: "5M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        folderStructure: FOLDER_STRUCTURE.segmentation,
        papers: [
            { title: "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization", authors: "Myronenko, 2018", url: "https://arxiv.org/abs/1810.11654" },
        ]
    },
    swinunetr: {
        blurb: "SwinUNETR combines Swin Transformers with a U-shaped encoder-decoder for 3D medical image segmentation. It captures long-range dependencies via shifted window attention, achieving state-of-the-art results on volumetric tasks at the cost of higher GPU memory (~20GB).",
        specs: { Parameters: "62M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        folderStructure: FOLDER_STRUCTURE.segmentation,
        papers: [
            { title: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", authors: "Hatamizadeh et al., 2022", url: "https://arxiv.org/abs/2201.01266" },
            { title: "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis", authors: "Tang et al., 2022", url: "https://arxiv.org/abs/2111.14791" },
        ]
    },
    unet: {
        blurb: "UNet is the classic encoder-decoder architecture with skip connections for medical image segmentation. The 3D variant extends this to volumetric data and remains a strong, well-understood baseline for organ and lesion segmentation tasks.",
        specs: { Parameters: "31M", "Input Size": "128×128×64" },
        dimensions: ["3D"],
        folderStructure: FOLDER_STRUCTURE.segmentation,
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
        document.getElementById('model-info-title').innerHTML = `<i class="fas fa-info-circle"></i> ${escapeHtml(getDisplayName(model.name))}`;
        document.getElementById('model-info-blurb').textContent = model.description || 'No description available.';
        const meta = document.getElementById('model-info-meta');
        meta.innerHTML = [
            model.parameters ? `<span class="model-meta-tag"><strong>Parameters:</strong> ${escapeHtml(model.parameters)}</span>` : '',
            model.input_size  ? `<span class="model-meta-tag"><strong>Input:</strong> ${escapeHtml(model.input_size)}</span>` : '',
            model.task        ? `<span class="model-meta-tag"><strong>Task:</strong> ${escapeHtml(model.task)}</span>` : '',
        ].join('');
        document.getElementById('model-papers-list').innerHTML = '<li>No papers listed.</li>';
        return;
    }

    placeholder.style.display = 'none';
    content.style.display = 'block';

    document.getElementById('model-info-title').innerHTML = `<i class="fas fa-info-circle"></i> ${escapeHtml(getDisplayName(model.name))}`;
    document.getElementById('model-info-blurb').textContent = info.blurb;

    const meta = document.getElementById('model-info-meta');
    meta.innerHTML = Object.entries(info.specs).map(([k, v]) =>
        `<span class="model-meta-tag"><strong>${k}:</strong> ${v}</span>`
    ).join('');

    const dimBadges = (info.dimensions || []).map(d =>
        `<span class="model-dim-badge model-dim-${d.toLowerCase()}">${d}</span>`
    ).join('');
    meta.innerHTML += `<div class="model-dim-row">${dimBadges}</div>`;

    // Folder structure
    const folderSection = document.getElementById('model-folder-structure');
    if (folderSection && info.folderStructure) {
        const fs = info.folderStructure;

        function renderTree(nodes, depth) {
            return nodes.map(n => {
                const hasChildren = n.children && n.children.length;
                const icon = hasChildren
                    ? '<i class="fas fa-folder"></i>'
                    : '<i class="fas fa-file"></i>';
                const note = n.note ? `<span class="folder-tree-note">${n.note}</span>` : '';
                const childHtml = hasChildren
                    ? `<div class="folder-tree-children">${renderTree(n.children, depth + 1)}</div>`
                    : '';
                return `<div class="folder-tree-file">
                    <span class="folder-tree-icon">${icon}</span>
                    <span class="folder-tree-name">${n.label}</span>${note}
                    ${childHtml}
                </div>`;
            }).join('');
        }

        folderSection.innerHTML = `
            <h4 class="model-papers-heading"><i class="fas fa-folder-open"></i> Data Layout</h4>
            <div class="folder-tree">
                <div class="folder-tree-root">
                    <i class="fas fa-folder-open"></i>
                    <span>${fs.label}</span>
                </div>
                <div class="folder-tree-children">${renderTree(fs.children, 1)}</div>
            </div>`;
        folderSection.style.display = 'block';
    } else if (folderSection) {
        folderSection.style.display = 'none';
    }

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
                <div class="job-id">Job: ${escapeHtml(job.job_id.substring(0, 8))}...</div>
                <div class="job-status ${job.status}">${escapeHtml(job.status.toUpperCase())}</div>
            </div>
            <div class="job-details">
                <div><strong>Model:</strong> ${escapeHtml(job.model_type)}</div>
                <div><strong>Created:</strong> ${escapeHtml(formatDate(job.created_at))}</div>
                ${job.completed_at ? `<div><strong>Completed:</strong> ${escapeHtml(formatDate(job.completed_at))}</div>` : ''}
            </div>
            <div class="job-actions">
                <button onclick="viewJobMetrics('${escapeJsString(job.job_id)}')" class="btn btn-secondary">
                    <i class="fas fa-chart-line"></i> View Metrics
                </button>
                ${job.status === 'running' ? 
                    `<button onclick="stopTrainingJob('${escapeJsString(job.job_id)}')" class="btn btn-secondary">
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
                    <title>Training Metrics - ${escapeHtml(jobId)}</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
                        .links { margin-top: 20px; }
                        .links a { display: inline-block; margin-right: 10px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
                    </style>
                </head>
                <body>
                    <h1>Training Metrics</h1>
                    <h2>Job ID: ${escapeHtml(jobId)}</h2>
                    <div class="metric"><strong>Epoch:</strong> ${escapeHtml(String(metrics.metrics.epoch))}</div>
                    <div class="metric"><strong>Training Loss:</strong> ${escapeHtml(String(metrics.metrics.train_loss))}</div>
                    <div class="metric"><strong>Validation Loss:</strong> ${escapeHtml(String(metrics.metrics.val_loss))}</div>
                    <div class="metric"><strong>Training Accuracy:</strong> ${escapeHtml(String((metrics.metrics.train_accuracy * 100).toFixed(2)))}%</div>
                    <div class="metric"><strong>Validation Accuracy:</strong> ${escapeHtml(String((metrics.metrics.val_accuracy * 100).toFixed(2)))}%</div>
                    <div class="metric"><strong>Learning Rate:</strong> ${escapeHtml(String(metrics.metrics.learning_rate))}</div>
                    
                    <div class="links">
                        <a href="${escapeHtml(safeUrl(metrics.metrics.tensorboard_url))}" target="_blank" rel="noopener noreferrer">Open TensorBoard</a>
                        <a href="${escapeHtml(safeUrl(metrics.metrics.mlflow_url))}" target="_blank" rel="noopener noreferrer">Open MLflow</a>
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
            <div class="model-name">${escapeHtml(model.model_name)}</div>
            <div class="model-description">Format: ${escapeHtml(model.model_format)} | Classes: ${model.num_classes}</div>
            <div class="model-specs">
                <span>Source: ${escapeHtml(model.s3_source)}</span>
                <span>Status: ${escapeHtml(model.status)}</span>
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
            <p>${escapeHtml(error.message)}</p>
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
                <strong>${escapeHtml(c.name)}</strong>
                <span>${escapeHtml(c.message)}</span>
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
        job_name: fd.get('sm_job_name') || null,
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
            statusBanner.innerHTML = `<i class="fas fa-check-circle"></i> Job submitted: <strong>${escapeHtml(response.job_name)}</strong> — Status: <strong>${escapeHtml(response.status || 'InProgress')}</strong>`;
        }
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Start SageMaker Training';
        loadSageMakerJobs();
    } catch (error) {
        if (statusBanner) {
            statusBanner.className = 'sm-status-banner sm-status-error';
            statusBanner.innerHTML = `<i class="fas fa-exclamation-circle"></i> Failed: ${escapeHtml(error.message)}`;
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

async function syncTrainingJobsFromAWS() {
    const btn = document.getElementById('btn-sync-training');
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Syncing...'; }
    try {
        const result = await apiCall('/api/training/sagemaker/sync', { method: 'POST' });
        const added = (result.added || []).length;
        const updated = (result.updated || []).length;
        alert(`Synced from AWS: ${added} added, ${updated} updated`);
        await loadSageMakerJobs();
    } catch (error) {
        alert('Sync failed: ' + (error.message || error));
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-sync"></i> Sync from AWS'; }
    }
}

function filterSageMakerJobs() {
    if (window._allSageMakerJobs) {
        displaySageMakerJobs(window._allSageMakerJobs);
    }
}

function switchJobsTab(tab) {
    window._jobsTab = tab;
    document.getElementById('tab-active').classList.toggle('active', tab === 'active');
    document.getElementById('tab-completed').classList.toggle('active', tab === 'completed');
    document.getElementById('tab-failed').classList.toggle('active', tab === 'failed');
    document.getElementById('tab-archived').classList.toggle('active', tab === 'archived');
    loadSageMakerJobs();
}

function displaySageMakerJobs(jobs) {
    window._allSageMakerJobs = jobs; // cache for search filtering
    const container = document.getElementById('sagemaker-jobs');
    if (!container) return;

    const tab = window._jobsTab || 'completed';
    const searchTerm = (document.getElementById('jobs-search')?.value || '').toLowerCase().trim();
    const ACTIVE_STATES = new Set(['InProgress', 'Stopping']);
    const COMPLETED_STATES = new Set(['Completed', 'Stopped']);

    const visible = jobs.filter(j => {
        if (tab === 'archived')   return j.archived;
        if (tab === 'failed')     return !j.archived && j.status === 'Failed';
        if (tab === 'completed')  return !j.archived && COMPLETED_STATES.has(j.status);
        return !j.archived && ACTIVE_STATES.has(j.status); // 'active'
    }).filter(j => {
        if (!searchTerm) return true;
        const name = (j.job_name || '').toLowerCase();
        const model = (j.config?.model_name || '').toLowerCase();
        const comment = (j.comment || '').toLowerCase();
        return name.includes(searchTerm) || model.includes(searchTerm) || comment.includes(searchTerm);
    });

    if (!visible.length) {
        const msgs = { active: 'No active jobs.', completed: 'No completed jobs.', failed: 'No failed jobs.', archived: 'No archived jobs.' };
        container.innerHTML = `<p style="color:#888; padding:0.5rem;">${msgs[tab] || 'No jobs.'}</p>`;
        return;
    }

    container.innerHTML = '';
    visible.forEach(job => {
        const statusClass = {
            'InProgress': 'status-running',
            'Completed':  'status-completed',
            'Failed':     'status-failed',
            'Stopping':   'status-stopping',
            'Stopped':    'status-stopped',
        }[job.status] || 'status-unknown';

        const terminal = ['Completed', 'Failed', 'Stopped'].includes(job.status);
        const r = job.result || {};

        // result.json highlight block
        const resultHtml = r.model_name ? `
            <div class="result-highlight">
                <div class="result-row"><span>Model</span><strong>${escapeHtml(r.model_name)}</strong></div>
                <div class="result-row"><span>Final epoch</span><strong>${r.final_epoch ?? '—'}</strong></div>
                ${r.best_val_dice != null
                    ? `<div class="result-row"><span>Best val dice</span><strong>${(r.best_val_dice * 100).toFixed(1)}%</strong></div>`
                    : `<div class="result-row"><span>Train acc</span><strong>${r.train_accuracy != null ? (r.train_accuracy * 100).toFixed(1) + '%' : '—'}</strong></div>
                       <div class="result-row"><span>Val acc</span><strong>${r.val_accuracy != null ? (r.val_accuracy * 100).toFixed(1) + '%' : '—'}</strong></div>
                       <div class="result-row"><span>Best val acc</span><strong>${r.best_val_accuracy != null ? (r.best_val_accuracy * 100).toFixed(1) + '%' : '—'}</strong></div>`
                }
                <div class="result-row"><span>Data</span><code style="font-size:0.75rem;">${escapeHtml(r.data_s3_uri || '—')}</code></div>
                ${r.fine_tuned_from ? `<div class="result-row" style="grid-column:1/-1;"><span>Fine-tuned from</span><code style="font-size:0.72rem;">${escapeHtml(r.fine_tuned_from)}</code></div>` : ''}
                ${r.reconstructed ? `<div class="result-row" style="grid-column:1/-1; color:#888; font-size:0.75rem; font-style:italic;"><span></span>Metrics unavailable — job predates result.json</div>` : ''}
            </div>` : '';

        // Lineage badge (fine-tuned from another job)
        const lineageHtml = job.fine_tuned_from
            ? `<div class="job-lineage"><i class="fas fa-code-branch"></i> Fine-tuned from: <code>${escapeHtml(job.fine_tuned_from)}</code></div>`
            : '';

        const commentHtml = job.comment
            ? `<div class="job-comment"><i class="fas fa-comment-alt"></i> ${escapeHtml(job.comment)}</div>` : '';

        const artifactHtml = job.model_artifacts
            ? (job.artifacts_deleted
                ? `<div style="color:#999; font-size:0.8rem;"><i class="fas fa-trash"></i> Artifacts deleted</div>`
                : `<div style="font-size:0.8rem;"><strong>Artifacts:</strong> <code style="font-size:0.75rem;">${escapeHtml(job.model_artifacts)}</code></div>`)
            : '';

        const el = document.createElement('div');
        el.className = `job-item${job.archived ? ' job-archived' : ''}`;
        el.id = `sm-job-${job.job_name}`;
        el.innerHTML = `
            <div class="job-header">
                <div class="job-id">${escapeHtml(job.job_name)}${job.archived ? ' <span style="color:#aaa;font-size:0.75rem;">[archived]</span>' : ''}</div>
                <div class="job-status ${statusClass}">${escapeHtml(job.status)}</div>
            </div>
            <div class="job-details">
                <div><strong>Model:</strong> ${escapeHtml(job.config?.model_name || 'N/A')} &nbsp;|&nbsp; <strong>Instance:</strong> ${escapeHtml(job.config?.instance_type || 'N/A')}</div>
                ${job.creation_time ? `<div><strong>Started:</strong> ${escapeHtml(new Date(job.creation_time).toLocaleString())}</div>` : ''}
                ${job.training_end_time ? `<div><strong>Ended:</strong> ${escapeHtml(new Date(job.training_end_time).toLocaleString())}</div>` : ''}
                ${job.billable_seconds ? `<div><strong>Training time:</strong> ${escapeHtml(formatDuration(job.billable_seconds))}</div>` : ''}
                ${job.estimated_cost_usd != null ? `<div><strong>Estimated cost:</strong> <span class="cost-badge">$${job.estimated_cost_usd.toFixed(4)} USD</span></div>` : ''}
                ${artifactHtml}
                ${resultHtml}
                ${lineageHtml}
                ${job.failure_reason ? `<div style="color:#c62828; font-size:0.82rem; margin-top:4px;"><strong>Failure:</strong> ${escapeHtml(job.failure_reason)}</div>` : ''}
            </div>
            <div id="sm-job-status-detail-${escapeHtml(job.job_name)}" class="job-status-detail" style="display:none;"></div>
            <div class="job-actions">
                ${!terminal ? `
                <button onclick="checkSageMakerStatus('${escapeJsString(job.job_name)}')" class="btn btn-secondary" id="sm-status-btn-${escapeHtml(job.job_name)}">
                    <i class="fas fa-sync"></i> Fetch Status
                </button>
                <button onclick="stopSageMakerJob('${escapeJsString(job.job_name)}')" class="btn btn-secondary">
                    <i class="fas fa-stop"></i> Stop
                </button>` : ''}
                ${job.mlflow_tracking_uri ? `
                <button type="button" class="btn btn-mlflow" onclick="openMlflowDashboard(this)" data-uri="${job.mlflow_tracking_uri.replace(/"/g, '&quot;')}">
                    <i class="fas fa-flask"></i> MLflow
                </button>` : ''}
                ${terminal ? `
                <button onclick="openCommentModal('${escapeJsString(job.job_name)}', \`${(job.comment||'').replace(/`/g,"'")}\`)" class="btn btn-secondary" title="Add comment">
                    <i class="fas fa-comment-alt"></i> Comment
                </button>
                <button onclick="archiveSageMakerJob('${escapeJsString(job.job_name)}')" class="btn btn-secondary" title="Archive" ${job.archived || tab !== 'completed' ? 'style="display:none"' : ''}>
                    <i class="fas fa-archive"></i> Archive
                </button>
                ${job.status === 'Completed' && !job.result && job.model_artifacts && !job.artifacts_deleted ? `
                <button onclick="loadJobResult('${escapeJsString(job.job_name)}')" class="btn btn-secondary" title="Load result.json from model artifacts">
                    <i class="fas fa-download"></i> Load Results
                </button>` : ''}
                ${job.status === 'Completed' && job.model_artifacts && !job.artifacts_deleted ? `
                <button onclick="openFineTuneModal('${escapeJsString(job.job_name)}')" class="btn btn-finetune" title="Fine-tune from this checkpoint">
                    <i class="fas fa-sliders-h"></i> Fine-tune
                </button>` : ''}
                ${job.model_artifacts && !job.artifacts_deleted ? `
                <button onclick="deleteJobArtifacts('${escapeJsString(job.job_name)}')" class="btn btn-secondary" title="Delete model.tar.gz from S3" style="color:#c62828;">
                    <i class="fas fa-trash"></i> Delete Artifacts
                </button>` : ''}
                <button onclick="removeSageMakerJob('${escapeJsString(job.job_name)}')" class="btn btn-secondary" style="color:#c62828;" title="Remove from list">
                    <i class="fas fa-times"></i> Remove
                </button>` : ''}
            </div>
            ${commentHtml}
        `;
        container.appendChild(el);
    });
}


async function loadJobResult(jobName) {
    try {
        const res = await apiCall(`/api/training/sagemaker/jobs/${jobName}/result`);
        loadSageMakerJobs();
    } catch (err) { alert(`Could not load result.json: ${err.message}`); }
}

async function removeSageMakerJob(jobName) {
    if (!confirm(`Remove "${jobName}" from the list? This does not affect AWS.`)) return;
    try {
        await apiCall(`/api/training/sagemaker/jobs/${jobName}`, { method: 'DELETE' });
        loadSageMakerJobs();
    } catch (err) { alert(`Error: ${err.message}`); }
}

async function archiveSageMakerJob(jobName) {
    try {
        await apiCall(`/api/training/sagemaker/jobs/${jobName}/archive`, { method: 'POST' });
        loadSageMakerJobs();
    } catch (err) { alert(`Error: ${err.message}`); }
}

async function deleteJobArtifacts(jobName) {
    if (!confirm(`Delete model.tar.gz from S3 for "${jobName}"? This cannot be undone.`)) return;
    try {
        const res = await apiCall(`/api/training/sagemaker/jobs/${jobName}/artifacts`, { method: 'DELETE' });
        // Show inline confirmation on the job card
        const jobEl = document.getElementById(`sm-job-${jobName}`);
        if (jobEl) {
            const banner = document.createElement('div');
            banner.className = 'sm-status-banner sm-status-success';
            banner.style.cssText = 'margin:0.5rem 0; font-size:0.85rem;';
            banner.innerHTML = `<i class="fas fa-check-circle"></i> Artifacts deleted: <code style="font-size:0.78rem;">${escapeHtml(res.deleted_uri)}</code>`;
            jobEl.querySelector('.job-details').appendChild(banner);
            setTimeout(() => loadSageMakerJobs(), 2000);
        } else {
            loadSageMakerJobs();
        }
    } catch (err) { alert(`Error: ${err.message}`); }
}

function openCommentModal(jobName, currentComment) {
    document.getElementById('comment-job-name').value  = jobName;
    document.getElementById('comment-text').value      = currentComment || '';
    document.getElementById('comment-modal').style.display = 'flex';
    document.getElementById('comment-text').focus();
}

function closeCommentModal() {
    document.getElementById('comment-modal').style.display = 'none';
}

async function saveComment() {
    const jobName = document.getElementById('comment-job-name').value;
    const comment = document.getElementById('comment-text').value.trim();
    try {
        await apiCall(`/api/training/sagemaker/jobs/${jobName}/comment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comment }),
        });
        closeCommentModal();
        loadSageMakerJobs();
    } catch (err) { alert(`Error: ${err.message}`); }
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
            let html = `<div class="status-detail-row"><span>Status:</span><strong>${escapeHtml(status.status)}</strong></div>`;
            if (status.creation_time) html += `<div class="status-detail-row"><span>Created:</span><strong>${escapeHtml(status.creation_time)}</strong></div>`;
            if (status.training_end_time) html += `<div class="status-detail-row"><span>Ended:</span><strong>${escapeHtml(status.training_end_time)}</strong></div>`;
            if (status.billable_seconds) html += `<div class="status-detail-row"><span>Billable seconds:</span><strong>${status.billable_seconds}s</strong></div>`;
            if (status.model_artifacts) html += `<div class="status-detail-row"><span>Artifacts:</span><strong>${escapeHtml(status.model_artifacts)}</strong></div>`;
            if (status.failure_reason) html += `<div class="status-detail-row status-error"><span>Failure:</span><strong>${escapeHtml(status.failure_reason)}</strong></div>`;
            detail.innerHTML = html;
            detail.style.display = 'block';
        }
    } catch (error) {
        if (detail) {
            detail.innerHTML = `<div class="status-detail-row status-error"><span>Error:</span><strong>${escapeHtml(error.message)}</strong></div>`;
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

// ── Utilities ─────────────────────────────────────────────────────────────────

async function openMlflowDashboard(btnOrUri) {
    const uri = (typeof btnOrUri === 'string') ? btnOrUri : btnOrUri.dataset.uri;
    if (!uri) return;

    // Plain HTTP URL — open directly
    if (uri.startsWith('http')) {
        window.open(uri, '_blank');
        return;
    }

    // Show loading state on the button
    const btn = (typeof btnOrUri === 'object') ? btnOrUri : null;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Opening…'; }

    try {
        const res = await apiCall(`/api/training/sagemaker/mlflow-url?arn=${encodeURIComponent(uri)}`);
        window.open(res.url, '_blank');
    } catch (e) {
        let msg;
        if (e.message.includes('expired') || e.message.includes('Expired') || e.message.includes('Invalid')) {
            msg = 'Your MLflow session has expired.\n\nClick "Open MLflow" again to get a fresh session (valid for 12 hours).';
        } else if (e.message.includes('does not exist') || e.message.includes('not found')) {
            msg = `MLflow tracking server not found.\n\nThe server may have been deleted or the ARN is outdated.\nARN: ${uri}\n\nOpen SageMaker Studio manually and navigate to the MLflow app.`;
        } else if (e.message.includes('Studio')) {
            msg = `${e.message}\n\nOpen SageMaker Studio manually and navigate to the MLflow app.`;
        } else {
            msg = `Could not open MLflow: ${e.message}`;
        }
        alert(msg);
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-flask"></i> MLflow'; }
    }
}

function generateJobName() {
    const model = document.getElementById('sm_model_name')?.value || 'model';
    const task  = document.querySelector('input[name="sm_task_type"]:checked')?.value || 'train';
    const date  = new Date();
    const ts    = `${date.getFullYear()}${String(date.getMonth()+1).padStart(2,'0')}${String(date.getDate()).padStart(2,'0')}-${String(date.getHours()).padStart(2,'0')}${String(date.getMinutes()).padStart(2,'0')}`;
    const name  = `${task}-${model}-${ts}`.toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 63);
    document.getElementById('sm_job_name').value = name;
}

function formatDuration(seconds) {
    if (!seconds) return 'N/A';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

// ── Container Image Management ────────────────────────────────────────────────
let DEFAULT_CONTAINERS = [
    { label: "classification-latest", uri: "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest" },
    { label: "segmentation-latest",   uri: "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest" },
];

/**
 * Called by index.html after /api/config loads. Replaces placeholder account IDs
 * in the default container list with real values from .env.local.
 */
function updateTrainingContainersFromConfig(cfg) {
    if (!cfg.training_image_classification && !cfg.training_image_segmentation) return;
    DEFAULT_CONTAINERS = [
        { label: "classification-latest", uri: cfg.training_image_classification || DEFAULT_CONTAINERS[0].uri },
        { label: "segmentation-latest",   uri: cfg.training_image_segmentation || DEFAULT_CONTAINERS[1].uri },
    ];
    // Clear localStorage to pick up new defaults if account changed
    localStorage.removeItem('sm_containers');
    populateContainerDropdown();
}

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
        `<option value="${escapeHtml(c.uri)}">${escapeHtml(c.label)} — ${escapeHtml(c.uri)}</option>`
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

// ── Custom Model Management ───────────────────────────────────────────────────

const CUSTOM_MODELS_KEY = 'medml_custom_models';

function loadCustomModels() {
    try { return JSON.parse(localStorage.getItem(CUSTOM_MODELS_KEY) || '[]'); }
    catch { return []; }
}

function saveCustomModels(models) {
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(models));
}

function openAddCustomModelModal() {
    document.getElementById('cm-name').value        = '';
    document.getElementById('cm-description').value = '';
    document.getElementById('cm-parameters').value  = '';
    document.getElementById('cm-input-size').value  = '';
    document.getElementById('cm-task').value        = 'classification';
    document.getElementById('cm-dimension').value   = '2D';
    document.getElementById('add-custom-model-modal').style.display = 'flex';
    document.getElementById('cm-name').focus();
}

function closeAddCustomModelModal() {
    document.getElementById('add-custom-model-modal').style.display = 'none';
}

function saveCustomModel() {
    const name = document.getElementById('cm-name').value.trim();
    if (!name) { alert('Model name is required.'); return; }

    const model = {
        name:        name.toLowerCase().replace(/\s+/g, '-'),
        description: document.getElementById('cm-description').value.trim() || 'Custom model',
        parameters:  document.getElementById('cm-parameters').value.trim() || 'N/A',
        input_size:  document.getElementById('cm-input-size').value.trim() || 'N/A',
        task:        document.getElementById('cm-task').value,
        dimension:   document.getElementById('cm-dimension').value,
        pretrained:  false,
        custom:      true,
    };

    const existing = loadCustomModels();
    // Replace if same name exists
    const idx = existing.findIndex(m => m.name === model.name);
    if (idx >= 0) existing[idx] = model;
    else existing.push(model);
    saveCustomModels(existing);

    closeAddCustomModelModal();
    loadAvailableModels();  // refresh the model list
}

function getCustomModels() {
    return loadCustomModels();
}

// ── Task type auto-wiring ─────────────────────────────────────────────────────

function onTaskTypeChange() {
    const task = document.querySelector('input[name="sm_task_type"]:checked')?.value || 'classification';

    // Auto-set dimension
    const dim = task === 'segmentation' ? '3D' : '2D';
    const dimRadio = document.querySelector(`input[name="sm_dimension"][value="${dim}"]`);
    if (dimRadio) dimRadio.checked = true;

    // Filter model dropdown to matching task
    const select = document.getElementById('sm_model_name');
    if (select) {
        Array.from(select.options).forEach(opt => {
            opt.hidden = opt.dataset.task && opt.dataset.task !== task;
        });
        // Select first visible option
        const first = Array.from(select.options).find(o => !o.hidden);
        if (first) select.value = first.value;
    }

    // Update default batch size for segmentation (smaller due to 3D volumes)
    const batchInput = document.getElementById('sm_batch_size');
    if (batchInput) batchInput.value = task === 'segmentation' ? '2' : '16';

    // Update default epochs
    const epochInput = document.getElementById('sm_epochs');
    if (epochInput) epochInput.value = task === 'segmentation' ? '10' : '1';
}

// ── Fine-tune Modal ───────────────────────────────────────────────────────────

function openFineTuneModal(jobName) {
    // Fetch fresh job list to get full config
    apiCall('/api/training/sagemaker/jobs?local_only=true').then(resp => {
        const job = (resp.jobs || []).find(j => j.job_name === jobName);
        if (!job) { alert('Job not found'); return; }
        const cfg     = job.config || {};
        const metrics = job.result || {};

        // Base checkpoint header
        document.getElementById('ft-base-job-name').textContent  = jobName;
        document.getElementById('ft-base-artifacts').textContent = job.model_artifacts || '';
        document.getElementById('ft-checkpoint-uri').value       = job.model_artifacts || '';
        document.getElementById('ft-source-job-name').value      = jobName;

        // Previous dataset + metrics panel
        const dataUri = cfg.data_s3_uri || '—';
        const metricsHtml = metrics.model_name ? `
            <div class="ft-metrics-grid">
                <div class="ft-metric"><span>Train acc</span><strong>${metrics.train_accuracy != null ? (metrics.train_accuracy*100).toFixed(1)+'%' : '—'}</strong></div>
                <div class="ft-metric"><span>Val acc</span><strong>${metrics.val_accuracy != null ? (metrics.val_accuracy*100).toFixed(1)+'%' : '—'}</strong></div>
                <div class="ft-metric"><span>Best val acc</span><strong>${metrics.best_val_accuracy != null ? (metrics.best_val_accuracy*100).toFixed(1)+'%' : '—'}</strong></div>
                <div class="ft-metric"><span>Epochs</span><strong>${metrics.final_epoch ?? cfg.epochs ?? '—'}</strong></div>
                <div class="ft-metric"><span>LR</span><strong>${metrics.learning_rate ?? cfg.learning_rate ?? '—'}</strong></div>
                <div class="ft-metric"><span>Batch</span><strong>${metrics.batch_size ?? cfg.batch_size ?? '—'}</strong></div>
            </div>` : `<p style="color:#888; font-size:0.8rem; margin:0.5rem 0 0;">No metrics loaded — use "Load Results" on the job card first.</p>`;

        document.getElementById('ft-base-info').innerHTML = `
            <div class="ft-info-row"><i class="fas fa-database" style="color:#16a085;"></i><span>Training data</span><code>${escapeHtml(dataUri)}</code></div>
            <div class="ft-info-row"><i class="fas fa-microchip" style="color:#8e44ad;"></i><span>Model</span><strong>${escapeHtml(cfg.model_name || '—')}</strong> &nbsp;·&nbsp; ${escapeHtml(cfg.task_type || '')} ${escapeHtml(cfg.dimension || '')}</div>
            <div class="ft-info-row"><i class="fas fa-server" style="color:#1a6fa0;"></i><span>Instance</span><strong>${escapeHtml(cfg.instance_type || '—')}</strong></div>
            ${metricsHtml}`;

        // Pre-fill form from base job config
        document.getElementById('ft-job-name').value         = '';
        document.getElementById('ft-data-s3-uri').value      = cfg.data_s3_uri || '';
        document.getElementById('ft-output-s3-uri').value    = cfg.output_s3_uri || '';
        document.getElementById('ft-epochs').value           = cfg.epochs || 5;
        document.getElementById('ft-learning-rate').value    = (cfg.learning_rate || 0.001) * 0.1;
        document.getElementById('ft-batch-size').value       = cfg.batch_size || 16;
        document.getElementById('ft-val-interval').value     = cfg.val_interval || 1;
        document.getElementById('ft-instance-type').value    = cfg.instance_type || 'ml.g5.xlarge';
        document.getElementById('ft-image-uri').value        = cfg.image_uri || '';
        document.getElementById('ft-role-arn').value         = cfg.role_arn || '';
        document.getElementById('ft-model-name').value       = cfg.model_name || '';
        document.getElementById('ft-task-type').value        = cfg.task_type || 'classification';
        document.getElementById('ft-dimension').value        = cfg.dimension || '2D';
        document.getElementById('ft-mlflow-uri').value       = cfg.mlflow_tracking_uri || '';

        document.getElementById('finetune-modal').style.display = 'flex';
    }).catch(err => alert(`Could not load job: ${err.message}`));
}

function closeFineTuneModal() {
    document.getElementById('finetune-modal').style.display = 'none';
    document.getElementById('ft-status').style.display = 'none';
}

async function submitFineTuneJob() {
    const btn = document.getElementById('ft-submit-btn');
    const statusEl = document.getElementById('ft-status');

    const jobNameRaw = document.getElementById('ft-job-name').value.trim();
    const date  = new Date();
    const ts    = `${date.getFullYear()}${String(date.getMonth()+1).padStart(2,'0')}${String(date.getDate()).padStart(2,'0')}-${String(date.getHours()).padStart(2,'0')}${String(date.getMinutes()).padStart(2,'0')}`;
    const autoName = `finetune-${document.getElementById('ft-model-name').value || 'model'}-${ts}`.toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 63);

    const payload = {
        job_name:            jobNameRaw || autoName,
        model_name:          document.getElementById('ft-model-name').value,
        task_type:           document.getElementById('ft-task-type').value,
        dimension:           document.getElementById('ft-dimension').value,
        epochs:              parseInt(document.getElementById('ft-epochs').value),
        batch_size:          parseInt(document.getElementById('ft-batch-size').value),
        learning_rate:       parseFloat(document.getElementById('ft-learning-rate').value),
        val_interval:        parseInt(document.getElementById('ft-val-interval').value),
        instance_type:       document.getElementById('ft-instance-type').value,
        instance_count:      1,
        image_uri:           document.getElementById('ft-image-uri').value,
        data_s3_uri:         document.getElementById('ft-data-s3-uri').value,
        role_arn:            document.getElementById('ft-role-arn').value,
        output_s3_uri:       document.getElementById('ft-output-s3-uri').value || null,
        mlflow_tracking_uri: document.getElementById('ft-mlflow-uri').value || null,
        checkpoint_s3_uri:   document.getElementById('ft-checkpoint-uri').value,
        fine_tuned_from:     document.getElementById('ft-source-job-name').value,
        source_dir:          'src/' + document.getElementById('ft-task-type').value,
        entry_point:         'train.py',
    };

    if (!payload.data_s3_uri)  { alert('Data S3 URI is required.'); return; }
    if (!payload.image_uri)    { alert('Container image URI is required.'); return; }
    if (!payload.role_arn)     { alert('IAM Role ARN is required.'); return; }

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting…';
    statusEl.style.display = 'none';

    try {
        const res = await apiCall('/api/training/sagemaker/finetune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        statusEl.className = 'sm-status-banner sm-status-success';
        statusEl.innerHTML = `<i class="fas fa-check-circle"></i> Fine-tune job submitted: <strong>${escapeHtml(res.job_name)}</strong>`;
        statusEl.style.display = 'flex';
        setTimeout(() => {
            closeFineTuneModal();
            switchJobsTab('active');
            loadSageMakerJobs();
        }, 1800);
    } catch (err) {
        statusEl.className = 'sm-status-banner sm-status-error';
        statusEl.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}`;
        statusEl.style.display = 'flex';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sliders-h"></i> Start Fine-tuning';
    }
}
