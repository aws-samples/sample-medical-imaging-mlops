// Preprocessing JavaScript

document.addEventListener('DOMContentLoaded', function() {
    loadPreprocessingSteps();
    
    const configureBtn = document.getElementById('configure-preprocessing');
    if (configureBtn) {
        configureBtn.addEventListener('click', configurePreprocessingPipeline);
    }
});

async function loadPreprocessingSteps() {
    try {
        const response = await apiCall('/api/preprocessing/steps');
        displayPreprocessingSteps(response.steps);
    } catch (error) {
        console.error('Error loading preprocessing steps:', error);
    }
}

function displayPreprocessingSteps(steps) {
    const container = document.getElementById('preprocessing-steps');
    if (!container) return;
    
    container.innerHTML = '';
    
    steps.forEach(step => {
        const stepElement = document.createElement('div');
        stepElement.className = 'preprocessing-step';
        stepElement.dataset.stepName = step.name;
        
        stepElement.innerHTML = `
            <div class="step-name">${step.name.replace('_', ' ').toUpperCase()}</div>
            <div class="step-description">${step.description}</div>
            <div class="step-parameters">
                ${Object.keys(step.parameters).map(param => 
                    `<small>${param}: ${JSON.stringify(step.parameters[param].default)}</small>`
                ).join('<br>')}
            </div>
        `;
        
        stepElement.addEventListener('click', () => togglePreprocessingStep(step, stepElement));
        container.appendChild(stepElement);
    });
}

function togglePreprocessingStep(step, element) {
    const isSelected = element.classList.contains('selected');
    
    if (isSelected) {
        // Deselect
        element.classList.remove('selected');
        window.MedMLApp.selectedSteps = window.MedMLApp.selectedSteps.filter(s => s.name !== step.name);
    } else {
        // Select
        element.classList.add('selected');
        window.MedMLApp.selectedSteps.push(step);
    }
    
    updateSelectedStepsDisplay();
}

function updateSelectedStepsDisplay() {
    const container = document.getElementById('selected-steps');
    if (!container) return;
    
    if (window.MedMLApp.selectedSteps.length === 0) {
        container.innerHTML = '<p>No preprocessing steps selected</p>';
        return;
    }
    
    container.innerHTML = window.MedMLApp.selectedSteps.map(step => 
        `<span class="selected-step-tag">${step.name.replace('_', ' ').toUpperCase()}</span>`
    ).join('');
}

async function configurePreprocessingPipeline() {
    if (window.MedMLApp.selectedSteps.length === 0) {
        alert('Please select at least one preprocessing step');
        return;
    }
    
    try {
        const config = {
            steps: window.MedMLApp.selectedSteps.map(step => step.name),
            parameters: {}
        };
        
        // Collect parameters from selected steps
        window.MedMLApp.selectedSteps.forEach(step => {
            Object.keys(step.parameters).forEach(param => {
                config.parameters[`${step.name}_${param}`] = step.parameters[param].default;
            });
        });
        
        const response = await apiCall('/api/preprocessing/configure', {
            method: 'POST',
            body: JSON.stringify(config)
        });
        
        alert(`Preprocessing pipeline configured successfully! Pipeline ID: ${response.pipeline_id}`);
        
        // Enable training section
        enableSection('training');
        
        // Execute preprocessing
        executePreprocessing(response.pipeline_id);
        
    } catch (error) {
        alert(`Error configuring preprocessing: ${error.message}`);
    }
}

async function executePreprocessing(pipelineId) {
    try {
        const response = await apiCall(`/api/preprocessing/execute/${pipelineId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `dataset_path=/path/to/imported/data`
        });
        
        console.log(`Preprocessing job started: ${response.job_id}`);
        
        // Monitor preprocessing progress
        monitorPreprocessingProgress(response.job_id);
        
    } catch (error) {
        console.error('Error executing preprocessing:', error);
    }
}

async function monitorPreprocessingProgress(jobId) {
    // This would monitor the actual preprocessing job
    // For now, simulate completion after 10 seconds
    setTimeout(() => {
        console.log(`Preprocessing job ${jobId} completed`);
    }, 10000);
}