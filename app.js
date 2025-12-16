/**
 * WaterSlayer - Frontend Application
 * 
 * Watermark removal using YOLO detection + ProPainter inpainting
 */

// Helper
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================
// API Client
// ============================================

const api = {
    baseUrl: '',

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: { 'Content-Type': 'application/json' }
        };

        const response = await fetch(url, { ...defaultOptions, ...options });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Request failed');
        }

        return response.json();
    },

    // System
    async getSystemInfo() {
        return this.request('/api/system/info');
    },

    // Models
    async getModels() {
        return this.request('/api/models');
    },

    async importModel(sourcePath) {
        return this.request(`/api/models/import?source_path=${encodeURIComponent(sourcePath)}`, {
            method: 'POST'
        });
    },

    async loadModel(modelPath) {
        return this.request(`/api/models/load?model_path=${encodeURIComponent(modelPath)}`, {
            method: 'POST'
        });
    },

    async getLoadedModel() {
        return this.request('/api/models/loaded');
    },

    // Videos
    async getVideos() {
        return this.request('/api/videos');
    },

    // Processing
    async startProcessing(videoPath, modelPath, confThreshold, method) {
        return this.request('/api/process', {
            method: 'POST',
            body: JSON.stringify({
                video_path: videoPath,
                model_path: modelPath,
                conf_threshold: confThreshold,
                inpaint_method: method
            })
        });
    },

    async getJobStatus(jobId) {
        return this.request(`/api/jobs/${jobId}`);
    },

    async cancelJob(jobId) {
        return this.request(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    },

    // Outputs
    async getOutputs() {
        return this.request('/api/outputs');
    },

    downloadOutput(filename) {
        window.location.href = `/api/outputs/${encodeURIComponent(filename)}/download`;
    },

    async deleteOutput(filename) {
        return this.request(`/api/outputs/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
    }
};

// ============================================
// State
// ============================================

const state = {
    connected: false,
    models: [],
    videos: [],
    outputs: [],

    currentJobId: null,
    isProcessing: false,

    selectedModel: null,
    selectedVideo: null,
    selectedVideoFile: null,  // File object from browse
    isUploading: false,

    loadedModel: null,  // Currently loaded model in GPU memory
    isLoadingModel: false
};

// ============================================
// UI Functions
// ============================================

const ui = {
    showToast(message, type = 'success') {
        const toast = $('#toast');
        const msgEl = $('#toast-message');

        msgEl.textContent = message;
        toast.className = `toast show ${type}`;

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    },

    updateConnectionStatus(connected) {
        state.connected = connected;
        const dot = $('#status-dot');
        const text = $('#status-text');

        if (connected) {
            dot.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            dot.classList.remove('connected');
            text.textContent = 'Disconnected';
        }
    },

    populateModels(models) {
        state.models = models;
        const select = $('#model-select');

        select.innerHTML = '<option value="">Select a model...</option>';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = `${model.name} (${(model.size / 1024 / 1024).toFixed(1)} MB)`;
            select.appendChild(option);
        });
    },

    populateVideos(videos) {
        state.videos = videos;
        const select = $('#video-select');

        select.innerHTML = '<option value="">Select a video...</option>';

        videos.forEach(video => {
            const option = document.createElement('option');
            option.value = video.path;
            option.textContent = `${video.name} (${(video.size / 1024 / 1024).toFixed(1)} MB)`;
            select.appendChild(option);
        });
    },

    populateOutputs(outputs) {
        state.outputs = outputs;
        const container = $('#outputs-list');

        if (!outputs || outputs.length === 0) {
            container.innerHTML = '<p class="empty-state">No processed videos yet</p>';
            return;
        }

        container.innerHTML = outputs.map(output => {
            const size = (output.size / 1024 / 1024).toFixed(1);
            const date = new Date(output.modified * 1000).toLocaleDateString();

            return `
                <div class="output-item">
                    <div class="output-info">
                        <div class="output-name">${output.name}</div>
                        <div class="output-meta">${size} MB • ${date}</div>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-primary btn-sm" onclick="api.downloadOutput('${output.filename}')">
                            Download
                        </button>
                        <button class="btn btn-ghost btn-sm" onclick="deleteOutput('${output.filename}')">
                            Delete
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    },

    updateProcessButton() {
        const btn = $('#process-btn');
        const model = $('#model-select').value;
        const video = $('#video-select').value;
        const hasFile = state.selectedVideoFile !== null;

        btn.disabled = !model || (!video && !hasFile) || state.isProcessing || state.isUploading;
    },

    showSelectedVideo(file) {
        const infoEl = $('#selected-video-info');
        const nameEl = $('#selected-video-name');
        const metaEl = $('#selected-video-meta');

        if (!file) {
            infoEl.classList.add('hidden');
            return;
        }

        const sizeMB = (file.size / 1024 / 1024).toFixed(1);
        nameEl.textContent = file.name;
        metaEl.textContent = `${sizeMB} MB`;
        infoEl.classList.remove('hidden');

        // Clear the dropdown selection since we're using a file
        $('#video-select').value = '';
    },

    clearSelectedVideo() {
        state.selectedVideoFile = null;
        state.selectedVideo = null;
        $('#selected-video-info').classList.add('hidden');
        $('#video-file-input').value = '';
        this.updateProcessButton();
    },

    showProgress(show) {
        const statusContent = $('#status-content');
        const progressSection = $('#progress-section');
        const cancelBtn = $('#cancel-btn');

        if (show) {
            statusContent.classList.add('hidden');
            progressSection.classList.remove('hidden');
            cancelBtn.classList.remove('hidden');
        } else {
            statusContent.classList.remove('hidden');
            progressSection.classList.add('hidden');
            cancelBtn.classList.add('hidden');
        }
    },

    updateProgress(job) {
        $('#stage-text').textContent = job.current_stage || 'Processing...';
        $('#progress-percent').textContent = `${Math.round(job.progress)}%`;
        $('#progress-fill').style.width = `${job.progress}%`;
        $('#stat-frames').textContent = job.total_frames || 0;
        $('#stat-watermarks').textContent = job.watermarks_found || 0;
    },

    updateSystemInfo(info) {
        $('#gpu-name').textContent = info.gpu_name || 'CPU';
        $('#cuda-status').textContent = info.cuda_available ? `✓ ${info.cuda_version}` : '✗ Not available';
        $('#pytorch-version').textContent = info.pytorch_version || 'Unknown';
    },

    updateModelStatus(modelName, isLoading = false) {
        const statusEl = $('#loaded-model-status');
        const nameEl = $('#loaded-model-name');

        if (isLoading) {
            statusEl.className = 'model-status loading';
            nameEl.textContent = 'Loading model...';
        } else if (modelName) {
            statusEl.className = 'model-status loaded';
            nameEl.textContent = `✓ ${modelName} loaded`;
        } else {
            statusEl.className = 'model-status';
            nameEl.textContent = 'No model loaded';
        }
    },

    updateLoadButton() {
        const btn = $('#load-model-btn');
        const modelPath = $('#model-select').value;

        btn.disabled = !modelPath || state.isLoadingModel;
    }
};

// ============================================
// Actions
// ============================================

async function loadModels() {
    try {
        const data = await api.getModels();
        ui.populateModels(data.models || []);
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

async function loadVideos() {
    try {
        const data = await api.getVideos();
        ui.populateVideos(data.videos || []);
    } catch (error) {
        console.error('Failed to load videos:', error);
    }
}

async function handleVideoFile(file) {
    if (!file) return;

    state.selectedVideoFile = file;
    ui.showSelectedVideo(file);

    // Upload the file to the server
    try {
        state.isUploading = true;
        ui.updateProcessButton();
        ui.showToast('Uploading video...', 'info');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/videos/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const result = await response.json();

        // Store the server path for processing
        state.selectedVideo = result.path;
        state.isUploading = false;

        ui.updateProcessButton();
        ui.showToast('Video ready!', 'success');

        // Refresh the video list to show the new file
        await loadVideos();

    } catch (error) {
        state.isUploading = false;
        ui.updateProcessButton();
        ui.showToast(`Upload failed: ${error.message}`, 'error');
    }
}

async function loadOutputs() {
    try {
        const data = await api.getOutputs();
        ui.populateOutputs(data.outputs || []);
    } catch (error) {
        console.error('Failed to load outputs:', error);
    }
}

async function loadSystemInfo() {
    try {
        const info = await api.getSystemInfo();
        ui.updateSystemInfo(info);
    } catch (error) {
        console.error('Failed to load system info:', error);
    }
}

async function importModel() {
    const path = prompt(
        'Enter the path to the model file (.pt):\n\n' +
        'Example: F:\\SoraWMR\\AnnoStudio\\data\\weights\\watermark_detector.pt'
    );

    if (!path) return;

    try {
        await api.importModel(path);
        ui.showToast('Model imported successfully!', 'success');
        await loadModels();
    } catch (error) {
        ui.showToast(`Failed to import: ${error.message}`, 'error');
    }
}

async function loadModelToGPU() {
    const modelPath = $('#model-select').value;

    if (!modelPath) {
        ui.showToast('Please select a model first', 'error');
        return;
    }

    try {
        state.isLoadingModel = true;
        ui.updateLoadButton();
        ui.updateModelStatus(null, true);  // Show loading state

        const result = await api.loadModel(modelPath);

        state.loadedModel = result.model_name;
        state.isLoadingModel = false;

        ui.updateModelStatus(result.model_name);
        ui.updateLoadButton();
        ui.showToast(`Model '${result.model_name}' loaded to GPU!`, 'success');

    } catch (error) {
        state.isLoadingModel = false;
        ui.updateModelStatus(null);
        ui.updateLoadButton();
        ui.showToast(`Failed to load model: ${error.message}`, 'error');
    }
}

async function checkLoadedModel() {
    try {
        const result = await api.getLoadedModel();
        if (result.loaded) {
            state.loadedModel = result.model_name;
            ui.updateModelStatus(result.model_name);
        }
    } catch (error) {
        console.error('Failed to check loaded model:', error);
    }
}

async function startProcessing() {
    const modelPath = $('#model-select').value;
    const videoPath = state.selectedVideo || $('#video-select').value;
    const confThreshold = parseInt($('#conf-slider').value) / 100;
    const method = $('#method-select').value;

    if (!modelPath) {
        ui.showToast('Please select a model', 'error');
        return;
    }

    if (!videoPath) {
        ui.showToast('Please select or browse a video', 'error');
        return;
    }

    try {
        state.isProcessing = true;
        ui.updateProcessButton();
        ui.showProgress(true);

        const result = await api.startProcessing(videoPath, modelPath, confThreshold, method);
        state.currentJobId = result.job_id;

        // Start polling for status
        pollJobStatus(result.job_id);

    } catch (error) {
        ui.showToast(`Failed to start: ${error.message}`, 'error');
        state.isProcessing = false;
        ui.updateProcessButton();
        ui.showProgress(false);
    }
}

async function pollJobStatus(jobId) {
    try {
        const job = await api.getJobStatus(jobId);

        ui.updateProgress(job);

        if (job.status === 'completed') {
            ui.showToast('Processing complete!', 'success');
            state.isProcessing = false;
            state.currentJobId = null;
            ui.updateProcessButton();
            ui.showProgress(false);
            loadOutputs();
            return;
        }

        if (job.status === 'failed') {
            ui.showToast(`Processing failed: ${job.error}`, 'error');
            state.isProcessing = false;
            state.currentJobId = null;
            ui.updateProcessButton();
            ui.showProgress(false);
            return;
        }

        if (job.status === 'cancelled') {
            ui.showToast('Processing cancelled', 'info');
            state.isProcessing = false;
            state.currentJobId = null;
            ui.updateProcessButton();
            ui.showProgress(false);
            return;
        }

        // Continue polling
        setTimeout(() => pollJobStatus(jobId), 500);

    } catch (error) {
        console.error('Failed to get job status:', error);
        setTimeout(() => pollJobStatus(jobId), 1000);
    }
}

async function cancelProcessing() {
    if (!state.currentJobId) return;

    try {
        await api.cancelJob(state.currentJobId);
        ui.showToast('Cancelling...', 'info');
    } catch (error) {
        ui.showToast('Failed to cancel', 'error');
    }
}

async function deleteOutput(filename) {
    if (!confirm(`Delete ${filename}?`)) return;

    try {
        await api.deleteOutput(filename);
        ui.showToast('Deleted', 'success');
        loadOutputs();
    } catch (error) {
        ui.showToast(`Failed to delete: ${error.message}`, 'error');
    }
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Model selection
    $('#model-select').addEventListener('change', (e) => {
        state.selectedModel = e.target.value;
        ui.updateProcessButton();
        ui.updateLoadButton();
    });

    // Video selection from dropdown
    $('#video-select').addEventListener('change', (e) => {
        state.selectedVideo = e.target.value;
        // Clear any browsed file when selecting from dropdown
        if (e.target.value) {
            state.selectedVideoFile = null;
            $('#selected-video-info').classList.add('hidden');
            $('#video-file-input').value = '';
        }
        ui.updateProcessButton();
    });

    // Browse video button
    $('#browse-video-btn').addEventListener('click', () => {
        $('#video-file-input').click();
    });

    // Video file selected
    $('#video-file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleVideoFile(file);
        }
    });

    // Clear selected video
    $('#clear-video-btn').addEventListener('click', () => {
        ui.clearSelectedVideo();
    });

    // Confidence slider
    $('#conf-slider').addEventListener('input', (e) => {
        $('#conf-display').textContent = `${e.target.value}%`;
    });

    // Refresh buttons
    $('#refresh-models-btn').addEventListener('click', loadModels);
    $('#refresh-videos-btn').addEventListener('click', loadVideos);
    $('#refresh-outputs-btn').addEventListener('click', loadOutputs);

    // Load model button
    $('#load-model-btn').addEventListener('click', loadModelToGPU);

    // Import model
    $('#import-model-btn').addEventListener('click', importModel);

    // Process button
    $('#process-btn').addEventListener('click', startProcessing);

    // Cancel button
    $('#cancel-btn').addEventListener('click', cancelProcessing);
}

// ============================================
// Initialization
// ============================================

async function init() {
    console.log('🗡️ WaterSlayer initializing...');

    setupEventListeners();

    // Check connection
    try {
        await loadSystemInfo();
        ui.updateConnectionStatus(true);

        // Load initial data
        await Promise.all([
            loadModels(),
            loadVideos(),
            loadOutputs(),
            checkLoadedModel()
        ]);

        console.log('✅ WaterSlayer ready!');

    } catch (error) {
        console.error('Failed to initialize:', error);
        ui.updateConnectionStatus(false);
        ui.showToast('Failed to connect to backend', 'error');
    }
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
