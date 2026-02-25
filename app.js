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
    async startProcessing(videoPath, modelPath, highThreshold, lowThreshold, detectionMode, method, qualityMode, detailRestoreMode) {
        return this.request('/api/process', {
            method: 'POST',
            body: JSON.stringify({
                video_path: videoPath,
                model_path: modelPath,
                conf_threshold: highThreshold,
                high_threshold: highThreshold,
                low_threshold: lowThreshold,
                detection_mode: detectionMode,
                inpaint_method: method,
                quality_mode: qualityMode,
                detail_restore_mode: detailRestoreMode
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
    },

    async getVideoFrame(videoPath, frame = 0) {
        const url = `/api/videos/frame?path=${encodeURIComponent(videoPath)}&frame=${frame}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch frame');
        const blob = await response.blob();
        const w = parseInt(response.headers.get('X-Video-Width') || '0');
        const h = parseInt(response.headers.get('X-Video-Height') || '0');
        const total = parseInt(response.headers.get('X-Total-Frames') || '0');
        return { blob, videoWidth: w, videoHeight: h, totalFrames: total };
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
    isLoadingModel: false,

    // Manual box mode
    detectionMode: 'yolo',  // 'yolo' or 'manual'
    manualBox: null,        // {x, y, width, height} in video pixel coords
    videoWidth: 0,
    videoHeight: 0,
    isDrawing: false,
    drawStart: null,
    frameImage: null        // Image element for canvas
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
        const hasVideo = video || hasFile;

        if (state.detectionMode === 'manual') {
            btn.disabled = !hasVideo || !state.manualBox || state.isProcessing || state.isUploading;
        } else {
            btn.disabled = !model || !hasVideo || state.isProcessing || state.isUploading;
        }
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

        // Reset manual box when video changes
        state.manualBox = null;
        $('#box-coords').classList.add('hidden');

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
    const highThreshold = parseInt($('#conf-slider').value) / 100;
    const lowThreshold = parseInt($('#low-conf-slider').value) / 100;
    const detectionMode = $('#detection-mode-select').value;
    const method = $('#method-select').value;
    const qualityMode = $('#quality-select').value;
    const detailRestoreMode = $('#detail-restore-select').value;

    if (state.detectionMode === 'manual') {
        if (!videoPath) {
            ui.showToast('Please select or browse a video', 'error');
            return;
        }
        if (!state.manualBox) {
            ui.showToast('Please draw a box on the frame', 'error');
            return;
        }
    } else {
        if (!modelPath) {
            ui.showToast('Please select a model', 'error');
            return;
        }
        if (!videoPath) {
            ui.showToast('Please select or browse a video', 'error');
            return;
        }
    }

    try {
        state.isProcessing = true;
        ui.updateProcessButton();
        ui.showProgress(true);

        const body = {
            video_path: videoPath,
            model_path: modelPath || '',
            conf_threshold: highThreshold,
            high_threshold: highThreshold,
            low_threshold: lowThreshold,
            detection_mode: detectionMode,
            inpaint_method: method,
            quality_mode: qualityMode,
            detail_restore_mode: detailRestoreMode
        };

        if (state.detectionMode === 'manual' && state.manualBox) {
            body.manual_box = state.manualBox;
        }

        const result = await api.request('/api/process', {
            method: 'POST',
            body: JSON.stringify(body)
        });
        state.currentJobId = result.job_id;

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
        state.manualBox = null;
        $('#box-coords').classList.add('hidden');
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
    $('#low-conf-slider').addEventListener('input', (e) => {
        $('#low-conf-display').textContent = `${e.target.value}%`;
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
// Manual Box Drawing (Modal)
// ============================================

// Temporary box while modal is open (committed on Confirm)
let _modalBox = null;
let _totalFrames = 0;
let _currentFrame = 0;

async function loadFramePreview(frameNum = 0) {
    const videoPath = state.selectedVideo || $('#video-select').value;
    if (!videoPath) return;

    const canvas = $('#box-canvas');
    const ctx = canvas.getContext('2d');

    try {
        const { blob, videoWidth, videoHeight } = await api.getVideoFrame(videoPath, frameNum);
        state.videoWidth = videoWidth;
        state.videoHeight = videoHeight;

        const img = new Image();
        img.onload = () => {
            state.frameImage = img;
            const wrap = $('#modal-canvas-wrap');
            const maxW = wrap.clientWidth;
            const maxH = window.innerHeight - 200; // leave room for header/footer
            let displayW = maxW;
            let displayH = Math.round(videoHeight * (maxW / videoWidth));

            // If too tall, scale down to fit
            if (displayH > maxH) {
                displayH = maxH;
                displayW = Math.round(videoWidth * (maxH / videoHeight));
            }

            canvas.width = displayW;
            canvas.height = displayH;

            ctx.drawImage(img, 0, 0, displayW, displayH);

            // Redraw existing box if any
            if (_modalBox) {
                drawBoxOverlay(_modalBox);
            }
        };
        img.src = URL.createObjectURL(blob);
    } catch (e) {
        console.error('Failed to load frame preview:', e);
    }
}

function drawBoxOverlay(box) {
    const canvas = $('#box-canvas');
    const ctx = canvas.getContext('2d');
    if (!state.frameImage) return;

    // Redraw frame
    ctx.drawImage(state.frameImage, 0, 0, canvas.width, canvas.height);

    if (!box) return;

    // Scale box from video coords to canvas coords
    const scaleX = canvas.width / state.videoWidth;
    const scaleY = canvas.height / state.videoHeight;
    const bx = box.x * scaleX;
    const by = box.y * scaleY;
    const bw = box.width * scaleX;
    const bh = box.height * scaleY;

    // Semi-transparent overlay outside the box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(0, 0, canvas.width, by);
    ctx.fillRect(0, by, bx, bh);
    ctx.fillRect(bx + bw, by, canvas.width - bx - bw, bh);
    ctx.fillRect(0, by + bh, canvas.width, canvas.height - by - bh);

    // Box outline
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);

    // Corner handles
    ctx.fillStyle = '#3b82f6';
    const hs = 5;
    [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]].forEach(([cx, cy]) => {
        ctx.fillRect(cx - hs, cy - hs, hs * 2, hs * 2);
    });
}

function canvasToVideoCoords(canvasX, canvasY) {
    const canvas = $('#box-canvas');
    const scaleX = state.videoWidth / canvas.width;
    const scaleY = state.videoHeight / canvas.height;
    return {
        x: Math.round(canvasX * scaleX),
        y: Math.round(canvasY * scaleY)
    };
}

async function openBoxModal() {
    const videoPath = state.selectedVideo || $('#video-select').value;
    if (!videoPath) {
        ui.showToast('Select a video first', 'error');
        return;
    }

    // Fetch first frame to get video dimensions and total frame count
    try {
        const { blob, videoWidth, videoHeight, totalFrames } = await api.getVideoFrame(videoPath, 0);
        state.videoWidth = videoWidth;
        state.videoHeight = videoHeight;
        _totalFrames = totalFrames;
    } catch (e) {
        ui.showToast('Failed to load video frame', 'error');
        return;
    }

    _currentFrame = 0;
    _modalBox = state.manualBox ? { ...state.manualBox } : null;

    const scrubber = $('#frame-scrubber');
    scrubber.max = Math.max(0, _totalFrames - 1);
    scrubber.value = 0;
    $('#frame-number').textContent = `0 / ${_totalFrames}`;

    // Show modal
    $('#box-modal').classList.remove('hidden');

    // Update confirm button state
    $('#modal-confirm-btn').disabled = !_modalBox;

    // Load first frame into canvas
    await loadFramePreview(0);
}

function closeBoxModal(confirm) {
    if (confirm && _modalBox) {
        state.manualBox = { ..._modalBox };
        const b = state.manualBox;
        $('#box-coords').classList.remove('hidden');
        $('#box-coords-text').textContent = `${b.width}x${b.height} at (${b.x}, ${b.y})`;
    }
    // If cancelled, don't touch state.manualBox

    $('#box-modal').classList.add('hidden');
    _modalBox = null;
    state.frameImage = null;
    ui.updateProcessButton();
}

function setupCanvasListeners() {
    const canvas = $('#box-canvas');

    canvas.addEventListener('mousedown', (e) => {
        if (!state.frameImage) return;
        const rect = canvas.getBoundingClientRect();
        state.isDrawing = true;
        state.drawStart = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.isDrawing || !state.drawStart) return;
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        const x = Math.min(state.drawStart.x, cx);
        const y = Math.min(state.drawStart.y, cy);
        const w = Math.abs(cx - state.drawStart.x);
        const h = Math.abs(cy - state.drawStart.y);

        const topLeft = canvasToVideoCoords(x, y);
        const bottomRight = canvasToVideoCoords(x + w, y + h);
        _modalBox = {
            x: topLeft.x,
            y: topLeft.y,
            width: bottomRight.x - topLeft.x,
            height: bottomRight.y - topLeft.y
        };
        drawBoxOverlay(_modalBox);
    });

    canvas.addEventListener('mouseup', () => {
        if (!state.isDrawing) return;
        state.isDrawing = false;

        if (_modalBox && _modalBox.width > 5 && _modalBox.height > 5) {
            $('#modal-confirm-btn').disabled = false;
        } else {
            _modalBox = null;
            drawBoxOverlay(null);
            $('#modal-confirm-btn').disabled = true;
        }
    });

    canvas.addEventListener('mouseleave', () => {
        if (state.isDrawing) {
            state.isDrawing = false;
        }
    });
}

function setupDetectionModeToggle() {
    const yoloBtn = $('#toggle-yolo');
    const manualBtn = $('#toggle-manual');
    const manualSection = $('#manual-box-section');

    yoloBtn.addEventListener('click', () => {
        state.detectionMode = 'yolo';
        yoloBtn.classList.add('active');
        manualBtn.classList.remove('active');
        manualSection.classList.add('hidden');
        ui.updateProcessButton();
    });

    manualBtn.addEventListener('click', () => {
        state.detectionMode = 'manual';
        manualBtn.classList.add('active');
        yoloBtn.classList.remove('active');
        manualSection.classList.remove('hidden');
        ui.updateProcessButton();
    });

    // Open modal button
    $('#open-box-modal-btn').addEventListener('click', openBoxModal);

    // Modal confirm
    $('#modal-confirm-btn').addEventListener('click', () => closeBoxModal(true));

    // Modal cancel
    $('#modal-cancel-btn').addEventListener('click', () => closeBoxModal(false));

    // Frame scrubber
    let scrubDebounce = null;
    $('#frame-scrubber').addEventListener('input', (e) => {
        _currentFrame = parseInt(e.target.value);
        $('#frame-number').textContent = `${_currentFrame} / ${_totalFrames}`;

        // Debounce frame loading to avoid hammering the API
        clearTimeout(scrubDebounce);
        scrubDebounce = setTimeout(() => {
            loadFramePreview(_currentFrame);
        }, 100);
    });

    // Clear box
    $('#clear-box-btn').addEventListener('click', () => {
        state.manualBox = null;
        $('#box-coords').classList.add('hidden');
        ui.updateProcessButton();
    });

    // Close modal on overlay click
    $('#box-modal').addEventListener('click', (e) => {
        if (e.target === $('#box-modal')) {
            closeBoxModal(false);
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !$('#box-modal').classList.contains('hidden')) {
            closeBoxModal(false);
        }
    });
}


// ============================================
// Initialization
// ============================================

async function init() {
    console.log('🗡️ WaterSlayer initializing...');

    setupEventListeners();
    setupCanvasListeners();
    setupDetectionModeToggle();

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
