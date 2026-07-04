/**
 * Character Datasheet Face Masker — Core Application
 *
 * Handles image loading, batch management, mask rendering, interactive
 * manipulation, and export. Works with face-detection.js for auto-detect.
 */

/* ═══════════════════════════════════════════════════════════
   Application State
   ═══════════════════════════════════════════════════════════ */

const App = {
    // Image data
    images: [],        // Array of ImageEntry objects
    currentIndex: -1,

    // Canvas rendering
    canvas: null,
    ctx: null,
    displayScale: 1,   // ratio: display size / original image size
    canvasOffsetX: 0,   // canvas position within wrapper
    canvasOffsetY: 0,

    // Face detector
    detector: null,

    // Drag interaction state
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragStartMaskX: 0,
    dragStartMaskY: 0,

    // Mask preview toggle
    maskPreviewVisible: true,

    // DOM elements (cached)
    el: {},
};

/**
 * @typedef {Object} ImageEntry
 * @property {File}    file
 * @property {HTMLImageElement} img
 * @property {string}  filename
 * @property {number}  origWidth
 * @property {number}  origHeight
 * @property {MaskParams} mask
 * @property {boolean} applied – true if user has explicitly applied the mask
 */

/**
 * @typedef {Object} MaskParams
 * @property {number} x       – center X in image coords
 * @property {number} y       – center Y in image coords
 * @property {number} width   – full mask width (px in image coords)
 * @property {number} height  – full mask height (px in image coords)
 * @property {number} feather – feather percentage (5–80)
 * @property {number} softness – softness/smoothing (5–100)
 * @property {'auto'|'heuristic'|'manual'} method
 * @property {number} confidence
 */

/* ═══════════════════════════════════════════════════════════
   Initialization
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

App.init = function () {
    this.cacheElements();
    this.canvas = this.el.previewCanvas;
    this.ctx = this.canvas.getContext('2d');
    this.detector = new FaceDetectorModule();

    this.setupEventListeners();
    this.setupDragAndDrop();
    this.setupCanvasInteraction();
    this.setupKeyboardShortcuts();
    this.initDetector();
};

App.cacheElements = function () {
    const q = (s) => document.querySelector(s);
    this.el = {
        previewCanvas: q('#previewCanvas'),
        canvasWrapper: q('#canvasWrapper'),
        dropZone: q('#dropZone'),
        previewContainer: q('#previewContainer'),
        fileInput: q('#fileInput'),
        browseBtn: q('#browseBtn'),

        // Navigation
        prevBtn: q('#prevBtn'),
        nextBtn: q('#nextBtn'),
        navInfo: q('#navInfo'),
        navFilename: q('#navFilename'),
        navDimensions: q('#navDimensions'),
        navBar: q('#navBar'),
        addMoreBtn: q('#addMoreBtn'),
        clearAllBtn: q('#clearAllBtn'),

        // Thumbnails
        thumbnailStrip: q('#thumbnailStrip'),

        // Controls
        controlsSection: q('#controlsSection'),
        detectBtn: q('#detectBtn'),
        detectAllBtn: q('#detectAllBtn'),
        infoMethod: q('#infoMethod'),
        infoConfidence: q('#infoConfidence'),

        maskX: q('#maskX'),
        maskY: q('#maskY'),
        maskW: q('#maskW'),
        maskH: q('#maskH'),
        maskXVal: q('#maskXVal'),
        maskYVal: q('#maskYVal'),
        maskWVal: q('#maskWVal'),
        maskHVal: q('#maskHVal'),

        scaleDownBtn: q('#scaleDownBtn'),
        scaleUpBtn: q('#scaleUpBtn'),
        resampleColorBtn: q('#resampleColorBtn'),

        lockAspect: q('#lockAspect'),

        featherRadius: q('#featherRadius'),
        softness: q('#softness'),
        featherVal: q('#featherVal'),
        softnessVal: q('#softnessVal'),

        resetBtn: q('#resetBtn'),
        applyBtn: q('#applyBtn'),
        eyeDropperBtn: q('#eyeDropperBtn'),

        exportCurrentBtn: q('#exportCurrentBtn'),
        exportAllZipBtn: q('#exportAllZipBtn'),
        exportAllIndivBtn: q('#exportAllIndivBtn'),

        // Status
        statusDot: q('#statusDot'),
        statusText: q('#statusText'),

        // Overlays
        dragOverlay: q('#dragOverlay'),
        loadingOverlay: q('#loadingOverlay'),
        loadingText: q('#loadingText'),
        loadingProgress: q('#loadingProgress'),
        progressBar: q('#progressBar'),
        toastContainer: q('#toastContainer'),

        // Panel guides
        panelGuide1: q('#panelGuide1'),
        panelGuide2: q('#panelGuide2'),
    };
};

App.initDetector = async function () {
    const ok = await this.detector.init((status, detail) => {
        this.el.statusDot.className = 'status-dot status-' + status;
        this.el.statusText.textContent = detail;
    });

    if (!ok) {
        this.showToast('Face detection models unavailable — manual mode active', 'warning');
    }
};

/* ═══════════════════════════════════════════════════════════
   Event Listeners
   ═══════════════════════════════════════════════════════════ */

App.setupEventListeners = function () {
    // File browse
    this.el.browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.el.fileInput.click();
    });
    this.el.dropZone.addEventListener('click', () => this.el.fileInput.click());
    this.el.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) this.handleFiles(e.target.files);
    });
    this.el.addMoreBtn.addEventListener('click', () => {
        this.el.fileInput.value = '';
        this.el.fileInput.click();
    });
    this.el.clearAllBtn.addEventListener('click', () => this.clearAll());

    // Navigation
    this.el.prevBtn.addEventListener('click', () => this.navigate(-1));
    this.el.nextBtn.addEventListener('click', () => this.navigate(1));

    // Detection
    this.el.detectBtn.addEventListener('click', () => this.detectCurrentFace());
    this.el.detectAllBtn.addEventListener('click', () => this.detectAllFaces());

    // Resample
    this.el.resampleColorBtn.addEventListener('click', () => {
        const entry = this.currentEntry();
        if (!entry) return;
        entry.mask.bgColor = this.getBackgroundColor(entry.img, entry.mask.y, entry.mask.x, entry.mask.width);
        entry.mask.method = 'manual';
        this.updateSlidersFromMask(entry.mask);
        this.render();
    });

    // Quick Scale Buttons
    this.el.scaleDownBtn.addEventListener('click', () => {
        const entry = this.currentEntry();
        if (!entry) return;
        entry.mask.width = Math.max(20, entry.mask.width * 0.9);
        entry.mask.height = Math.max(20, entry.mask.height * 0.9);
        entry.mask.method = 'manual';
        this.updateSlidersFromMask(entry.mask);
        this.render();
    });

    this.el.scaleUpBtn.addEventListener('click', () => {
        const entry = this.currentEntry();
        if (!entry) return;
        entry.mask.width = Math.min(1000, entry.mask.width * 1.1);
        entry.mask.height = Math.min(1000, entry.mask.height * 1.1);
        entry.mask.method = 'manual';
        this.updateSlidersFromMask(entry.mask);
        this.render();
    });

    // Eyedropper
    this.el.eyeDropperBtn.addEventListener('click', () => {
        this.isEyeDropping = !this.isEyeDropping;
        if (this.isEyeDropping) {
            this.el.eyeDropperBtn.classList.add('active');
            this.el.previewCanvas.style.cursor = 'crosshair';
        } else {
            this.el.eyeDropperBtn.classList.remove('active');
            this.el.previewCanvas.style.cursor = '';
        }
    });

    // Sliders
    const sliderInputHandler = (slider, display, prop) => {
        slider.addEventListener('input', () => {
            const val = parseInt(slider.value);
            display.textContent = val;
            const entry = this.currentEntry();
            if (!entry) return;

            if (prop === 'width' && this.el.lockAspect.checked) {
                const ratio = entry.mask.height / entry.mask.width;
                entry.mask.width = val;
                entry.mask.height = Math.round(val * ratio);
                this.el.maskH.value = entry.mask.height;
                this.el.maskHVal.textContent = entry.mask.height;
            } else if (prop === 'height' && this.el.lockAspect.checked) {
                const ratio = entry.mask.width / entry.mask.height;
                entry.mask.height = val;
                entry.mask.width = Math.round(val * ratio);
                this.el.maskW.value = entry.mask.width;
                this.el.maskWVal.textContent = entry.mask.width;
            } else {
                entry.mask[prop] = val;
            }

            entry.mask.method = 'manual';
            this.render();
        });
    };

    sliderInputHandler(this.el.maskX, this.el.maskXVal, 'x');
    sliderInputHandler(this.el.maskY, this.el.maskYVal, 'y');
    sliderInputHandler(this.el.maskW, this.el.maskWVal, 'width');
    sliderInputHandler(this.el.maskH, this.el.maskHVal, 'height');
    sliderInputHandler(this.el.featherRadius, this.el.featherVal, 'feather');
    sliderInputHandler(this.el.softness, this.el.softnessVal, 'softness');

    // Actions
    this.el.resetBtn.addEventListener('click', () => this.resetCurrentMask());
    this.el.applyBtn.addEventListener('click', () => this.applyCurrentMask());

    // Export
    this.el.exportCurrentBtn.addEventListener('click', () => this.exportCurrent());
    this.el.exportAllZipBtn.addEventListener('click', () => this.exportAllZip());
    this.el.exportAllIndivBtn.addEventListener('click', () => this.exportAllIndividually());
};

/* ═══════════════════════════════════════════════════════════
   Drag & Drop
   ═══════════════════════════════════════════════════════════ */

App.setupDragAndDrop = function () {
    let dragCounter = 0;

    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };

    document.addEventListener('dragenter', (e) => {
        prevent(e);
        dragCounter++;
        this.el.dragOverlay.classList.add('visible');
    });

    document.addEventListener('dragleave', (e) => {
        prevent(e);
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            this.el.dragOverlay.classList.remove('visible');
        }
    });

    document.addEventListener('dragover', prevent);

    document.addEventListener('drop', (e) => {
        prevent(e);
        dragCounter = 0;
        this.el.dragOverlay.classList.remove('visible');

        if (e.dataTransfer.files.length > 0) {
            this.handleFiles(e.dataTransfer.files);
        }
    });
};

/* ═══════════════════════════════════════════════════════════
   Canvas Interaction (click-to-place + drag mask)
   ═══════════════════════════════════════════════════════════ */

App.setupCanvasInteraction = function () {
    const canvas = this.el.previewCanvas;

    // Track whether the current mousedown resulted in a drag
    this._wasDragging = false;

    canvas.addEventListener('mousedown', (e) => {
        const entry = this.currentEntry();
        if (!entry) return;

        const { imgX, imgY } = this.canvasToImage(e);
        const m = entry.mask;

        this._wasDragging = false;

        // Check if click is within the mask ellipse → start drag
        const dx = (imgX - m.x) / (m.width / 2);
        const dy = (imgY - m.y) / (m.height / 2);
        if (dx * dx + dy * dy <= 1.5) { // generous hit area
            this.isDragging = true;
            this.dragStartX = imgX;
            this.dragStartY = imgY;
            this.dragStartMaskX = m.x;
            this.dragStartMaskY = m.y;
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!this.isDragging) return;
        this._wasDragging = true;
        const entry = this.currentEntry();
        if (!entry) return;

        const { imgX, imgY } = this.canvasToImage(e);
        entry.mask.x = this.dragStartMaskX + (imgX - this.dragStartX);
        entry.mask.y = this.dragStartMaskY + (imgY - this.dragStartY);
        entry.mask.method = 'manual';

        this.updateSlidersFromMask(entry.mask);
        this.render();
    });

    window.addEventListener('mouseup', () => {
        if (this.isDragging) {
            this.isDragging = false;
            this.el.previewCanvas.style.cursor = 'crosshair';
        }
    });

    // Click-to-place: clicking on the LEFT PANEL (outside the mask) moves the mask there
    canvas.addEventListener('click', (e) => {
        // Skip if this was a drag operation
        if (this._wasDragging) {
            this._wasDragging = false;
            return;
        }

        const entry = this.currentEntry();
        if (!entry) return;

        const { imgX, imgY } = this.canvasToImage(e);

        // Handle EyeDropper Mode
        if (this.isEyeDropping) {
            // Draw original image to temp canvas to sample exact pixel
            const tmp = document.createElement('canvas');
            tmp.width = 1;
            tmp.height = 1;
            const tCtx = tmp.getContext('2d');
            tCtx.drawImage(entry.img, imgX, imgY, 1, 1, 0, 0, 1, 1);
            const p = tCtx.getImageData(0, 0, 1, 1).data;
            
            entry.mask.bgColor = { r: p[0], g: p[1], b: p[2] };
            entry.mask.method = 'manual';
            
            this.isEyeDropping = false;
            this.el.eyeDropperBtn.classList.remove('active');
            this.el.previewCanvas.style.cursor = '';
            
            this.updateSlidersFromMask(entry.mask);
            this.render();
            return;
        }

        const m = entry.mask;

        // Don't re-place if clicking inside the existing mask
        const dx = (imgX - m.x) / (m.width / 2);
        const dy = (imgY - m.y) / (m.height / 2);
        if (dx * dx + dy * dy <= 1.2) return;

        // Only place within the left panel (first third of image)
        const panelW = Math.floor(entry.origWidth / 3);
        if (imgX < 0 || imgX > panelW || imgY < 0 || imgY > entry.origHeight) return;

        // Move mask center to clicked position
        entry.mask.x = imgX;
        entry.mask.y = imgY;
        entry.mask.method = 'manual';

        this.updateSlidersFromMask(entry.mask);
        this.updateDetectionInfo(entry.mask);
        this.render();
        this.showToast('Mask placed — drag to fine-tune', 'info');
    });

    // Hover cursor
    canvas.addEventListener('mousemove', (e) => {
        if (this.isDragging) return;
        const entry = this.currentEntry();
        if (!entry) return;

        const { imgX, imgY } = this.canvasToImage(e);
        const m = entry.mask;
        const dx = (imgX - m.x) / (m.width / 2);
        const dy = (imgY - m.y) / (m.height / 2);

        // Show different cursors based on position
        const panelW = Math.floor(entry.origWidth / 3);
        if (dx * dx + dy * dy <= 1.5) {
            canvas.style.cursor = 'grab';
        } else if (imgX >= 0 && imgX <= panelW) {
            canvas.style.cursor = 'crosshair'; // indicates click-to-place is available
        } else {
            canvas.style.cursor = 'default';
        }
    });
};

/**
 * Map canvas mouse event coordinates to original image coordinates.
 */
App.canvasToImage = function (e) {
    const rect = this.el.previewCanvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    return {
        imgX: canvasX / this.displayScale,
        imgY: canvasY / this.displayScale,
    };
};

/* ═══════════════════════════════════════════════════════════
   Keyboard Shortcuts
   ═══════════════════════════════════════════════════════════ */

App.setupKeyboardShortcuts = function () {
    document.addEventListener('keydown', (e) => {
        // Don't trigger shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        const entry = this.currentEntry();
        const nudge = e.shiftKey ? 10 : 2;

        switch (e.key) {
            case 'ArrowLeft':
                if (e.shiftKey && entry) {
                    entry.mask.x -= nudge;
                    entry.mask.method = 'manual';
                    this.updateSlidersFromMask(entry.mask);
                    this.render();
                } else {
                    this.navigate(-1);
                }
                e.preventDefault();
                break;

            case 'ArrowRight':
                if (e.shiftKey && entry) {
                    entry.mask.x += nudge;
                    entry.mask.method = 'manual';
                    this.updateSlidersFromMask(entry.mask);
                    this.render();
                } else {
                    this.navigate(1);
                }
                e.preventDefault();
                break;

            case 'ArrowUp':
                if (entry) {
                    entry.mask.y -= nudge;
                    entry.mask.method = 'manual';
                    this.updateSlidersFromMask(entry.mask);
                    this.render();
                    e.preventDefault();
                }
                break;

            case 'ArrowDown':
                if (entry) {
                    entry.mask.y += nudge;
                    entry.mask.method = 'manual';
                    this.updateSlidersFromMask(entry.mask);
                    this.render();
                    e.preventDefault();
                }
                break;

            case ' ': // Space
                this.maskPreviewVisible = !this.maskPreviewVisible;
                this.render();
                e.preventDefault();
                break;

            case 'd':
            case 'D':
                this.detectCurrentFace();
                e.preventDefault();
                break;

            case 'Enter':
                this.applyCurrentMask();
                e.preventDefault();
                break;

            case 'e':
            case 'E':
                this.exportCurrent();
                e.preventDefault();
                break;
        }
    });
};

/* ═══════════════════════════════════════════════════════════
   Image Loading & Management
   ═══════════════════════════════════════════════════════════ */

App.handleFiles = function (fileList) {
    const validTypes = ['image/png', 'image/jpeg', 'image/webp'];
    const files = Array.from(fileList).filter(f => validTypes.includes(f.type));

    if (files.length === 0) {
        this.showToast('No valid image files found (PNG, JPG, WebP)', 'error');
        return;
    }

    const isFirst = this.images.length === 0;
    let loaded = 0;

    this.showLoading(`Loading ${files.length} image${files.length > 1 ? 's' : ''}…`);

    files.forEach((file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.images.push({
                    file: file,
                    img: img,
                    filename: file.name,
                    origWidth: img.naturalWidth,
                    origHeight: img.naturalHeight,
                    mask: this.getDefaultMask(img),
                    applied: false,
                });

                loaded++;
                if (loaded === files.length) {
                    this.hideLoading();
                    if (isFirst) {
                        this.navigateTo(0);
                    } else {
                        this.updateThumbnails();
                        this.updateNavigation();
                    }
                    this.enableControls();
                    this.showToast(`Loaded ${files.length} datasheet${files.length > 1 ? 's' : ''}`, 'success');

                    // Auto-detect faces on newly loaded images
                    this.detectAllFaces();
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
};

App.getBackgroundColor = function (img, faceY, faceX, faceW) {
    const sampleW = 10;
    const sampleH = 10;
    const c = document.createElement('canvas');
    c.width = sampleW;
    c.height = sampleH;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    
    const panelW = Math.floor(img.naturalWidth / 3);
    
    // Default to the right side of the panel if we don't have face info
    let startXRight = Math.max(15, panelW - sampleW - 15);
    
    if (faceX !== undefined && faceW !== undefined) {
        // Go 2.0x face widths to the right of the face center
        startXRight = Math.min(panelW - sampleW - 5, Math.floor(faceX + (faceW * 2.0)));
    }
    
    const startY = faceY ? Math.max(15, Math.floor(faceY - 5)) : 15;
    
    // Get right sample ONLY to avoid left-side hair/vignettes
    ctx.drawImage(img, startXRight, startY, sampleW, sampleH, 0, 0, sampleW, sampleH);
    const dataR = ctx.getImageData(0, 0, sampleW, sampleH).data;
    
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < dataR.length; i += 4) {
        r += dataR[i];
        g += dataR[i+1];
        b += dataR[i+2];
    }
    const count = dataR.length / 4;
    return { r: Math.round(r/count), g: Math.round(g/count), b: Math.round(b/count) };
};

App.getDefaultMask = function (img) {
    const imgW = img.naturalWidth;
    const imgH = img.naturalHeight;
    // Default heuristic position for the left panel
    const panel = this.getPanelBounds(imgW, imgH);
    
    let pos;
    if (this.detector && this.detector.getHeuristicPosition) {
        pos = this.detector.getHeuristicPosition(panel);
    } else {
        pos = {
            x: panel.x + panel.w * 0.50,
            y: panel.y + panel.h * 0.14,
            width: panel.w * 0.25, // default size bumped up
            height: panel.h * 0.12,
        };
    }
    
    return {
        bgColor: this.getBackgroundColor(img, pos.y, pos.x, pos.width),
        x: pos.x,
        y: pos.y,
        width: pos.width,
        height: pos.height,
        feather: 40,
        softness: 30,
        method: 'heuristic',
        confidence: 0,
    };
};

App.getPanelBounds = function (imgW, imgH) {
    // The left panel is the first third of the image
    return {
        x: 0,
        y: 0,
        w: Math.floor(imgW / 3),
        h: imgH,
    };
};

App.currentEntry = function () {
    if (this.currentIndex < 0 || this.currentIndex >= this.images.length) return null;
    return this.images[this.currentIndex];
};

App.navigateTo = function (index) {
    if (index < 0 || index >= this.images.length) return;
    this.currentIndex = index;

    // Show preview, hide drop zone
    this.el.dropZone.style.display = 'none';
    this.el.previewContainer.classList.add('visible');

    this.updateNavigation();
    this.updateThumbnails();
    this.updateSlidersFromMask(this.currentEntry().mask);
    this.updateDetectionInfo(this.currentEntry().mask);
    this.render();
};

App.navigate = function (delta) {
    const newIndex = this.currentIndex + delta;
    if (newIndex >= 0 && newIndex < this.images.length) {
        this.navigateTo(newIndex);
    }
};

App.clearAll = function () {
    this.images = [];
    this.currentIndex = -1;
    this.el.previewContainer.classList.remove('visible');
    this.el.dropZone.style.display = '';
    this.el.thumbnailStrip.classList.remove('visible');
    this.el.thumbnailStrip.innerHTML = '';
    this.disableControls();
    this.showToast('All images cleared', 'info');
};

/* ═══════════════════════════════════════════════════════════
   Navigation & Thumbnails
   ═══════════════════════════════════════════════════════════ */

App.updateNavigation = function () {
    const total = this.images.length;
    const current = this.currentIndex + 1;
    const entry = this.currentEntry();

    this.el.navInfo.textContent = `${current} / ${total}`;
    this.el.navFilename.textContent = entry ? entry.filename : '—';
    this.el.navDimensions.textContent = entry
        ? `${entry.origWidth} × ${entry.origHeight}`
        : '';

    this.el.prevBtn.disabled = this.currentIndex <= 0;
    this.el.nextBtn.disabled = this.currentIndex >= total - 1;
};

App.updateThumbnails = function () {
    const strip = this.el.thumbnailStrip;
    strip.innerHTML = '';

    if (this.images.length <= 1) {
        strip.classList.remove('visible');
        return;
    }

    strip.classList.add('visible');

    this.images.forEach((entry, i) => {
        const thumb = document.createElement('div');
        thumb.className = 'thumb-item' +
            (i === this.currentIndex ? ' active' : '') +
            (entry.applied ? ' applied' : '');
        thumb.addEventListener('click', () => this.navigateTo(i));

        const img = document.createElement('img');
        img.src = entry.img.src;
        img.alt = entry.filename;
        thumb.appendChild(img);

        if (entry.applied) {
            const badge = document.createElement('div');
            badge.className = 'thumb-badge badge-applied';
            badge.textContent = '✓';
            thumb.appendChild(badge);
        }

        strip.appendChild(thumb);
    });

    // Scroll active thumbnail into view
    const active = strip.querySelector('.active');
    if (active) {
        active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
};

/* ═══════════════════════════════════════════════════════════
   Controls Management
   ═══════════════════════════════════════════════════════════ */

App.enableControls = function () {
    const sliders = this.el.controlsSection.querySelectorAll('.control-slider');
    sliders.forEach(s => s.disabled = false);

    this.el.detectBtn.disabled = false;
    this.el.detectAllBtn.disabled = false;
    this.el.resampleColorBtn.disabled = false;
    this.el.scaleDownBtn.disabled = false;
    this.el.scaleUpBtn.disabled = false;
    this.el.eyeDropperBtn.disabled = false;
    this.el.resetBtn.disabled = false;
    this.el.applyBtn.disabled = false;
    this.el.exportCurrentBtn.disabled = false;
    this.el.exportAllZipBtn.disabled = this.images.length < 2;
    this.el.exportAllIndivBtn.disabled = this.images.length < 2;
};

App.disableControls = function () {
    const sliders = this.el.controlsSection.querySelectorAll('.control-slider');
    sliders.forEach(s => s.disabled = true);

    this.el.detectBtn.disabled = true;
    this.el.detectAllBtn.disabled = true;
    this.el.resampleColorBtn.disabled = true;
    this.el.scaleDownBtn.disabled = true;
    this.el.scaleUpBtn.disabled = true;
    this.el.eyeDropperBtn.disabled = true;
    this.el.resetBtn.disabled = true;
    this.el.applyBtn.disabled = true;
    this.el.exportCurrentBtn.disabled = true;
    this.el.exportAllZipBtn.disabled = true;
    this.el.exportAllIndivBtn.disabled = true;
};

App.updateSlidersFromMask = function (mask) {
    this.el.maskX.value = Math.round(mask.x);
    this.el.maskY.value = Math.round(mask.y);
    this.el.maskW.value = Math.round(mask.width);
    this.el.maskH.value = Math.round(mask.height);
    this.el.featherRadius.value = mask.feather;
    this.el.softness.value = mask.softness;

    this.el.maskXVal.textContent = Math.round(mask.x);
    this.el.maskYVal.textContent = Math.round(mask.y);
    this.el.maskWVal.textContent = Math.round(mask.width);
    this.el.maskHVal.textContent = Math.round(mask.height);
    this.el.featherVal.textContent = mask.feather;
    this.el.softnessVal.textContent = mask.softness;

    // Update slider ranges based on image dimensions
    const entry = this.currentEntry();
    if (entry) {
        const panelW = Math.floor(entry.origWidth / 3);
        this.el.maskX.max = panelW;
        this.el.maskY.max = entry.origHeight;
        this.el.maskW.max = panelW;
        this.el.maskH.max = Math.floor(entry.origHeight / 2);
    }
};

App.updateDetectionInfo = function (mask) {
    this.el.infoMethod.textContent = mask.method;
    this.el.infoConfidence.textContent = mask.confidence > 0
        ? (mask.confidence * 100).toFixed(1) + '%'
        : '—';
};

/* ═══════════════════════════════════════════════════════════
   Face Detection
   ═══════════════════════════════════════════════════════════ */

App.detectCurrentFace = async function () {
    const entry = this.currentEntry();
    if (!entry) return;

    this.showLoading('Detecting face…');

    const panel = this.getPanelBounds(entry.origWidth, entry.origHeight);
    const result = await this.detector.detect(entry.img, panel);
    console.log('Detection pipeline result:', result);

    entry.mask.x = result.x;
    entry.mask.y = result.y;
    entry.mask.width = result.width;
    entry.mask.height = result.height;
    entry.mask.feather = result.feather;
    entry.mask.softness = result.softness;
    entry.mask.method = result.method;
    entry.mask.confidence = result.confidence;
    entry.applied = false;

    this.updateSlidersFromMask(entry.mask);
    this.updateDetectionInfo(entry.mask);
    this.render();
    this.hideLoading();

    let msg, type;
    if (result.method === 'auto') {
        msg = `Face detected (${(result.confidence * 100).toFixed(0)}% confidence)`;
        type = 'success';
    } else if (result.method === 'existing-mask') {
        msg = `Existing mask detected — repositioned for soft brush`;
        type = 'success';
    } else {
        msg = 'Using heuristic position — click on the face to adjust';
        type = 'warning';
    }
    this.showToast(msg, type);
};

App.detectAllFaces = async function () {
    if (this.images.length === 0) return;

    this.showLoading(`Detecting faces (0 / ${this.images.length})…`);
    this.el.loadingProgress.classList.add('visible');

    for (let i = 0; i < this.images.length; i++) {
        const entry = this.images[i];
        this.el.loadingText.textContent = `Detecting faces (${i + 1} / ${this.images.length})…`;
        this.el.progressBar.style.width = `${((i + 1) / this.images.length) * 100}%`;

        const panel = this.getPanelBounds(entry.origWidth, entry.origHeight);
        const result = await this.detector.detect(entry.img, panel);

        entry.mask.x = result.x;
        entry.mask.y = result.y;
        entry.mask.width = result.width;
        entry.mask.height = result.height;
        entry.mask.feather = result.feather;
        entry.mask.softness = result.softness;
        entry.mask.method = result.method;
        entry.mask.confidence = result.confidence;
    }

    this.el.loadingProgress.classList.remove('visible');
    this.hideLoading();

    // Refresh current view
    const entry = this.currentEntry();
    if (entry) {
        this.updateSlidersFromMask(entry.mask);
        this.updateDetectionInfo(entry.mask);
        this.render();
    }

    this.showToast(`Detected faces on ${this.images.length} datasheet${this.images.length > 1 ? 's' : ''}`, 'success');
};

/* ═══════════════════════════════════════════════════════════
   Mask Actions
   ═══════════════════════════════════════════════════════════ */

App.resetCurrentMask = function () {
    const entry = this.currentEntry();
    if (!entry) return;

    entry.mask = this.getDefaultMask(entry.img);
    entry.applied = false;

    this.updateSlidersFromMask(entry.mask);
    this.updateDetectionInfo(entry.mask);
    this.updateThumbnails();
    this.render();

    this.showToast('Mask reset to default', 'info');
};

App.applyCurrentMask = function () {
    const entry = this.currentEntry();
    if (!entry) return;

    entry.applied = true;
    this.updateThumbnails();
    this.showToast(`Mask applied to ${entry.filename}`, 'success');
};

/* ═══════════════════════════════════════════════════════════
   Canvas Rendering
   ═══════════════════════════════════════════════════════════ */

App.render = function () {
    const entry = this.currentEntry();
    if (!entry) return;

    const wrapper = this.el.canvasWrapper;
    const wrapW = wrapper.clientWidth;
    const wrapH = wrapper.clientHeight;

    // Calculate scale to fit image in wrapper with padding
    const pad = 24;
    const availW = wrapW - pad * 2;
    const availH = wrapH - pad * 2;
    this.displayScale = Math.min(availW / entry.origWidth, availH / entry.origHeight, 1);

    const dispW = Math.round(entry.origWidth * this.displayScale);
    const dispH = Math.round(entry.origHeight * this.displayScale);

    // Set canvas size to display size
    this.canvas.width = dispW;
    this.canvas.height = dispH;
    this.canvas.style.width = dispW + 'px';
    this.canvas.style.height = dispH + 'px';

    // Draw image
    this.ctx.drawImage(entry.img, 0, 0, dispW, dispH);

    // Draw mask overlay
    if (this.maskPreviewVisible) {
        this.renderMaskOnCtx(this.ctx, entry.mask, this.displayScale);
    }

    // Draw mask boundary indicator (dashed ellipse)
    this.renderMaskBoundary(this.ctx, entry.mask, this.displayScale);

    // Update panel guide positions
    this.updatePanelGuides(dispW, dispH);
};

/**
 * Render the soft elliptical mask.
 *
 * The mask simulates a soft brush:
 * - Core region is solid #808080 (fully opaque)
 * - Feather zone transitions smoothly from opaque to transparent
 * - Uses multiple concentric ellipses for smooth gradient (Canvas doesn't
 *   natively support elliptical radial gradients)
 */
App.renderMaskOnCtx = function (ctx, mask, scale) {
    const cx = mask.x * scale;
    const cy = mask.y * scale;
    const radiusX = (mask.width / 2) * scale;
    const radiusY = (mask.height / 2) * scale;
    const featherPct = mask.feather / 100;  // 0–0.8
    const softness = mask.softness / 100;   // 0–1

    // Inner radii (fully opaque core)
    const innerRX = radiusX * (1 - featherPct);
    const innerRY = radiusY * (1 - featherPct);

    // Outer radii (mask extends beyond for visible feathering)
    const outerRX = radiusX;
    const outerRY = radiusY;

    ctx.save();

    // ---- Method: scale trick for elliptical radial gradient ----
    // Transform so we can use a circular gradient that maps to an ellipse
    ctx.save();
    ctx.translate(cx, cy);
    const ratio = outerRY / outerRX;
    ctx.scale(1, ratio);

    // Adjusted inner/outer radii in the scaled coordinate space
    const scaledInnerR = innerRX;
    const scaledOuterR = outerRX;

    // Create gradient from inner edge to outer edge
    const grad = ctx.createRadialGradient(0, 0, scaledInnerR, 0, 0, scaledOuterR);

    // The softness parameter controls the falloff curve shape
    // Lower softness = more gradual falloff (softer brush)
    // Higher softness = more concentrated center (harder edge feel)
    const softPower = 0.3 + softness * 1.2; // maps 0–1 → 0.3–1.5

    const steps = 16;
    const bg = mask.bgColor || { r: 128, g: 128, b: 128 };
    for (let i = 0; i <= steps; i++) {
        const t = i / steps; // 0 (inner edge) → 1 (outer edge)
        const alpha = 1 - Math.pow(t, softPower);
        grad.addColorStop(t, `rgba(${bg.r}, ${bg.g}, ${bg.b}, ${alpha.toFixed(4)})`);
    }

    // Draw the feather zone
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, scaledOuterR, 0, Math.PI * 2);
    ctx.fill();

    // Draw the solid core
    ctx.fillStyle = `rgb(${bg.r}, ${bg.g}, ${bg.b})`;
    ctx.beginPath();
    ctx.arc(0, 0, scaledInnerR, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
    ctx.restore();
};

/**
 * Draw a dashed ellipse boundary around the mask for visual reference.
 */
App.renderMaskBoundary = function (ctx, mask, scale) {
    const cx = mask.x * scale;
    const cy = mask.y * scale;
    const rx = (mask.width / 2) * scale;
    const ry = (mask.height / 2) * scale;

    ctx.save();
    ctx.strokeStyle = 'rgba(124, 107, 240, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);

    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw center crosshair
    const ch = 8;
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(124, 107, 240, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - ch, cy);
    ctx.lineTo(cx + ch, cy);
    ctx.moveTo(cx, cy - ch);
    ctx.lineTo(cx, cy + ch);
    ctx.stroke();

    ctx.restore();
};

App.updatePanelGuides = function (dispW, dispH) {
    const thirdW = dispW / 3;
    const canvasRect = this.canvas.getBoundingClientRect();
    const wrapperRect = this.el.canvasWrapper.getBoundingClientRect();

    const canvasLeft = canvasRect.left - wrapperRect.left;

    this.el.panelGuide1.style.left = (canvasLeft + thirdW) + 'px';
    this.el.panelGuide2.style.left = (canvasLeft + thirdW * 2) + 'px';
    this.el.panelGuide1.classList.add('visible');
    this.el.panelGuide2.classList.add('visible');
};

/* ═══════════════════════════════════════════════════════════
   Full-Resolution Export Rendering
   ═══════════════════════════════════════════════════════════ */

/**
 * Render a masked datasheet at original resolution.
 * @param {ImageEntry} entry
 * @returns {HTMLCanvasElement}
 */
App.renderFullRes = function (entry) {
    const c = document.createElement('canvas');
    c.width = entry.origWidth;
    c.height = entry.origHeight;
    const ctx = c.getContext('2d');

    // Draw original image at full resolution
    ctx.drawImage(entry.img, 0, 0, entry.origWidth, entry.origHeight);

    // Apply mask at full resolution (scale = 1)
    this.renderMaskOnCtx(ctx, entry.mask, 1);

    return c;
};

/**
 * Convert a canvas to a PNG Blob.
 * @param {HTMLCanvasElement} canvas
 * @returns {Promise<Blob>}
 */
App.canvasToBlob = function (canvas) {
    return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), 'image/png');
    });
};

/**
 * Generate the output filename.
 * @param {string} original – e.g. "character_01.png"
 * @returns {string} – e.g. "character_01_masked.png"
 */
App.getMaskedFilename = function (original) {
    const dot = original.lastIndexOf('.');
    if (dot === -1) return original + '_masked.png';
    const name = original.substring(0, dot);
    return name + '_masked.png';
};

/* ═══════════════════════════════════════════════════════════
   Export Functions
   ═══════════════════════════════════════════════════════════ */

App.exportCurrent = async function () {
    const entry = this.currentEntry();
    if (!entry) return;

    this.showLoading('Exporting…');

    try {
        const fullCanvas = this.renderFullRes(entry);
        const blob = await this.canvasToBlob(fullCanvas);
        const filename = this.getMaskedFilename(entry.filename);

        this.downloadBlob(blob, filename);
        entry.applied = true;
        this.updateThumbnails();

        this.hideLoading();
        this.showToast(`Exported: ${filename}`, 'success');
    } catch (err) {
        this.hideLoading();
        this.showToast('Export failed: ' + err.message, 'error');
    }
};

App.exportAllZip = async function () {
    if (this.images.length === 0) return;

    // Check if JSZip is available, load if needed
    if (typeof JSZip === 'undefined') {
        this.showLoading('Loading ZIP library…');
        await this.loadScript('https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js');
    }
    // Check if FileSaver is available, load if needed
    if (typeof saveAs === 'undefined') {
        await this.loadScript('https://cdn.jsdelivr.net/npm/file-saver@2.0.5/dist/FileSaver.min.js');
    }

    this.showLoading(`Exporting ${this.images.length} images as ZIP…`);
    this.el.loadingProgress.classList.add('visible');

    try {
        const zip = new JSZip();

        for (let i = 0; i < this.images.length; i++) {
            const entry = this.images[i];
            this.el.loadingText.textContent = `Rendering ${i + 1} / ${this.images.length}…`;
            this.el.progressBar.style.width = `${((i + 1) / this.images.length) * 100}%`;

            const fullCanvas = this.renderFullRes(entry);
            const blob = await this.canvasToBlob(fullCanvas);
            const filename = this.getMaskedFilename(entry.filename);
            zip.file(filename, blob);

            entry.applied = true;

            // Yield to UI thread
            await new Promise(r => setTimeout(r, 10));
        }

        this.el.loadingText.textContent = 'Creating ZIP file…';
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        saveAs(zipBlob, 'datasheets_masked.zip');

        this.el.loadingProgress.classList.remove('visible');
        this.hideLoading();
        this.updateThumbnails();
        this.showToast(`Exported ${this.images.length} datasheets as ZIP`, 'success');
    } catch (err) {
        this.el.loadingProgress.classList.remove('visible');
        this.hideLoading();
        this.showToast('ZIP export failed: ' + err.message, 'error');
    }
};

App.exportAllIndividually = async function () {
    if (this.images.length === 0) return;

    this.showLoading(`Exporting ${this.images.length} images individually…`);
    this.el.loadingProgress.classList.add('visible');

    try {
        for (let i = 0; i < this.images.length; i++) {
            const entry = this.images[i];
            this.el.loadingText.textContent = `Exporting ${i + 1} / ${this.images.length}…`;
            this.el.progressBar.style.width = `${((i + 1) / this.images.length) * 100}%`;

            const fullCanvas = this.renderFullRes(entry);
            const blob = await this.canvasToBlob(fullCanvas);
            const filename = this.getMaskedFilename(entry.filename);

            this.downloadBlob(blob, filename);
            entry.applied = true;

            // Small delay between downloads so browser doesn't block them
            await new Promise(r => setTimeout(r, 300));
        }

        this.el.loadingProgress.classList.remove('visible');
        this.hideLoading();
        this.updateThumbnails();
        this.showToast(`Exported ${this.images.length} datasheets individually`, 'success');
    } catch (err) {
        this.el.loadingProgress.classList.remove('visible');
        this.hideLoading();
        this.showToast('Export failed: ' + err.message, 'error');
    }
};

/* ═══════════════════════════════════════════════════════════
   Utility Functions
   ═══════════════════════════════════════════════════════════ */

App.downloadBlob = function (blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
};

App.loadScript = function (src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Failed to load: ${src}`));
        document.head.appendChild(script);
    });
};

App.showLoading = function (text) {
    this.el.loadingText.textContent = text || 'Processing…';
    this.el.loadingOverlay.classList.add('visible');
};

App.hideLoading = function () {
    this.el.loadingOverlay.classList.remove('visible');
    this.el.progressBar.style.width = '0';
};

App.showToast = function (message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    this.el.toastContainer.appendChild(toast);

    // Auto-remove after animation
    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 4000);
};
