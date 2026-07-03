/**
 * Face Detection Module (Perfect Synthesis Pipeline)
 * 
 * Uses a 5-pass algorithm:
 * 1. Figure Isolation (Canvas Center of Mass)
 * 2. Anatomical Proportion Deduction
 * 3. Existing Mask Detection (HSV Thresholding)
 * 4. ML Face Detection (MediaPipe Face Landmarker)
 * 5. Algorithmic Synthesis
 */

class FaceDetectorModule {
    constructor() {
        this.isLoaded = false;
        this.isLoading = false;
        this.faceLandmarker = null;
        this._onStatus = null;
    }

    /* ═══════════════════════════════════════════════════════════
       1. Initialization (MediaPipe)
       ═══════════════════════════════════════════════════════════ */

    async init(onStatus) {
        if (this.isLoaded) return true;
        if (this.isLoading) return false;

        this._onStatus = onStatus || (() => {});
        this.isLoading = true;
        this._onStatus('loading', 'Loading MediaPipe Vision Tasks…');

        try {
            // Load MediaPipe from CDN if not present
            if (typeof window.FilesetResolver === 'undefined') {
                await this._loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js');
            }

            this._onStatus('loading', 'Initializing Face Landmarker model…');
            
            // Note: MediaPipe tasks-vision creates global objects or module exports.
            // When loaded via script tag, it puts classes on window.
            const vision = window; 
            
            const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );

            this.faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    delegate: "GPU"
                },
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: true,
                runningMode: "IMAGE",
                numFaces: 1
            });

            this.isLoaded = true;
            this.isLoading = false;
            this._onStatus('ready', 'Advanced Face Detection ready');
            return true;
        } catch (e) {
            console.warn('Failed to load MediaPipe:', e);
            this.isLoading = false;
            this._onStatus('fallback', 'ML Unavailable — using Heuristic pipeline');
            return false;
        }
    }

    _loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.crossOrigin = 'anonymous';
            script.onload = resolve;
            script.onerror = () => reject(new Error('Failed to load: ' + src));
            document.head.appendChild(script);
        });
    }

    /* ═══════════════════════════════════════════════════════════
       Main Detection Pipeline
       ═══════════════════════════════════════════════════════════ */

    async detect(source, panel, imgOrigWidth, imgOrigHeight) {
        // Prepare temporary canvas for the panel region
        const panelCanvas = document.createElement('canvas');
        panelCanvas.width = panel.w;
        panelCanvas.height = panel.h;
        const ctx = panelCanvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(source, panel.x, panel.y, panel.w, panel.h, 0, 0, panel.w, panel.h);

        // 1. Figure Isolation
        const figure = this._isolateFigure(panelCanvas);

        // 2. Anatomical Proportions Fallback
        const anatomy = this._deduceAnatomy(figure, panel.h);

        // 3. Existing Mask Detection
        const existingMask = this._detectExistingMask(panelCanvas, anatomy);

        // 4. ML Face Detection
        let mlFace = null;
        if (this.isLoaded && this.faceLandmarker) {
            mlFace = await this._detectWithMediaPipe(panelCanvas);
        }

        // 5. Algorithmic Synthesis
        const result = this._synthesize(mlFace, existingMask, anatomy, panelCanvas.width, panelCanvas.height);
        
        // Translate back to absolute image coordinates
        return {
            x: panel.x + result.cx,
            y: panel.y + result.cy,
            width: result.width,
            height: result.height,
            feather: result.feather,
            softness: 30, // constant ideal softness
            confidence: result.confidence,
            method: result.method
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 1: Figure Isolation (Pure Canvas API)
       ═══════════════════════════════════════════════════════════ */

    _isolateFigure(canvas) {
        const width = canvas.width;
        const height = canvas.height;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.getImageData(0, 0, width, height).data;

        // Sample background from top-left (0,0)
        const bgR = imgData[0];
        const bgG = imgData[1];
        const bgB = imgData[2];

        let minX = width, maxX = -1, minY = height, maxY = -1;
        let sumX = 0, count = 0;
        const threshold = 25; // Tolerance for non-background

        const isChar = (r, g, b) => {
            return Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB) > threshold;
        };

        // Pass A: Find global bounding box
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                if (isChar(imgData[i], imgData[i+1], imgData[i+2])) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    sumX += x;
                    count++;
                }
            }
        }

        if (count === 0) return null;

        // Pass B: Center of Mass for Head (top 15% of figure)
        const charHeight = maxY - minY;
        const headThresholdY = minY + Math.floor(charHeight * 0.15);
        let headSumX = 0, headCount = 0;

        for (let y = minY; y <= headThresholdY; y++) {
            for (let x = minX; x <= maxX; x++) {
                const i = (y * width + x) * 4;
                if (isChar(imgData[i], imgData[i+1], imgData[i+2])) {
                    headSumX += x;
                    headCount++;
                }
            }
        }

        return {
            top: minY,
            bottom: maxY,
            height: charHeight,
            headCenterX: headCount > 0 ? headSumX / headCount : sumX / count,
            bg: { r: bgR, g: bgG, b: bgB }
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 2: Anatomical Proportion Deduction
       ═══════════════════════════════════════════════════════════ */

    _deduceAnatomy(figure, panelHeight) {
        if (!figure) {
            // Absolute fallback if canvas is blank
            return {
                cx: panelHeight * 0.5, // approx middle of panel
                cy: panelHeight * 0.2, // approx head height
                w: panelHeight * 0.15,
                h: panelHeight * 0.15
            };
        }

        // Based on 8-head Loomis canon relative to total figure height
        const H = figure.height;
        return {
            cx: figure.headCenterX,
            cy: figure.top + (H * 0.0625), // 6.25% from top
            w: H * 0.0833,                 // 8.33% wide
            h: H * 0.125                   // 12.5% tall
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 3: Existing Mask Detection
       ═══════════════════════════════════════════════════════════ */

    _detectExistingMask(canvas, anatomy) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const data = ctx.getImageData(0, 0, width, height).data;

        // Scan only within the anatomical head bounding box
        // Expanded slightly to ensure we capture the whole mask
        const scanTop = Math.max(0, Math.floor(anatomy.cy - anatomy.h * 0.8));
        const scanBottom = Math.min(height - 1, Math.floor(anatomy.cy + anatomy.h * 0.8));
        const scanLeft = Math.max(0, Math.floor(anatomy.cx - anatomy.w * 0.8));
        const scanRight = Math.min(width - 1, Math.floor(anatomy.cx + anatomy.w * 0.8));

        let grayPixelsX = [];
        let grayPixelsY = [];

        for (let y = scanTop; y <= scanBottom; y++) {
            for (let x = scanLeft; x <= scanRight; x++) {
                const i = (y * width + x) * 4;
                const r = data[i], g = data[i+1], b = data[i+2];

                // Simple grayscale detection (low saturation check)
                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                const saturation = max === 0 ? 0 : (max - min) / max;

                // Typical gray masks have near-zero saturation and are not pure black/white
                if (saturation < 0.15 && max > 50 && max < 200) {
                    grayPixelsX.push(x);
                    grayPixelsY.push(y);
                }
            }
        }

        // If we found a significant clump of gray pixels
        if (grayPixelsX.length > (anatomy.w * anatomy.h * 0.1)) {
            // Sort to find bounds (ignoring extreme outliers)
            grayPixelsX.sort((a, b) => a - b);
            grayPixelsY.sort((a, b) => a - b);
            
            const trim = Math.floor(grayPixelsX.length * 0.05); // trim 5% tails
            const pX = grayPixelsX.slice(trim, -trim || undefined);
            const pY = grayPixelsY.slice(trim, -trim || undefined);

            const minX = pX[0], maxX = pX[pX.length - 1];
            const minY = pY[0], maxY = pY[pY.length - 1];

            return {
                cx: (minX + maxX) / 2,
                cy: (minY + maxY) / 2,
                w: maxX - minX,
                h: maxY - minY
            };
        }

        return null;
    }

    /* ═══════════════════════════════════════════════════════════
       Step 4: ML Face Detection (MediaPipe)
       ═══════════════════════════════════════════════════════════ */

    async _detectWithMediaPipe(canvas) {
        try {
            const results = this.faceLandmarker.detect(canvas);
            
            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                const landmarks = results.faceLandmarks[0];
                const width = canvas.width;
                const height = canvas.height;

                // Nose tip (index 4) for exact center
                const cx = landmarks[4].x * width;
                const cy = landmarks[4].y * height;

                // Left (234) and Right (454) cheeks for exact width
                const leftX = landmarks[234].x * width;
                const rightX = landmarks[454].x * width;
                const faceW = rightX - leftX;

                // Chin (152) and Forehead (10) for exact height
                const topY = landmarks[10].y * height;
                const bottomY = landmarks[152].y * height;
                const faceH = bottomY - topY;

                return { cx, cy, w: faceW, h: faceH };
            }
        } catch (e) {
            console.warn('MediaPipe detection failed:', e);
        }
        return null;
    }

    /* ═══════════════════════════════════════════════════════════
       Step 5: Algorithmic Synthesis
       ═══════════════════════════════════════════════════════════ */

    _synthesize(mlFace, existingMask, anatomy, panelW, panelH) {
        let cx, cy, w, h, method, confidence;

        if (mlFace) {
            // Highest confidence: MediaPipe found a real face
            cx = mlFace.cx;
            cy = mlFace.cy;
            w = mlFace.w;
            h = mlFace.h;
            method = 'auto';
            confidence = 0.95;
        } else if (existingMask) {
            // Medium confidence: We found an existing AI mask to replace
            cx = existingMask.cx;
            cy = existingMask.cy;
            w = existingMask.w;
            h = existingMask.h;
            method = 'existing-mask';
            confidence = 0.70;
        } else {
            // Fallback confidence: Mathematical deduction from body bounds
            cx = anatomy.cx;
            cy = anatomy.cy;
            w = anatomy.w;
            h = anatomy.h;
            method = 'heuristic';
            confidence = 0.40;
        }

        // Apply Padding (20% to enclose the face/mask perfectly)
        w *= 1.2;
        h *= 1.2;

        // Aspect Ratio Clamping (Prevent thin slivers)
        w = Math.max(w, h * 0.45);

        // Calculate dynamic feather radius (15% of largest dimension)
        // Scaled as a percentage of the width for the UI slider (which expects 0-100)
        // In our app.js, `feather` is a percentage of the radius.
        // If we want the feather zone to be 15% of the total dimension, that's roughly 30% of the radius.
        const featherPercentage = 30; 

        return {
            cx, cy, width: w, height: h,
            feather: featherPercentage,
            method, confidence
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Fallback / Default
       ═══════════════════════════════════════════════════════════ */

    getHeuristicPosition(panel) {
        return {
            x: panel.x + panel.w * 0.50,
            y: panel.y + panel.h * 0.15,
            width:  panel.w * 0.15,
            height: panel.h * 0.12,
            confidence: 0,
            method: 'heuristic',
        };
    }
}
