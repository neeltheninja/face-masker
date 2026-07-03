/**
 * Face Detection Module for Character Datasheet Face Masker
 *
 * Detection strategies (tried in order):
 *   1. face-api.js auto-detection (for unmasked faces)
 *   2. Existing mask detection (finds gray blobs inside the figure — for already-masked faces)
 *   3. Smart figure-boundary heuristic (pixel scanning)
 *   4. Basic fixed-percentage heuristic (last resort)
 */

class FaceDetectorModule {
    constructor() {
        this.isLoaded = false;
        this.isLoading = false;
        this.loadError = null;
        this._onStatus = null;
    }

    /* ───────────────────── Initialization ───────────────────── */

    async init(onStatus) {
        if (this.isLoaded) return true;
        if (this.isLoading) return false;

        this._onStatus = onStatus || (() => {});
        this.isLoading = true;
        this._onStatus('loading', 'Loading face detection…');

        const libSources = [
            'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js',
            'https://unpkg.com/@vladmandic/face-api@1.7.12/dist/face-api.js',
            'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js',
        ];

        let libLoaded = false;
        for (const src of libSources) {
            try {
                await this._loadScript(src);
                libLoaded = true;
                console.log('face-api.js loaded from:', src);
                break;
            } catch (e) {
                console.warn('Failed to load from:', src, e.message);
            }
        }

        if (!libLoaded || typeof faceapi === 'undefined') {
            this.isLoading = false;
            this._onStatus('fallback', 'Smart heuristic active (face-api unavailable)');
            return false;
        }

        this._onStatus('loading', 'Loading detection model…');
        const modelSources = [
            'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/',
            'https://unpkg.com/@vladmandic/face-api@1.7.12/model/',
            'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights/',
        ];

        let modelLoaded = false;
        for (const modelPath of modelSources) {
            try {
                await faceapi.nets.tinyFaceDetector.loadFromUri(modelPath);
                modelLoaded = true;
                console.log('Models loaded from:', modelPath);
                break;
            } catch (e) {
                console.warn('Failed to load models from:', modelPath, e.message);
            }
        }

        if (!modelLoaded) {
            this.isLoading = false;
            this._onStatus('fallback', 'Smart heuristic active (models unavailable)');
            return false;
        }

        this.isLoaded = true;
        this.isLoading = false;
        this._onStatus('ready', 'Face detection ready');
        return true;
    }

    _loadScript(src) {
        return new Promise((resolve, reject) => {
            if (typeof faceapi !== 'undefined') { resolve(); return; }
            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            script.crossOrigin = 'anonymous';
            const timeout = setTimeout(() => reject(new Error('Timeout: ' + src)), 10000);
            script.onload = () => { clearTimeout(timeout); resolve(); };
            script.onerror = () => { clearTimeout(timeout); reject(new Error('Failed: ' + src)); };
            document.head.appendChild(script);
        });
    }

    /* ───────────────────── Main Detection ───────────────────── */

    async detect(source, panel) {
        // Strategy 1: face-api.js (works for unmasked faces)
        if (this.isLoaded) {
            try {
                const result = await this._detectWithFaceApi(source, panel);
                if (result) return result;
            } catch (err) {
                console.warn('face-api detection failed:', err);
            }
        }

        // Strategy 2 + 3: Smart scan (finds existing masks OR figure boundary)
        try {
            const result = this._smartScan(source, panel);
            if (result) return result;
        } catch (err) {
            console.warn('Smart scan failed:', err);
        }

        // Strategy 4: Basic heuristic
        return this.getHeuristicPosition(panel);
    }

    async _detectWithFaceApi(source, panel) {
        const temp = document.createElement('canvas');
        temp.width = panel.w;
        temp.height = panel.h;
        const ctx = temp.getContext('2d');
        ctx.drawImage(source, panel.x, panel.y, panel.w, panel.h, 0, 0, panel.w, panel.h);

        const detections = await faceapi.detectAllFaces(
            temp, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.25 })
        );
        if (!detections || detections.length === 0) return null;

        const best = detections.reduce((a, b) => a.box.area > b.box.area ? a : b);
        const box = best.box;

        return {
            x: panel.x + box.x + box.width / 2,
            y: panel.y + box.y + box.height / 2,
            width:  box.width  * 1.5,
            height: box.height * 1.6,
            confidence: best.score,
            method: 'auto',
        };
    }

    /* ═══════════════════════════════════════════════════════
       Smart Scan — handles BOTH masked and unmasked faces
       ═══════════════════════════════════════════════════════ */

    _smartScan(source, panel) {
        // Downsample for speed
        const temp = document.createElement('canvas');
        const scanScale = Math.min(1, 500 / panel.w);
        const scanW = Math.round(panel.w * scanScale);
        const scanH = Math.round(panel.h * scanScale);
        temp.width = scanW;
        temp.height = scanH;
        const ctx = temp.getContext('2d');
        ctx.drawImage(source, panel.x, panel.y, panel.w, panel.h, 0, 0, scanW, scanH);

        const imgData = ctx.getImageData(0, 0, scanW, scanH);
        const data = imgData.data;

        // ─── Sample background color from corners ───
        const bg = this._sampleBackground(data, scanW, scanH);
        const threshold = 35;

        // ─── Find figure boundaries ───
        const figure = this._findFigureBounds(data, scanW, scanH, bg, threshold);
        if (!figure) return null;

        // ─── Try to detect an EXISTING gray mask within the figure ───
        const existingMask = this._detectExistingMask(
            data, scanW, scanH, bg, threshold, figure
        );

        if (existingMask) {
            // Found an existing mask — use its position
            return {
                x: panel.x + existingMask.cx / scanScale,
                y: panel.y + existingMask.cy / scanScale,
                width:  existingMask.w / scanScale * 1.2,
                height: existingMask.h / scanScale * 1.2,
                confidence: 0.5,
                method: 'heuristic',
            };
        }

        // ─── No existing mask — estimate from figure boundary ───
        return this._estimateFromFigure(panel, figure, data, scanW, bg, threshold, scanScale);
    }

    /**
     * Sample the average background color from the image corners.
     */
    _sampleBackground(data, w, h) {
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        const corners = [
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [1, 0], [0, 1], [w-2, 0], [w-1, 1],
            [1, 1], [w-2, 1], [1, h-2], [w-2, h-2],
        ];
        for (const [x, y] of corners) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                const i = (y * w + x) * 4;
                rSum += data[i]; gSum += data[i+1]; bSum += data[i+2];
                count++;
            }
        }
        return {
            r: Math.round(rSum / count),
            g: Math.round(gSum / count),
            b: Math.round(bSum / count),
        };
    }

    /**
     * Find the top and bottom of the character figure by scanning for
     * rows with significant non-background content.
     */
    _findFigureBounds(data, scanW, scanH, bg, threshold) {
        const xStart = Math.floor(scanW * 0.15);
        const xEnd   = Math.floor(scanW * 0.85);
        const xRange = xEnd - xStart;
        const rowThresh = xRange * 0.06; // 6% of scan width must be non-bg

        let figureTop = -1;
        for (let y = 0; y < scanH; y++) {
            let count = 0;
            for (let x = xStart; x < xEnd; x++) {
                const i = (y * scanW + x) * 4;
                if (Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b) > threshold) {
                    count++;
                }
            }
            if (count >= rowThresh) { figureTop = y; break; }
        }

        if (figureTop < 0) return null;

        let figureBottom = scanH;
        for (let y = scanH - 1; y > figureTop; y--) {
            let count = 0;
            for (let x = xStart; x < xEnd; x++) {
                const i = (y * scanW + x) * 4;
                if (Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b) > threshold) {
                    count++;
                }
            }
            if (count >= rowThresh) { figureBottom = y; break; }
        }

        const height = figureBottom - figureTop;
        if (height < scanH * 0.15) return null;

        return { top: figureTop, bottom: figureBottom, height };
    }

    /**
     * Detect an existing gray mask within the upper portion of the figure.
     *
     * An existing mask shows up as a horizontal "gray gap" INSIDE the figure —
     * gray pixels flanked by non-gray content (hair on sides, body below).
     * We scan each row in the face zone for such gaps.
     */
    _detectExistingMask(data, scanW, scanH, bg, threshold, figure) {
        // Scan the upper 18% of the figure (the face zone)
        const zoneStart = figure.top;
        const zoneEnd = Math.min(
            Math.floor(figure.top + figure.height * 0.18),
            scanH
        );

        const gapRows = []; // rows where we found a gray gap inside the figure

        for (let y = zoneStart; y < zoneEnd; y++) {
            // Step 1: Find the figure's left and right edges at this row
            let leftEdge = -1, rightEdge = -1;
            for (let x = 0; x < scanW; x++) {
                const i = (y * scanW + x) * 4;
                const diff = Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b);
                if (diff > threshold) {
                    if (leftEdge === -1) leftEdge = x;
                    rightEdge = x;
                }
            }

            if (leftEdge === -1 || rightEdge - leftEdge < scanW * 0.05) continue;

            // Step 2: Within figure bounds, find the longest gray run
            let bestStart = -1, bestEnd = -1, bestLen = 0;
            let runStart = -1, runLen = 0;

            for (let x = leftEdge + 1; x < rightEdge; x++) {
                const i = (y * scanW + x) * 4;
                const diff = Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b);

                if (diff <= threshold) {
                    // Gray pixel inside figure
                    if (runStart === -1) runStart = x;
                    runLen++;
                } else {
                    if (runLen > bestLen) {
                        bestLen = runLen;
                        bestStart = runStart;
                        bestEnd = x - 1;
                    }
                    runStart = -1;
                    runLen = 0;
                }
            }
            if (runLen > bestLen) {
                bestLen = runLen;
                bestStart = runStart;
                bestEnd = rightEdge - 1;
            }

            // A significant gray run (> 5% of panel width) inside the figure = existing mask
            if (bestLen > scanW * 0.05 && bestStart > -1) {
                gapRows.push({
                    y,
                    gapStart: bestStart,
                    gapEnd: bestEnd,
                    gapWidth: bestLen,
                });
            }
        }

        // Need at least several rows with consistent gray gaps to confirm a mask
        if (gapRows.length < 5) return null;

        // Calculate the mask center and size from the gap rows
        const midIdx = Math.floor(gapRows.length / 2);
        const cy = (gapRows[0].y + gapRows[gapRows.length - 1].y) / 2;
        const cx = gapRows.reduce((s, r) => s + (r.gapStart + r.gapEnd) / 2, 0) / gapRows.length;
        const maxGapW = Math.max(...gapRows.map(r => r.gapWidth));
        const gapH = gapRows[gapRows.length - 1].y - gapRows[0].y;

        // Sanity check: mask should be roughly face-shaped (not a thin strip)
        if (gapH < 3 || maxGapW < 3) return null;

        console.log(`Existing mask detected: center=(${cx.toFixed(0)}, ${cy.toFixed(0)}), size=${maxGapW}×${gapH}, rows=${gapRows.length}`);

        return {
            cx,
            cy,
            w: maxGapW * 1.1,
            h: gapH * 1.1,
        };
    }

    /**
     * Estimate face position from the figure boundary (for unmasked images).
     */
    _estimateFromFigure(panel, figure, data, scanW, bg, threshold, scanScale) {
        // Face center Y: approximately 5.5% of figure height below figure top
        const faceCenterY = figure.top + figure.height * 0.06;

        // For horizontal center: scan near the top of the figure (hair area)
        // which is reliable even if the face is masked
        const scanRow = Math.min(Math.round(figure.top + figure.height * 0.03), scanW - 1);
        let leftEdge = 0, rightEdge = scanW;

        if (scanRow >= 0 && scanRow < Math.round(panel.h * scanScale)) {
            for (let x = 0; x < scanW; x++) {
                const i = (scanRow * scanW + x) * 4;
                if (Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b) > threshold) {
                    leftEdge = x;
                    break;
                }
            }
            for (let x = scanW - 1; x >= 0; x--) {
                const i = (scanRow * scanW + x) * 4;
                if (Math.abs(data[i]-bg.r) + Math.abs(data[i+1]-bg.g) + Math.abs(data[i+2]-bg.b) > threshold) {
                    rightEdge = x;
                    break;
                }
            }
        }

        const faceCenterX = (leftEdge + rightEdge) / 2;
        const faceW = figure.height * 0.10;
        const faceH = figure.height * 0.09;

        return {
            x: panel.x + faceCenterX / scanScale,
            y: panel.y + faceCenterY / scanScale,
            width:  faceW / scanScale,
            height: faceH / scanScale,
            confidence: 0.3,
            method: 'heuristic',
        };
    }

    /* ───────────────────── Basic Heuristic ───────────────────── */

    getHeuristicPosition(panel) {
        return {
            x: panel.x + panel.w * 0.50,
            y: panel.y + panel.h * 0.20,
            width:  panel.w * 0.22,
            height: panel.h * 0.14,
            confidence: 0,
            method: 'heuristic',
        };
    }
}

/**
 * @typedef {Object} FaceDetectionResult
 * @property {number} x       – Center X in full-image coordinates
 * @property {number} y       – Center Y in full-image coordinates
 * @property {number} width   – Full width of the detected face region
 * @property {number} height  – Full height of the detected face region
 * @property {number} confidence – 0–1 (0 = heuristic)
 * @property {'auto'|'heuristic'} method
 */
