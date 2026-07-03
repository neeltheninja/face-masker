/**
 * Face Detection Module — Robust Pipeline for Character Datasheets
 * 
 * Strategy:
 *   The datasheets always have 3 panels: [Full-body Front | Back | Close-up Face]
 *   
 *   Path A — "Clean Face" (no existing mask on left panel):
 *     1. Try MediaPipe on the LEFT panel directly to get precise landmarks.
 *     2. Fallback: Try MediaPipe on the RIGHT panel (close-up face) to learn the
 *        face proportions relative to the body, then map back to the left panel
 *        using figure isolation.
 *     3. Final fallback: Use figure isolation + anatomical proportions.
 *
 *   Path B — "Pre-masked Face" (existing gray/blurred mask detected):
 *     1. Detect the existing mask blob on the left panel.
 *     2. Use the blob's center and size (expanded slightly for coverage).
 *     3. Fallback: Use figure isolation + anatomical proportions.
 *
 *   The existing-mask detector compares pixels against the SAMPLED background
 *   color to avoid false positives when the background itself is gray.
 */

class FaceDetectorModule {
    constructor() {
        this.isLoaded = false;
        this.isLoading = false;
        this.faceLandmarker = null;
        this._onStatus = null;
    }

    /* ═══════════════════════════════════════════════════════════
       Initialization (MediaPipe)
       ═══════════════════════════════════════════════════════════ */

    async init(onStatus) {
        if (this.isLoaded) return true;
        if (this.isLoading) return false;

        this._onStatus = onStatus || (() => {});
        this.isLoading = true;
        this._onStatus('loading', 'Loading MediaPipe Vision…');

        try {
            const vision = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs'
            );

            this._onStatus('loading', 'Initializing Face Landmarker…');

            const fileset = await vision.FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm'
            );

            this.faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
                baseOptions: {
                    modelAssetPath:
                        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    delegate: 'CPU',
                },
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false,
                runningMode: 'IMAGE',
                numFaces: 1,
            });

            this.isLoaded = true;
            this.isLoading = false;
            this._onStatus('ready', 'Face Detection ready');
            return true;
        } catch (e) {
            console.warn('MediaPipe load failed:', e);
            this.isLoading = false;
            this._onStatus('fallback', 'ML unavailable — heuristic mode');
            return false;
        }
    }

    /* ═══════════════════════════════════════════════════════════
       Main Detection Entry Point
       ═══════════════════════════════════════════════════════════ */

    /**
     * @param {HTMLImageElement} source - The full datasheet image
     * @param {Object} panel - { x, y, w, h } of the left panel in image coords
     * @returns {{ x, y, width, height, feather, softness, confidence, method }}
     *          x,y = center of mask in full-image coords
     */
    async detect(source, panel) {
        console.log('[detect] Panel:', panel, 'Image:', source.naturalWidth, 'x', source.naturalHeight);

        // Extract the left panel into its own canvas
        const panelCanvas = this._cropToCanvas(source, panel.x, panel.y, panel.w, panel.h);

        // Step 1: Isolate the figure from the background
        const figure = this._isolateFigure(panelCanvas);
        console.log('[detect] Figure:', figure);

        // Step 2: Check for an existing mask (gray blob that differs from background)
        const existingMask = figure ? this._detectExistingMask(panelCanvas, figure) : null;
        console.log('[detect] Existing mask:', existingMask);

        let result;

        if (existingMask) {
            // ──── Path B: Pre-masked face ────
            // Use the existing mask's position (it tells us where the face WAS)
            result = {
                cx: existingMask.cx,
                cy: existingMask.cy,
                w: existingMask.w * 1.15,  // slight expansion for full coverage
                h: existingMask.h * 1.15,
                method: 'existing-mask',
                confidence: 0.75,
            };
            console.log('[detect] Using existing mask position');
        } else {
            // ──── Path A: No mask — try ML detection ────
            let mlFace = null;

            if (this.isLoaded && this.faceLandmarker) {
                // First try: detect face directly on the left panel
                mlFace = this._runMediaPipe(panelCanvas);
                console.log('[detect] MediaPipe on left panel:', mlFace);

                // Second try: detect on the RIGHT panel (close-up) and map proportions
                if (!mlFace && figure) {
                    const rightPanel = this._cropToCanvas(
                        source,
                        Math.floor(source.naturalWidth * 2 / 3), 0,
                        Math.floor(source.naturalWidth / 3), source.naturalHeight
                    );
                    const rightFace = this._runMediaPipe(rightPanel);
                    console.log('[detect] MediaPipe on right panel:', rightFace);

                    if (rightFace) {
                        // The right panel face gives us proportional info
                        // Map it onto the left panel using figure bounds
                        mlFace = this._mapRightFaceToLeft(rightFace, rightPanel, figure, panelCanvas);
                        console.log('[detect] Mapped right→left:', mlFace);
                    }
                }
            }

            if (mlFace) {
                result = {
                    cx: mlFace.cx,
                    cy: mlFace.cy,
                    w: mlFace.w * 1.2,
                    h: mlFace.h * 1.2,
                    method: 'auto',
                    confidence: 0.92,
                };
            } else {
                // Final fallback: anatomical heuristics from figure bounds
                const anatomy = this._deduceAnatomy(figure, panel.w, panel.h);
                console.log('[detect] Anatomy fallback:', anatomy);
                result = {
                    cx: anatomy.cx,
                    cy: anatomy.cy,
                    w: anatomy.w,
                    h: anatomy.h,
                    method: 'heuristic',
                    confidence: 0.45,
                };
            }
        }

        // Ensure minimum dimensions
        result.w = Math.max(result.w, 40);
        result.h = Math.max(result.h, 50);

        // Translate from panel-local coords back to full-image coords
        return {
            x: panel.x + result.cx,
            y: panel.y + result.cy,
            width: result.w,
            height: result.h,
            feather: 35,
            softness: 30,
            confidence: result.confidence,
            method: result.method,
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Canvas Helpers
       ═══════════════════════════════════════════════════════════ */

    _cropToCanvas(source, sx, sy, sw, sh) {
        const c = document.createElement('canvas');
        c.width = sw;
        c.height = sh;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
        return c;
    }

    /* ═══════════════════════════════════════════════════════════
       Step 1: Figure Isolation
       
       Scans the panel to find non-background pixels.
       Returns bounding box + head center of mass.
       Background is sampled from the corners (average of 4 corners)
       to handle edge cases where a single corner might be occluded.
       ═══════════════════════════════════════════════════════════ */

    _isolateFigure(canvas) {
        const w = canvas.width;
        const h = canvas.height;
        const ctx = canvas.getContext('2d');
        const data = ctx.getImageData(0, 0, w, h).data;

        // Sample background from all 4 corners (5×5 average each)
        const bg = this._sampleBackground(data, w, h);

        // Use per-channel threshold — more robust for gray backgrounds
        const threshold = 30;
        const isForeground = (idx) => {
            const dr = Math.abs(data[idx] - bg.r);
            const dg = Math.abs(data[idx + 1] - bg.g);
            const db = Math.abs(data[idx + 2] - bg.b);
            return (dr + dg + db) > threshold;
        };

        let minX = w, maxX = -1, minY = h, maxY = -1;
        let totalX = 0, totalCount = 0;

        // Single pass: find bounding box + center of mass
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                if (isForeground(idx)) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    totalX += x;
                    totalCount++;
                }
            }
        }

        if (totalCount < 100) return null; // No significant figure found

        const figH = maxY - minY;
        const figW = maxX - minX;

        // Second pass: center of mass for the HEAD region (top 18% of figure)
        const headBottomY = minY + Math.floor(figH * 0.18);
        let headSumX = 0, headSumY = 0, headCount = 0;

        for (let y = minY; y <= headBottomY; y++) {
            for (let x = minX; x <= maxX; x++) {
                const idx = (y * w + x) * 4;
                if (isForeground(idx)) {
                    headSumX += x;
                    headSumY += y;
                    headCount++;
                }
            }
        }

        return {
            top: minY,
            bottom: maxY,
            left: minX,
            right: maxX,
            figW,
            figH,
            bodyCenterX: totalX / totalCount,
            headCenterX: headCount > 0 ? headSumX / headCount : totalX / totalCount,
            headCenterY: headCount > 0 ? headSumY / headCount : minY + figH * 0.08,
            bg,
        };
    }

    /**
     * Sample background color by averaging the 4 corners of the image (5×5 each).
     * This is more robust than a single pixel.
     */
    _sampleBackground(data, w, h) {
        let r = 0, g = 0, b = 0, count = 0;
        const sampleSize = 5;

        // Corners: top-left, top-right, bottom-left, bottom-right
        const corners = [
            [0, 0], [w - sampleSize, 0],
            [0, h - sampleSize], [w - sampleSize, h - sampleSize],
        ];

        for (const [cx, cy] of corners) {
            for (let dy = 0; dy < sampleSize; dy++) {
                for (let dx = 0; dx < sampleSize; dx++) {
                    const x = cx + dx;
                    const y = cy + dy;
                    if (x >= 0 && x < w && y >= 0 && y < h) {
                        const idx = (y * w + x) * 4;
                        r += data[idx];
                        g += data[idx + 1];
                        b += data[idx + 2];
                        count++;
                    }
                }
            }
        }

        return {
            r: Math.round(r / count),
            g: Math.round(g / count),
            b: Math.round(b / count),
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 2: Existing Mask Detection
       
       Looks for a gray blob that is DIFFERENT from the background.
       Key insight: The background is often gray too, so we must
       check that pixels differ from the background AND are grayish
       AND are located within the expected head region.
       ═══════════════════════════════════════════════════════════ */

    _detectExistingMask(canvas, figure) {
        const w = canvas.width;
        const h = canvas.height;
        const ctx = canvas.getContext('2d');
        const data = ctx.getImageData(0, 0, w, h).data;

        const bg = figure.bg;

        // Scan only the head region of the figure (top 25% of body)
        // with some horizontal padding
        const headRegionTop = Math.max(0, figure.top - 10);
        const headRegionBottom = Math.min(h - 1, figure.top + Math.floor(figure.figH * 0.25));
        const headRegionLeft = Math.max(0, Math.floor(figure.headCenterX - figure.figW * 0.2));
        const headRegionRight = Math.min(w - 1, Math.floor(figure.headCenterX + figure.figW * 0.2));

        let maskPixelsX = [];
        let maskPixelsY = [];

        for (let y = headRegionTop; y <= headRegionBottom; y++) {
            for (let x = headRegionLeft; x <= headRegionRight; x++) {
                const idx = (y * w + x) * 4;
                const r = data[idx], g = data[idx + 1], b = data[idx + 2];

                // Is this pixel significantly different from the background?
                const diffFromBg = Math.abs(r - bg.r) + Math.abs(g - bg.g) + Math.abs(b - bg.b);
                if (diffFromBg < 25) continue; // Same as background — skip

                // Is this pixel grayish? (low saturation)
                const maxC = Math.max(r, g, b);
                const minC = Math.min(r, g, b);
                const sat = maxC === 0 ? 0 : (maxC - minC) / maxC;

                // Is it in the "mask gray" range — not too dark, not too light
                // and low saturation (desaturated = gray)
                if (sat < 0.20 && maxC > 80 && maxC < 220 && minC > 60) {
                    maskPixelsX.push(x);
                    maskPixelsY.push(y);
                }
            }
        }

        // Need a significant cluster to be considered a mask
        // At least 200 pixels for a reasonable blob
        if (maskPixelsX.length < 200) return null;

        // Calculate bounds with outlier trimming (5% each end)
        maskPixelsX.sort((a, b) => a - b);
        maskPixelsY.sort((a, b) => a - b);

        const trim = Math.max(1, Math.floor(maskPixelsX.length * 0.05));
        const trimmedX = maskPixelsX.slice(trim, -trim);
        const trimmedY = maskPixelsY.slice(trim, -trim);

        if (trimmedX.length === 0) return null;

        const blobMinX = trimmedX[0];
        const blobMaxX = trimmedX[trimmedX.length - 1];
        const blobMinY = trimmedY[0];
        const blobMaxY = trimmedY[trimmedY.length - 1];
        const blobW = blobMaxX - blobMinX;
        const blobH = blobMaxY - blobMinY;

        // Sanity check: the blob should be roughly face-shaped (not a thin line)
        const aspect = blobW / Math.max(blobH, 1);
        if (aspect < 0.3 || aspect > 3.0) return null;
        if (blobW < 20 || blobH < 20) return null;

        console.log(`[mask-detect] Found mask blob: center=(${(blobMinX + blobMaxX) / 2}, ${(blobMinY + blobMaxY) / 2}), size=${blobW}×${blobH}, pixels=${maskPixelsX.length}`);

        return {
            cx: (blobMinX + blobMaxX) / 2,
            cy: (blobMinY + blobMaxY) / 2,
            w: blobW,
            h: blobH,
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 3: MediaPipe Face Landmark Detection
       ═══════════════════════════════════════════════════════════ */

    _runMediaPipe(canvas) {
        if (!this.faceLandmarker) return null;

        try {
            const results = this.faceLandmarker.detect(canvas);

            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                const lm = results.faceLandmarks[0];
                const cw = canvas.width;
                const ch = canvas.height;

                // Key landmarks for face bounds:
                // 10 = forehead top, 152 = chin bottom
                // 234 = left cheek, 454 = right cheek
                // 4 = nose tip (center anchor)
                const noseTip = lm[4];
                const forehead = lm[10];
                const chin = lm[152];
                const leftCheek = lm[234];
                const rightCheek = lm[454];

                const cx = noseTip.x * cw;
                const cy = noseTip.y * ch;
                const faceW = Math.abs(rightCheek.x - leftCheek.x) * cw;
                const faceH = Math.abs(chin.y - forehead.y) * ch;

                // Sanity check
                if (faceW < 10 || faceH < 10) return null;

                return { cx, cy, w: faceW, h: faceH };
            }
        } catch (e) {
            console.warn('MediaPipe detection error:', e);
        }
        return null;
    }

    /**
     * Maps face proportions detected on the RIGHT panel (close-up)
     * onto the LEFT panel using figure isolation data.
     *
     * The right panel shows a cropped bust/face shot. We use the
     * ratio of face-to-visible-body to estimate where the face
     * sits on the full-body left panel.
     */
    _mapRightFaceToLeft(rightFace, rightCanvas, figure, leftCanvas) {
        // On the right panel, the face occupies a large portion.
        // The face height relative to figure height gives us the proportion.
        // On the left panel, apply that proportion to the full figure height.
        
        // We know the head center on the left panel from figure isolation
        const faceFractionOfFigure = rightFace.h / rightCanvas.height;
        
        // Scale to left panel figure dimensions
        const leftFaceH = figure.figH * faceFractionOfFigure * 0.85; // close-ups are cropped tighter
        const leftFaceW = leftFaceH * (rightFace.w / rightFace.h); // preserve aspect ratio

        return {
            cx: figure.headCenterX,
            cy: figure.headCenterY,
            w: leftFaceW,
            h: leftFaceH,
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Step 4: Anatomical Proportion Fallback
       
       When neither ML nor existing-mask detection works,
       deduce head position from the figure's bounding box.
       
       Real character datasheets: figures fill ~85% of panel height.
       Head is roughly:
         - 1/7 to 1/8 of total figure height
         - Centered at ~8% below the top of the figure
         - Width ≈ 60-75% of head height
       ═══════════════════════════════════════════════════════════ */

    _deduceAnatomy(figure, panelW, panelH) {
        if (!figure) {
            // Complete fallback — no figure found at all
            return {
                cx: panelW * 0.5,
                cy: panelH * 0.12,
                w: panelW * 0.14,
                h: panelH * 0.10,
            };
        }

        const H = figure.figH;

        // Head height: ~1/7.5 of figure (slightly more generous than strict 1/8)
        const headH = H / 7.5;
        // Head width: ~70% of head height (oval shape)
        const headW = headH * 0.72;
        // Head center Y: top of figure + half a head height
        const headCY = figure.top + headH * 0.55;
        // Head center X: from the head center of mass
        const headCX = figure.headCenterX;

        return { cx: headCX, cy: headCY, w: headW, h: headH };
    }

    /* ═══════════════════════════════════════════════════════════
       Default Heuristic (used by App.getDefaultMask for initial load)
       ═══════════════════════════════════════════════════════════ */

    getHeuristicPosition(panel) {
        return {
            x: panel.x + panel.w * 0.50,
            y: panel.y + panel.h * 0.14,
            width: panel.w * 0.18,
            height: panel.h * 0.10,
            confidence: 0,
            method: 'heuristic',
        };
    }
}
