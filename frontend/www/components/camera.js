class CameraManager {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('captureCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlays = document.getElementById('overlays');
        this.loading = document.getElementById('loading');
        this.detectionsList = document.getElementById('detectionsList');

        this.isActive = false;
        this.stream = null;
        this.processingInterval = null;
        this.apiEndpoint = 'https://live-waste-classifier.onrender.com/predict';

        this.init();

        document.addEventListener('toggle-camera', (e) => {
            if (e.detail) {
                this.startCamera();
            } else {
                this.stopCamera();
            }
        });
    }

    async init() {
        await this.startCamera();
    }

    async startCamera() {
        try {
            this.loading.style.display = 'flex';
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' }
            });
            this.video.srcObject = this.stream;
            this.isActive = true;

            // Wait for video to be ready before starting inference loop
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.loading.style.display = 'none';
                this.startProcessingLoop();
            };
        } catch (err) {
            console.error('Error accessing camera:', err);
            this.loading.querySelector('span').textContent = 'Camera access denied';
            this.loading.querySelector('.spinner').style.display = 'none';
            document.dispatchEvent(new CustomEvent('api-status', { detail: 'error' }));
        }
    }

    stopCamera() {
        this.isActive = false;
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
        }
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        this.clearOverlays();
        this.updateDetectionsList([]);
    }

    startProcessingLoop() {
        // Send frames to API at approx 2 FPS for realtime capability without overwhelming
        this.processingInterval = setInterval(() => {
            if (this.isActive && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                this.processFrame();
            }
        }, 500);
    }

    async processFrame() {
        // Draw current video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Convert to blob
        this.canvas.toBlob(async (blob) => {
            if (!blob) return;

            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch(this.apiEndpoint, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    document.dispatchEvent(new CustomEvent('api-status', { detail: 'connected' }));
                    this.drawOverlays(data.predictions);
                    this.updateDetectionsList(data.predictions);
                } else {
                    document.dispatchEvent(new CustomEvent('api-status', { detail: 'error' }));
                }
            } catch (err) {
                console.error('API Error:', err);
                document.dispatchEvent(new CustomEvent('api-status', { detail: 'error' }));
            }
        }, 'image/jpeg', 0.8);
    }

    drawOverlays(predictions) {
        this.clearOverlays();

        if (!predictions || predictions.length === 0) return;

        // Calculate scale factors since video display size might differ from video intrinsic size
        const rect = this.video.getBoundingClientRect();
        const scaleX = rect.width / this.video.videoWidth;
        const scaleY = rect.height / this.video.videoHeight;

        predictions.forEach(pred => {
            const [x1, y1, x2, y2] = pred.box;

            // Calculate actual CSS positions taking mirroring into account
            // Since the video is mirrored with scaleX(-1), the X coordinates need to be flipped
            const mirroredX1 = this.video.videoWidth - x2;

            const left = mirroredX1 * scaleX;
            const top = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;

            const boxEl = document.createElement('div');
            boxEl.className = 'bounding-box';
            boxEl.style.left = `${left}px`;
            boxEl.style.top = `${top}px`;
            boxEl.style.width = `${width}px`;
            boxEl.style.height = `${height}px`;

            const labelEl = document.createElement('div');
            labelEl.className = 'label';
            // Use confidence percentage if available, fallback to raw confidence
            const confText = pred.confidence_pct || `${Math.round(pred.confidence * 100)}%`;
            labelEl.textContent = `${pred.class} ${confText}`;

            boxEl.appendChild(labelEl);
            this.overlays.appendChild(boxEl);
        });
    }

    clearOverlays() {
        this.overlays.innerHTML = '';
    }

    updateDetectionsList(predictions) {
        if (!predictions || predictions.length === 0) {
            this.detectionsList.innerHTML = '<li class="empty-state">No objects detected yet</li>';
            return;
        }

        this.detectionsList.innerHTML = '';
        predictions.forEach(pred => {
            const confText = pred.confidence_pct || `${Math.round(pred.confidence * 100)}%`;
            const html = `
                <li class="detection-item">
                    <span class="detection-name">${pred.class}</span>
                    <span class="detection-conf">${confText}</span>
                </li>
            `;
            this.detectionsList.insertAdjacentHTML('beforeend', html);
        });
    }
}

// Initialize camera manager on load
window.addEventListener('load', () => {
    new CameraManager();
});
