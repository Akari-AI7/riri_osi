class FaceAnalysisApp {
    constructor() {
        this.baseImage = null;
        this.compareImage = null;
        this.stream = null;
        this.faceMesh = null;
        this.onResultsBound = this.onResults.bind(this);
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // 基準画像関連
        document.getElementById('baseUploadArea').addEventListener('click', () => {
            document.getElementById('baseImageInput').click();
        });
        document.getElementById('baseImageInput').addEventListener('change', (e) => {
            this.handleBaseImageSelect(e);
        });
        document.getElementById('setBaseBtn').addEventListener('click', () => {
            this.uploadBaseImage();
        });

        // 比較画像関連
        document.getElementById('compareUploadArea').addEventListener('click', () => {
            document.getElementById('compareImageInput').click();
        });
        document.getElementById('compareImageInput').addEventListener('change', (e) => {
            this.handleCompareImageSelect(e);
        });
        document.getElementById('compareBtn').addEventListener('click', () => {
            this.compareImages();
        });

        // Webカメラ関連
        document.getElementById('startCameraBtn').addEventListener('click', () => {
            this.startCamera();
        });
        document.getElementById('stopCameraBtn').addEventListener('click', () => {
            this.stopCamera();
        });
        document.getElementById('captureBtn').addEventListener('click', () => {
            this.captureFromCamera();
        });

        // ドラッグ&ドロップ
        this.setupDragAndDrop();
    }

    setupDragAndDrop() {
        const areas = ['baseUploadArea', 'compareUploadArea'];
        areas.forEach(areaId => {
            const area = document.getElementById(areaId);
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });
            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });
            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const input = document.getElementById(areaId === 'baseUploadArea' ? 'baseImageInput' : 'compareImageInput');
                    input.files = files;
                    const event = new Event('change', { bubbles: true });
                    input.dispatchEvent(event);
                }
            });
        });
    }

    handleBaseImageSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.baseImage = file;
            this.previewImage(file, 'baseImagePreview');
        }
    }

    handleCompareImageSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.compareImage = file;
            this.previewImage(file, 'compareImagePreview');
        }
    }

    previewImage(file, containerId) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const container = document.getElementById(containerId);
            container.innerHTML = `<img src="${e.target.result}" class="image-preview" alt="プレビュー">`;
        };
        reader.readAsDataURL(file);
    }

    async uploadBaseImage() {
        if (!this.baseImage) return;

        const formData = new FormData();
        formData.append('image', this.baseImage);

        try {
            this.showLoading();
            const response = await fetch('/upload_base', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideLoading();

            if (result.success) {
                this.showSuccess(result.message);
                // ランドマーク画像を表示
                document.getElementById('baseImagePreview').innerHTML = 
                    `<img src="${result.landmark_image}" class="image-preview" alt="基準画像（ランドマーク付き）">`;
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.hideLoading();
            this.showError('通信エラーが発生しました');
        }
    }

    async compareImages() {
        if (!this.compareImage) return;

        const formData = new FormData();
        formData.append('image', this.compareImage);

        try {
            this.showLoading();
            const response = await fetch('/compare', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideLoading();

            if (result.success) {
                this.displayResults(result);
                this.showSuccess('分析が完了しました');
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.hideLoading();
            this.showError('通信エラーが発生しました');
        }
    }

    displayResults(result) {
        const resultsDiv = document.getElementById('results');
        const analysisGrid = document.getElementById('analysisGrid');
        const descriptionsDiv = document.getElementById('descriptions');

        // 数値データ表示
        analysisGrid.innerHTML = '';
        for (const [feature, data] of Object.entries(result.differences)) {
            const changeClass = data.change_percent > 0 ? 'change-positive' : 'change-negative';
            const featureDiv = document.createElement('div');
            featureDiv.className = 'feature-item';
            featureDiv.innerHTML = `
                <div class="feature-name">${feature}</div>
                <div class="feature-change ${changeClass}">
                    ${data.change_percent > 0 ? '+' : ''}${data.change_percent.toFixed(1)}%
                    (${data.pixel_change > 0 ? '+' : ''}${data.pixel_change.toFixed(1)}px)
                </div>
            `;
            analysisGrid.appendChild(featureDiv);
        }

        // AI分析結果表示
        descriptionsDiv.innerHTML = `
            <h3>AI分析による詳細</h3>
            <ul>
                ${result.descriptions.map(desc => `<li>${desc}</li>`).join('')}
            </ul>
        `;

        resultsDiv.classList.remove('hidden');
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const video = document.getElementById('video');
            video.srcObject = this.stream;
            video.style.display = 'block';
            await this.initFaceMesh();
            this.resizeCanvasToVideo();
            this.startProcessingLoop();
            
        } catch (error) {
            this.showError('カメラへのアクセスが許可されていません');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            document.getElementById('video').style.display = 'none';
            const ctx = document.getElementById('landmarksCanvas').getContext('2d');
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            
        }
    }

    captureFromCamera() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg');
        
        // 撮影した画像を比較画像として設定
        document.getElementById('compareImagePreview').innerHTML = 
            `<img src="${imageData}" class="image-preview" alt="撮影した画像">`;
        
        // File オブジェクトを作成
        canvas.toBlob((blob) => {
            this.compareImage = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
        }, 'image/jpeg');
    }

    async initFaceMesh() {
        // MediaPipe FaceMesh (browser) セットアップ
        if (!window.FaceMesh) return;
        this.faceMesh = new FaceMesh.FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        this.faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        this.faceMesh.onResults(this.onResultsBound);
    }

    resizeCanvasToVideo() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('landmarksCanvas');
        const update = () => {
            const rect = video.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        };
        update();
        window.addEventListener('resize', update);
    }

    async startProcessingLoop() {
        if (!this.faceMesh) return;
        const video = document.getElementById('video');
        const onFrame = async () => {
            if (video.readyState >= 2 && this.stream) {
                await this.faceMesh.send({ image: video });
            }
            if (this.stream) requestAnimationFrame(onFrame);
        };
        requestAnimationFrame(onFrame);
    }

    onResults(results) {
        const canvas = document.getElementById('landmarksCanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

        const video = document.getElementById('video');
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = canvas.width;
        const ch = canvas.height;

        const scaleX = cw / vw;
        const scaleY = ch / vh;

        ctx.strokeStyle = '#00ffff';
        ctx.fillStyle = '#00ffff';
        ctx.lineWidth = 1.5;

        const points = results.multiFaceLandmarks[0];
        for (let i = 0; i < points.length; i++) {
            const x = points[i].x * vw * scaleX;
            const y = points[i].y * vh * scaleY;
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }

    showSuccess(message) {
        const successDiv = document.getElementById('successMessage');
        successDiv.textContent = message;
        successDiv.style.display = 'block';
        setTimeout(() => {
            successDiv.style.display = 'none';
        }, 3000);
    }
}

// アプリケーション初期化
document.addEventListener('DOMContentLoaded', () => {
    new FaceAnalysisApp();
});