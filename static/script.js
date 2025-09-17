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
            await video.play();
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
        console.log('Checking FaceMesh namespace...');
        const FaceMeshNS = window.FaceMesh || window.faceMesh;
        console.log('FaceMeshNS:', FaceMeshNS);
        if (!FaceMeshNS) {
            console.warn('FaceMesh namespace not found. Check CDN script loading.');
            return;
        }
        console.log('Initializing FaceMesh...');
        try {
            console.log('Creating FaceMesh instance...');
            this.faceMesh = new FaceMeshNS.FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
            });
            console.log('Setting FaceMesh options...');
            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            console.log('Setting FaceMesh onResults callback...');
            this.faceMesh.onResults(this.onResultsBound);
            console.log('FaceMesh initialized');
        } catch (error) {
            console.error('FaceMesh initialization failed:', error);
        }
    }

    resizeCanvasToVideo() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('landmarksCanvas');
        const update = () => {
            const rect = video.getBoundingClientRect();
            if (rect.width && rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
            }
        };
        if (video.readyState >= 1) {
            update();
        } else {
            video.addEventListener('loadedmetadata', update, { once: true });
        }
        window.addEventListener('resize', update);
    }

    async startProcessingLoop() {
        if (!this.faceMesh) return;
        console.log('Starting processing loop...');
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
        console.log('FaceMesh results:', results);
        const canvas = document.getElementById('landmarksCanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
        console.log('Drawing landmarks...');

        const video = document.getElementById('video');
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = canvas.width;
        const ch = canvas.height;

        const scaleX = cw / vw;
        const scaleY = ch / vh;

        const points = results.multiFaceLandmarks[0];

        const drawPoints = (indices, color) => {
            ctx.fillStyle = color;
            for (let i = 0; i < indices.length; i++) {
                const p = points[indices[i]];
                if (!p) continue;
                const x = p.x * vw * scaleX;
                const y = p.y * vh * scaleY;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        };

        // インデックス定義（サーバ側と概ね合わせる）
        const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
        const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
        const NOSE = [1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122];
        const OUTER_LIPS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318];
        const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

        drawPoints(LEFT_EYE, '#00ff00');    // 左目: 緑
        drawPoints(RIGHT_EYE, '#0080ff');   // 右目: 青
        drawPoints(NOSE, '#ff3333');        // 鼻: 赤
        drawPoints(OUTER_LIPS, '#ff33ff');  // 口: 紫
        drawPoints(FACE_OVAL, '#ffd400');   // 顔輪郭: 黄

        // 主要キーポイントにラベル
        const labelPoint = (idx, label, color) => {
            const p = points[idx];
            if (!p) return;
            const x = p.x * vw * scaleX;
            const y = p.y * vh * scaleY;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#222';
            ctx.font = '12px sans-serif';
            ctx.fillText(label, x + 4, y - 4);
        };
        labelPoint(33, 'L-eye L', '#00ff00');
        labelPoint(133, 'L-eye R', '#00ff00');
        labelPoint(362, 'R-eye L', '#0080ff');
        labelPoint(263, 'R-eye R', '#0080ff');
        labelPoint(1, 'Nose tip', '#ff3333');
        labelPoint(61, 'Mouth L', '#ff33ff');
        labelPoint(291, 'Mouth R', '#ff33ff');
        labelPoint(234, 'Face L', '#ffd400');
        labelPoint(454, 'Face R', '#ffd400');
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