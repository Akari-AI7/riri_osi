from flask import Flask, render_template, jsonify, Response, request, send_from_directory
import os
from datetime import datetime

# 重いライブラリは起動時例外を避けるため遅延インポート/ガード
LIBS_OK = True
_import_error_message = None
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import mediapipe as mp  # type: ignore
    from face_compare_heatmap import (  # type: ignore
        extract_landmarks,
        draw_landmarks, 
        calculate_differences,
        FaceFeatureAnalyzer,
        LANDMARK_POINTS,
        mp_face_mesh
    )
except Exception as _e:  # ImportError など
    LIBS_OK = False
    _import_error_message = str(_e)
    cv2 = None  # type: ignore
    np = None  # type: ignore
    mp = None  # type: ignore

app = Flask(__name__)

# グローバル変数（Web特有の状態管理）
camera = None
face_mesh_instance = None
capture_result = None
comparison_result = None

# ファイル保存設定
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)
PAST_IMAGE_PATH = os.path.join(SAVE_DIR, "past.jpg")

def start_camera():
    """カメラ開始"""
    global camera, face_mesh_instance
    if not LIBS_OK:
        return False
    try:
        camera = cv2.VideoCapture(0)
        face_mesh_instance = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
        )
        return True
    except Exception as e:
        print(f"カメラ開始エラー: {e}")
        return False

def stop_camera():
    """カメラ停止"""
    global camera, face_mesh_instance
    if camera:
        camera.release()
        camera = None
    if face_mesh_instance:
        face_mesh_instance.close()
        face_mesh_instance = None

def init_face_mesh():
    global face_mesh_instance
    if not LIBS_OK:
        return
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_instance = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def capture_current_frame():
    """現在のフレームを撮影"""
    global capture_result
    if not LIBS_OK:
        return {"success": False, "message": f"依存ライブラリの読み込みに失敗しました: {_import_error_message}"}
    if camera is None:
        return {"success": False, "message": "カメラが開始されていません"}
    
    ret, frame = camera.read()
    if not ret:
        return {"success": False, "message": "フレームの取得に失敗しました"}
    
    # 元のモジュールの関数を使用
    landmarks = extract_landmarks(frame, face_mesh_instance)
    if landmarks is None:
        return {"success": False, "message": "顔が検出されませんでした"}
    
    # 撮影処理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(SAVE_DIR, f"{timestamp}_raw.jpg")
    lm_path = os.path.join(SAVE_DIR, f"{timestamp}_landmarks.jpg")

    # 生画像保存
    cv2.imwrite(raw_path, frame)

    # 元のモジュールの関数を使用してランドマーク描画
    lm_frame = draw_landmarks(frame.copy(), landmarks)
    cv2.imwrite(lm_path, lm_frame)

    # 過去画像更新
    cv2.imwrite(PAST_IMAGE_PATH, frame)

    capture_result = {
        "timestamp": timestamp,
        "raw_path": raw_path,
        "landmarks_path": lm_path
    }
    
    return {"success": True, "message": "撮影が完了しました"}

def compare_current_frame():
    """現在のフレームと過去の画像を比較"""
    global comparison_result
    if not LIBS_OK:
        return {"success": False, "message": f"依存ライブラリの読み込みに失敗しました: {_import_error_message}"}
    if camera is None:
        return {"success": False, "message": "カメラが開始されていません"}
    
    if not os.path.exists(PAST_IMAGE_PATH):
        return {"success": False, "message": "先に撮影を行ってください"}
    
    ret, frame = camera.read()
    if not ret:
        return {"success": False, "message": "フレームの取得に失敗しました"}
    
    # 元のモジュールの関数を使用
    past_img = cv2.imread(PAST_IMAGE_PATH)
    past_lm = extract_landmarks(past_img, face_mesh_instance)
    current_lm = extract_landmarks(frame, face_mesh_instance)
    
    if past_lm is None or current_lm is None:
        return {"success": False, "message": "顔が検出されませんでした"}
    
    # 元のモジュールの関数を使用して差異計算
    diffs = calculate_differences(past_lm, current_lm)
    
    # 元のモジュールのクラスを使用してAI分析
    analyzer = FaceFeatureAnalyzer()
    descriptions = analyzer.generate_feature_descriptions(diffs)
    
    # 有意な変化の検出
    significant_changes = [k for k, v in diffs.items() if abs(v['change_percent']) > 5.0]
    
    comparison_result = {
        "numerical_data": diffs,
        "descriptions": descriptions,
        "significant_changes": significant_changes,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    return {"success": True, "message": "比較分析が完了しました"}

# Flask エンドポイント
@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/health')
def health():
    status = {"status": "ok", "libs_ok": LIBS_OK}
    if not LIBS_OK:
        status["error"] = _import_error_message
    return jsonify(status)

# 静的に保存した撮影ファイル配信用
@app.route('/captures/<path:filename>')
def serve_captures(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=False)

# ブラウザの自動リクエストに対する簡易favicon応答（404抑止）
@app.route('/favicon.ico')
def favicon():
    return ("", 204)

# ========== アップロード型フロー API ==========
def _read_uploaded_image_to_cv2(upload_file):
    import io
    import numpy as np  # type: ignore
    file_bytes = np.frombuffer(upload_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # type: ignore
    return img

@app.route('/upload_base', methods=['POST'])
def upload_base():
    if not LIBS_OK:
        return jsonify({"success": False, "error": f"依存ライブラリの読み込みに失敗しました: {_import_error_message}"}), 500
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "画像ファイルがありません"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "ファイル名が不正です"}), 400

    img = _read_uploaded_image_to_cv2(file)
    if img is None:
        return jsonify({"success": False, "error": "画像の読み込みに失敗しました"}), 400

    # ランドマーク抽出
    global face_mesh_instance
    if face_mesh_instance is None:
        init_face_mesh()
    lms = extract_landmarks(img, face_mesh_instance)
    if lms is None:
        return jsonify({"success": False, "error": "顔が検出されませんでした"}), 200

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_name = f"{timestamp}_base_raw.jpg"
    lm_name = f"{timestamp}_base_landmarks.jpg"
    raw_path = os.path.join(SAVE_DIR, raw_name)
    lm_path = os.path.join(SAVE_DIR, lm_name)

    cv2.imwrite(raw_path, img)  # type: ignore
    lm_img = draw_landmarks(img.copy(), lms)
    cv2.imwrite(lm_path, lm_img)  # type: ignore

    # 過去画像として確定
    cv2.imwrite(PAST_IMAGE_PATH, img)  # type: ignore

    return jsonify({
        "success": True,
        "message": "基準画像を設定しました",
        "landmark_image": f"/captures/{lm_name}"
    })

@app.route('/compare', methods=['POST'])
def compare_uploaded():
    if not LIBS_OK:
        return jsonify({"success": False, "error": f"依存ライブラリの読み込みに失敗しました: {_import_error_message}"}), 500
    if not os.path.exists(PAST_IMAGE_PATH):
        return jsonify({"success": False, "error": "先に基準画像をアップロードしてください"}), 200
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "画像ファイルがありません"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "ファイル名が不正です"}), 400

    current_img = _read_uploaded_image_to_cv2(file)
    if current_img is None:
        return jsonify({"success": False, "error": "画像の読み込みに失敗しました"}), 400

    # ランドマーク
    global face_mesh_instance
    if face_mesh_instance is None:
        init_face_mesh()
    past_img = cv2.imread(PAST_IMAGE_PATH)  # type: ignore
    past_lm = extract_landmarks(past_img, face_mesh_instance)
    current_lm = extract_landmarks(current_img, face_mesh_instance)
    if past_lm is None or current_lm is None:
        return jsonify({"success": False, "error": "顔が検出されませんでした"}), 200

    diffs = calculate_differences(past_lm, current_lm)
    analyzer = FaceFeatureAnalyzer()
    descriptions = analyzer.generate_feature_descriptions(diffs)

    return jsonify({
        "success": True,
        "differences": diffs,
        "descriptions": descriptions
    })

@app.route('/results')
def results():
    """結果ページ"""
    return render_template('results.html')

@app.route('/start_camera', methods=['POST'])
def start_camera_route():
    """カメラ開始API"""
    if not LIBS_OK:
        return jsonify({"success": False, "message": f"依存ライブラリの読み込みに失敗しました: {_import_error_message}"})
    success = start_camera()
    if success:
        return jsonify({"success": True, "message": "カメラが開始されました"})
    else:
        return jsonify({"success": False, "message": "カメラの開始に失敗しました"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera_route():
    """カメラ停止API"""
    stop_camera()
    return jsonify({"success": True, "message": "カメラが停止されました"})

@app.route('/capture', methods=['POST'])
def capture_route():
    """撮影API"""
    result = capture_current_frame()
    return jsonify(result)

@app.route('/compare', methods=['POST'])
def compare_route():
    """比較API"""
    result = compare_current_frame()
    return jsonify(result)

@app.route('/get_results')
def get_results():
    """結果取得API"""
    global capture_result, comparison_result
    return jsonify({
        "capture_result": capture_result,
        "comparison_result": comparison_result
    })

@app.route('/video_feed')
def video_feed():
    """ビデオストリーミング"""
    if not LIBS_OK:
        return Response("", status=503)
    def generate():
        global camera, face_mesh_instance
        if camera is None:
            camera = cv2.VideoCapture(0)
            init_face_mesh()

        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            # 元のモジュールの関数を使用
            landmarks = extract_landmarks(frame, face_mesh_instance)
            if landmarks is not None:
                frame = draw_landmarks(frame, landmarks)
            
            # ガイド描画
            h, w = frame.shape[:2]
            overlay = frame.copy()
            center = (w//2, h//2)
            axes = (w//4, h//3)
            cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 255, 255), -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 200, 200), 2)
            
            # フレームをJPEG形式でエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)