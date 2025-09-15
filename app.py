from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import os
from datetime import datetime
import mediapipe as mp

# 元のコードから必要な機能をインポート
from face_compare_heatmap import (
    extract_landmarks,
    draw_landmarks, 
    calculate_differences,
    FaceFeatureAnalyzer,
    LANDMARK_POINTS,
    mp_face_mesh
)

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

@app.route('/results')
def results():
    """結果ページ"""
    return render_template('results.html')

@app.route('/start_camera', methods=['POST'])
def start_camera_route():
    """カメラ開始API"""
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