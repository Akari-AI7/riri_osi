import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import platform

mp_face_mesh = mp.solutions.face_mesh


SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)
PAST_IMAGE_PATH = os.path.join(SAVE_DIR, "past.jpg")

# MediaPipe FaceMeshの正確なランドマーク定義
LANDMARK_POINTS = {
    # 左目の輪郭（時計回り）
    'LEFT_EYE': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    
    # 右目の輪郭（時計回り）
    'RIGHT_EYE': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    
    # 左眉毛
    'LEFT_EYEBROW': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    
    # 右眉毛  
    'RIGHT_EYEBROW': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
    
    # 鼻の輪郭
    'NOSE': [1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    
    # 口の外側輪郭
    'OUTER_LIPS': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318],
    
    # 口の内側輪郭
    'INNER_LIPS': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    
    # 顔の外側輪郭
    'FACE_OVAL': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    
    # 重要なポイント（距離計算用）
    'KEY_POINTS': {
        'left_eye_left': 33,      # 左目の左端
        'left_eye_right': 133,    # 左目の右端
        'right_eye_left': 362,    # 右目の左端
        'right_eye_right': 263,   # 右目の右端
        'nose_left': 131,         # 鼻の左端
        'nose_right': 358,        # 鼻の右端
        'mouth_left': 61,         # 口の左端
        'mouth_right': 291,       # 口の右端
        'face_left': 234,         # 顔の左端
        'face_right': 454,        # 顔の右端
        'nose_tip': 1,            # 鼻先
        'chin': 18,               # 顎
        'forehead': 9,            # 額
        'left_eye_center': 468,   # 左目中心（瞳孔）※実際は計算で求める
        'right_eye_center': 469,  # 右目中心（瞳孔）※実際は計算で求める
    }
}

class FaceFeatureAnalyzer:
    """顔特徴分析クラス"""
    def __init__(self):
        pass

    def generate_feature_descriptions(self, changes):
        """特徴変化の自然言語記述生成（輪郭・むくみ・対称性にも対応）"""
        descriptions = []

        for feature, data in changes.items():
            if isinstance(data, dict) and "change_percent" in data:
                change_percent = data["change_percent"]
                abs_percent = abs(change_percent)

                # 変化量に応じた表現
                if abs_percent < 5:
                    magnitude = "わずかに"
                elif abs_percent < 10:
                    magnitude = "やや"
                else:
                    magnitude = "顕著に"

                # 各特徴ごとの表現ルール
                if feature == "左目の幅":
                    direction = "大きくなっています" if change_percent > 0 else "小さくなっています"
                elif feature == "左目の高さ":
                    direction = "高くなっています" if change_percent > 0 else "低くなっています"
                elif feature == "右目の幅":
                    direction = "大きくなっています" if change_percent > 0 else "小さくなっています"
                elif feature == "右目の高さ":
                    direction = "高くなっています" if change_percent > 0 else "低くなっています"
                elif feature == "両目間の距離":
                    direction = "広がっています" if change_percent > 0 else "狭まっています"
                elif feature == "鼻の幅":
                    direction = "長くなっています" if change_percent > 0 else "短くなっています"
                elif feature == "口の幅":
                    direction = "横に広がっています" if change_percent > 0 else "横幅が狭まっています"
                elif feature == "輪郭":
                    direction = "シャープになっています" if change_percent < 0 else "丸みを帯びています"
                elif feature == "顔の幅":
                    direction = "大きくなっています" if change_percent > 0 else "小さくなっています"
                else:
                    direction = "変化があります"

                description = f"{feature}が{magnitude}{direction}"

                # 大きめの変化は数値を表示
                if abs_percent > 10:
                    description += f"（{change_percent:+.1f}%）"

                descriptions.append(description)

        if not descriptions:
            descriptions.append("顔の特徴に顕著な変化は見られませんでした。")

        # 総合的な洞察を追加
        self.add_insights(descriptions, changes)
        return descriptions

    def add_insights(self, descriptions, changes):
        """変化の洞察を追加"""
        if "輪郭" in changes and "顔の幅" in changes:
            if changes["顔の幅"]["change_percent"] > 0 and changes["輪郭"]["change_percent"] > 0:
                descriptions.append("全体的に顔がふっくらしており、むくみが目立ちます。")
            elif changes["顔の幅"]["change_percent"] < 0 and changes["輪郭"]["change_percent"] < 0:
                descriptions.append("顔全体がすっきりしてシャープになっています。")

# フォント設定関数
def setup_japanese_font():
    """日本語フォントを設定"""
    try:
        system = platform.system()
        if system == "Windows":
            # Windows環境
            font_paths = [
                "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
                "C:/Windows/Fonts/msgothic.ttc", 
                "C:/Windows/Fonts/meiryo.ttc",
                "C:/Windows/Fonts/arial.ttf"
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/Arial.ttf"
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 20)
        
        # フォールバック
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()

# OpenCV画像にPILで日本語テキストを描画
def draw_japanese_text(img, text, position, font, color=(255, 255, 255)):
    """OpenCV画像に日本語テキストを描画"""
    # OpenCV -> PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # テキスト描画
    draw.text(position, text, font=font, fill=color)
    
    # PIL -> OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ランドマーク抽出関数
def extract_landmarks(image, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        h, w = image.shape[:2]
        points = []
        for lm in results.multi_face_landmarks[0].landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])
        return np.array(points)
    return None

# 目の中心を計算する関数
def calculate_eye_center(landmarks, eye_points):
    """目のランドマークから中心座標を計算"""
    eye_coords = landmarks[eye_points]
    return np.mean(eye_coords, axis=0).astype(int)

# ランドマーク描画
def draw_landmarks(image, landmarks):
    img = image.copy()
    
    # 左目（緑）
    for point in LANDMARK_POINTS['LEFT_EYE']:
        if point < len(landmarks):
            cv2.circle(img, tuple(landmarks[point]), 2, (0, 255, 0), -1)
    
    # 右目（青）
    for point in LANDMARK_POINTS['RIGHT_EYE']:
        if point < len(landmarks):
            cv2.circle(img, tuple(landmarks[point]), 2, (255, 0, 0), -1)
    
    # 鼻（赤）
    for point in LANDMARK_POINTS['NOSE'][:10]:  # 鼻の主要ポイントのみ
        if point < len(landmarks):
            cv2.circle(img, tuple(landmarks[point]), 2, (0, 0, 255), -1)
    
    # 口（紫）
    for point in LANDMARK_POINTS['OUTER_LIPS'][:12]:  # 口の主要ポイントのみ
        if point < len(landmarks):
            cv2.circle(img, tuple(landmarks[point]), 2, (255, 0, 255), -1)
    
    # キーポイント（黄色、大きめ）
    key_points = LANDMARK_POINTS['KEY_POINTS']
    for name, point in key_points.items():
        if point < len(landmarks):
            cv2.circle(img, tuple(landmarks[point]), 4, (0, 255, 255), -1)
            # ラベル表示
            cv2.putText(img, name, (landmarks[point][0] + 5, landmarks[point][1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return img

# ===== 差異計算関数 =====
def calculate_differences(lm_past, lm_current):
    diffs = {}
    key_points = LANDMARK_POINTS['KEY_POINTS']
    
    # 左目の幅
    left_eye_width_past = np.linalg.norm(lm_past[key_points['left_eye_left']] - lm_past[key_points['left_eye_right']])
    left_eye_width_current = np.linalg.norm(lm_current[key_points['left_eye_left']] - lm_current[key_points['left_eye_right']])
    pixel_change = left_eye_width_current - left_eye_width_past
    change_percent = (pixel_change / left_eye_width_past) * 100 if left_eye_width_past != 0 else 0
    diffs['左目の幅'] =  {"pixel_change": pixel_change,"change_percent": change_percent}
    # 右目の幅
    right_eye_width_past = np.linalg.norm(lm_past[key_points['right_eye_left']] - lm_past[key_points['right_eye_right']])
    right_eye_width_current = np.linalg.norm(lm_current[key_points['right_eye_left']] - lm_current[key_points['right_eye_right']])
    pixel_change = right_eye_width_current - right_eye_width_past
    change_percent = (pixel_change / right_eye_width_past) * 100 if right_eye_width_past != 0 else 0
    diffs['右目の幅'] =  {"pixel_change": pixel_change,"change_percent": change_percent}

    # 両目間の距離
    eye_distance_past = np.linalg.norm(lm_past[key_points['left_eye_right']] - lm_past[key_points['right_eye_left']])
    eye_distance_current = np.linalg.norm(lm_current[key_points['left_eye_right']] - lm_current[key_points['right_eye_left']])
    pixel_change = eye_distance_current - eye_distance_past
    change_percent = (pixel_change / eye_distance_past) * 100 if eye_distance_past != 0 else 0
    diffs['両目間の距離'] =  {"pixel_change": pixel_change,"change_percent": change_percent}
    # 鼻の幅
    nose_width_past = np.linalg.norm(lm_past[key_points['nose_left']] - lm_past[key_points['nose_right']])
    nose_width_current = np.linalg.norm(lm_current[key_points['nose_left']] - lm_current[key_points['nose_right']])
    pixel_change = nose_width_current - nose_width_past 
    change_percent = (pixel_change / nose_width_past ) * 100 if nose_width_past != 0 else 0
    diffs['鼻の幅'] = {"pixel_change": pixel_change,"change_percent": change_percent}

    # 口の幅
    mouth_width_past = np.linalg.norm(lm_past[key_points['mouth_left']] - lm_past[key_points['mouth_right']])
    mouth_width_current = np.linalg.norm(lm_current[key_points['mouth_left']] - lm_current[key_points['mouth_right']])
    pixel_change = mouth_width_current - mouth_width_past
    change_percent = (pixel_change / mouth_width_past) * 100 if mouth_width_past != 0 else 0
    diffs['口の幅'] =  {"pixel_change": pixel_change,"change_percent": change_percent}

    # 顔の幅
    face_width_past = np.linalg.norm(lm_past[key_points['face_left']] - lm_past[key_points['face_right']])
    face_width_current = np.linalg.norm(lm_current[key_points['face_left']] - lm_current[key_points['face_right']])
    pixel_change = face_width_current - face_width_past
    change_percent = (pixel_change / face_width_past) * 100 if face_width_past != 0 else 0
    diffs['顔の幅'] =  {"pixel_change": pixel_change,"change_percent": change_percent}

    # 顔の高さ（額から顎まで）
    face_height_past = np.linalg.norm(lm_past[key_points['forehead']] - lm_past[key_points['chin']])
    face_height_current = np.linalg.norm(lm_current[key_points['forehead']] - lm_current[key_points['chin']])
    pixel_change = face_height_current - face_height_past
    change_percent = (pixel_change / face_height_past) * 100 if face_height_past != 0 else 0
    diffs['顔の高さ'] =  {"pixel_change": pixel_change,"change_percent": change_percent}

    # 左目の高さ（上下の幅）
    left_eye_top = lm_past[159]  # 左目上部
    left_eye_bottom = lm_past[145]  # 左目下部
    left_eye_height_past = np.linalg.norm(left_eye_top - left_eye_bottom)
    
    left_eye_top = lm_current[159]
    left_eye_bottom = lm_current[145]
    left_eye_height_current = np.linalg.norm(left_eye_top - left_eye_bottom)
    pixel_change = left_eye_height_current - left_eye_height_past
    change_percent = (pixel_change / left_eye_height_past) * 100 if left_eye_height_past != 0 else 0
    diffs['左目の高さ'] = {"pixel_change": pixel_change,"change_percent": change_percent}


    # 右目の高さ
    right_eye_top = lm_past[386]  # 右目上部
    right_eye_bottom = lm_past[374]  # 右目下部
    right_eye_height_past = np.linalg.norm(right_eye_top - right_eye_bottom)
    
    right_eye_top = lm_current[386]
    right_eye_bottom = lm_current[374]
    right_eye_height_current = np.linalg.norm(right_eye_top - right_eye_bottom)
    pixel_change = right_eye_height_current - right_eye_height_past
    change_percent = (pixel_change / right_eye_height_past) * 100 if right_eye_height_past != 0 else 0
    diffs['右目の高さ'] = {"pixel_change": pixel_change,"change_percent": change_percent}

    return diffs
    
# ================== 撮影処理 ==================
def capture_image(frame, landmarks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(SAVE_DIR, f"{timestamp}_raw.jpg")
    lm_path = os.path.join(SAVE_DIR, f"{timestamp}_landmarks.jpg")

    # 生画像保存（ガイドなし）
    cv2.imwrite(raw_path, frame)

    # ランドマーク描画した画像保存
    cv2.imwrite(lm_path, draw_landmarks(frame.copy(), landmarks))

    # 過去画像更新
    cv2.imwrite(PAST_IMAGE_PATH, frame)

    print(f"[SAVED] Raw: {raw_path}, Landmarks: {lm_path}, Past: {PAST_IMAGE_PATH}")


# ================== 比較処理 ==================
def compare_images(frame, face_mesh):
    if not os.path.exists(PAST_IMAGE_PATH):
        print("[WARN] 先に 's' で撮影してください")
        return

    past_img = cv2.imread(PAST_IMAGE_PATH)
    past_lm = extract_landmarks(past_img, face_mesh)
    current_lm = extract_landmarks(frame, face_mesh)

    if past_lm is not None and current_lm is not None:
        diffs = calculate_differences(past_lm, current_lm)

        print("\n" + "="*60)
        print("=== 顔分析結果（数値データ） ===")
        print("="*60)
        for part, data in diffs.items():
            pixel_change = data['pixel_change']
            change_percent = data['change_percent']
            status = "拡大" if pixel_change > 0 else "縮小"
            print(f"{part:12s}: {pixel_change:+6.2f} px ({change_percent:+5.1f}%) [{status}]")
        print("="*60)
        
        # 有意な変化の検出
        significant_changes = [k for k, v in diffs.items() if abs(v['change_percent']) > 5.0]
        if significant_changes:
            print(f"有意な変化: {', '.join(significant_changes)}")
        else:
            print("大きな変化は検出されませんでした")

        print("\n" + "="*60)
        print("=== AI分析結果（自然言語） ===")
        print("="*60)
        
        # AI分析による自然言語記述
        analyzer = FaceFeatureAnalyzer()
        descriptions = analyzer.generate_feature_descriptions(diffs)
        for i, desc in enumerate(descriptions, 1):
            print(f"{i:2d}. {desc}")
            
        print("="*60)
        
    else:
        print("[WARN] 顔が検出できませんでした")

# ===== カメラ起動 =====
def main():
    print("[INFO] MediaPipe FaceMesh 正確なランドマーク比較システム")
    print("[INFO] OpenCV バージョン:", cv2.__version__)
    cap = cv2.VideoCapture(0)
    
    # 日本語フォント設定
    font = setup_japanese_font()
    print("[情報] 日本語フォントを設定しました")

    # AI分析器の初期化
    analyzer = FaceFeatureAnalyzer()
    print("[情報] AI分析器を初期化しました")
    
    if not cap.isOpened():
        print("[ERROR] カメラを開けませんでした")
        return
    
    
    print("[INFO] 's' = 撮影 / 'c' = 比較  / 'q' = 終了")


    # ★ FaceMeshを最初に1回だけ作る（高速 & 警告抑制）
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    ) as face_mesh:


        while True:
           ret, frame = cap.read()
           if not ret:
               break
    
           frame_disp = frame.copy()
           h, w = frame_disp.shape[:2]

           # ランドマーク描画
           landmarks = extract_landmarks(frame_disp,face_mesh)
           if landmarks is not None:
               frame_disp = draw_landmarks(frame_disp, landmarks)


           # ===== 卵型ガイドを描画 =====

           overlay = frame_disp.copy()
           center = (w//2, h//2)
           axes = (w//4, h//3)
           cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 255, 255), -1)  # 塗りつぶし
           alpha = 0.3
           cv2.addWeighted(overlay, alpha, frame_disp, 1 - alpha, 0, frame_disp)
           cv2.ellipse(frame_disp,  center, axes,0, 0, 360, (0, 200, 200), 2)
           
           # 操作ガイド表示
           frame_disp = draw_japanese_text(frame_disp, "統合顔分析システム", (10, 30), font, (255, 255, 255))
           frame_disp = draw_japanese_text(frame_disp, "s:撮影 c:比較  q:終了", (10, h-20), font, (255, 255, 255))

           cv2.imshow("Camera Preview", frame_disp)
           key = cv2.waitKey(1) & 0xFF
    

           # 撮影
           if key == ord('s')and landmarks is not None:
               capture_image(frame, landmarks)
           elif key == ord('c'):
               compare_images(frame, face_mesh)
           elif key == ord('q'):
               break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] カメラを正常に終了しました")


if __name__ == "__main__":
    main()