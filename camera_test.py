import cv2
import sys

def test_camera():
    print("カメラテストを開始します...")
    
    # 利用可能なカメラインデックスをテスト
    for i in range(5):  # 0-4まで試す
        print(f"カメラインデックス {i} をテスト中...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ カメラインデックス {i} が利用可能です")
                height, width = frame.shape[:2]
                print(f"   解像度: {width}x{height}")
                
                # テスト用ウィンドウを表示（5秒間）
                cv2.imshow(f'Camera Test {i}', frame)
                cv2.waitKey(2000)  # 2秒間表示
                cv2.destroyAllWindows()
                
                cap.release()
                return i  # 最初に見つかったカメラのインデックスを返す
            else:
                print(f"❌ カメラインデックス {i} は開けませんが、フレーム取得に失敗")
        else:
            print(f"❌ カメラインデックス {i} は利用できません")
        
        cap.release()
    
    print("❌ 利用可能なカメラが見つかりませんでした")
    return None

def check_opencv_backends():
    """OpenCVのバックエンドを確認"""
    print("\n=== OpenCV情報 ===")
    print(f"OpenCV バージョン: {cv2.__version__}")
    
    # 利用可能なバックエンドを表示
    backends = cv2.videoio_registry.getCameraBackends()
    print(f"利用可能なカメラバックエンド: {backends}")

if __name__ == "__main__":
    check_opencv_backends()
    camera_index = test_camera()
    
    if camera_index is not None:
        print(f"\n🎉 カメラインデックス {camera_index} を使用してください")
        print(f"app.pyの cv2.VideoCapture(0) を cv2.VideoCapture({camera_index}) に変更してください")
    else:
        print("\n⚠️ カメラの問題を解決する必要があります")