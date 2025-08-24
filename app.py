from flask import Flask, request, render_template, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def calculate_differences(image1, image2):
    face_landmarks1 = face_recognition.face_landmarks(image1)
    face_landmarks2 = face_recognition.face_landmarks(image2)

    if not face_landmarks1 or not face_landmarks2:
        return None, None

    points1 = []
    points2 = []

    for key in face_landmarks1[0]:
        points1.extend(face_landmarks1[0][key])
    for key in face_landmarks2[0]:
        points2.extend(face_landmarks2[0][key])

    points1 = np.array(points1)
    points2 = np.array(points2)

    diffs = np.linalg.norm(points1 - points2, axis=1)
    avg_diff = np.mean(diffs)

    # 部位ごとの差異（例: 目・鼻・口）
    part_diffs = {}
    for part in face_landmarks1[0]:
        pts1 = np.array(face_landmarks1[0][part])
        pts2 = np.array(face_landmarks2[0][part])
        part_diffs[part] = float(np.mean(np.linalg.norm(pts1 - pts2, axis=1)))

    return avg_diff, part_diffs

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image1" not in request.files or "image2" not in request.files:
            return "画像を2枚アップロードしてください。"

        file1 = request.files["image1"]
        file2 = request.files["image2"]

        if file1.filename == "" or file2.filename == "":
            return "画像を選択してください。"

        path1 = os.path.join(UPLOAD_FOLDER, "image1.jpg")
        path2 = os.path.join(UPLOAD_FOLDER, "image2.jpg")
        file1.save(path1)
        file2.save(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        result = calculate_differences(img1, img2)
        if result is None:
            return "顔が検出されませんでした。"

        avg_diff, part_diffs = result

       

if __name__ == "__main__":
    app.run(debug=True)
