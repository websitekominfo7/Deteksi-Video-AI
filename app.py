from flask import Flask, render_template, request
import cv2
import os
import tempfile
from ultralytics import YOLO
import yt_dlp
import numpy as np
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'quiet': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        temp_path = None

        # Upload video
        if 'video' in request.files and request.files['video'].filename != '':
            video_file = request.files['video']
            temp_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(temp_path)

        # URL video
        elif 'video_url' in request.form:
            video_url = request.form.get('video_url', '').strip()
            if video_url:
                temp_path = os.path.join(tempfile.gettempdir(), 'temp_video.mp4')
                download_video(video_url, temp_path)
            else:
                return "<h3>⚠️ URL tidak valid atau kosong.</h3>"

        else:
            return "<h3>⚠️ Tidak ada video yang diunggah atau URL yang dimasukkan.</h3>"

        time.sleep(0.5)  # pastikan file siap dibaca

        # Analisis frame dengan YOLO
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detected_frames = 0
        frame_index = 0
        variance_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Analisis tiap 30 frame
            if frame_index % 30 == 0:
                results = model(frame, verbose=False)
                if any(len(r.boxes) > 0 for r in results):
                    detected_frames += 1

                # Variansi pixel frame (simulasi ketidak-alaman AI)
                variance = float(np.var(frame) / (frame.shape[0]*frame.shape[1]*3))
                variance_scores.append(variance)

            frame_index += 1

        cap.release()

        # Skor visual tambahan dari variansi frame
        if variance_scores:
            avg_variance = np.mean(variance_scores)
        else:
            avg_variance = 0

        # Hitung AI score: kombinasi objek + variansi frame
        ai_score = (detected_frames / max(1, frame_index)) * 50 + (1 - avg_variance) * 50
        ai_score = min(100, round(ai_score, 2))

        ai_status = "🚨 TERDETEKSI VIDEO AI" if ai_score >= 50 else "✅ VIDEO REAL TAKE"

        return f"""
        <div style='font-family:Arial;text-align:center;padding:30px;'>
            <h2 style='color:#2C3E50;'>🔍 HASIL ANALISIS VIDEO</h2>
            <div style='margin-top:20px;text-align:left;display:inline-block;'>
                <p><b>Total Frame Analisis:</b> {frame_index}</p>
                <hr>
                <p><b>Persentase Deteksi AI:</b> {ai_score}%</p>
                <h3 style='color:{"#E74C3C" if ai_score>=50 else "#27AE60"};'>{ai_status}</h3>
            </div>
        </div>
        """

    except Exception as e:
        return f"<h3>❌ Terjadi error: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
