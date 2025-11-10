import cv2
import os

def analyze_video(video_path, output_path):
    """
    Analisis sederhana video menggunakan OpenCV.
    Mendeteksi pergerakan dan menandai area gerak dengan kotak hijau.
    """
    # Pastikan video ada
    if not os.path.exists(video_path):
        return {"status": "error", "message": "File video tidak ditemukan"}

    # Buka video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Gagal membuka video"}

    # Gunakan frame pertama sebagai pembanding
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Definisikan writer untuk hasil output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

    frame_count = 0
    motion_detected = False

    while cap.isOpened():
        # Hitung perbedaan antar frame
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Tandai area gerak
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tulis frame ke video output
        out.write(frame1)

        # Update frame
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return {
        "status": "success",
        "frames_analyzed": frame_count,
        "motion_detected": motion_detected
    }
