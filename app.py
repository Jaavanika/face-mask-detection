from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model('face_mask_detector_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera_index = 0
camera = None
video_running = False

def open_camera(index):
    global camera
    if camera is not None:
        camera.release()
    camera = cv2.VideoCapture(index)

@app.route('/')
def index():
    return render_template('index.html', video_active=video_running)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result = "No face detected."
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 3))
        prediction = model.predict(reshaped)
        result = "Mask" if prediction[0][0] > 0.5 else "No Mask"
        break

    return render_template('index.html', result=result, image_path=path, video_active=video_running)

@app.route('/start_detection')
def start_detection():
    global video_running
    video_running = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global video_running
    video_running = False
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    global camera, video_running
    open_camera(camera_index)

    def generate_frames():
        while video_running and camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 3))
                prediction = model.predict(reshaped)
                label = "No Mask" if prediction[0][0] > 0.5 else "Mask"
                color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if camera:
            camera.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera')
def switch_camera():
    global camera_index
    camera_index = 1 - camera_index  # Toggle between 0 and 1
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
