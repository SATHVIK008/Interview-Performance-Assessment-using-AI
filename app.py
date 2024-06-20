import cv2
import tensorflow as tf
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

def predict_emotion_percentages(video_path, num_frames_to_extract):
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    model = tf.keras.models.load_model('final_model_weights.hdf5')
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    frames_to_predict = []
    frame_interval = 0  # Variable to store the interval between frames to be extracted

    with cv2.VideoCapture(video_path) as video:
        # Calculate the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate the interval to skip frames
        frame_interval = total_frames // num_frames_to_extract

        for frame_number in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break

            # Process every nth frame
            if frame_number % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceDetect.detectMultiScale(gray, 1.3, 3)

                for x, y, w, h in faces:
                    sub_face_img = gray[y: y + h, x: x + w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalized = resized / 255.0
                    reshaped = np.reshape(normalized, (1, 48, 48, 1))
                    frames_to_predict.append(reshaped)

                if len(frames_to_predict) == num_frames_to_extract:
                    break

    # Make predictions on the extracted frames
    predictions = []
    for frame_data in frames_to_predict:
        result = model.predict(frame_data)
        label = np.argmax(result, axis=1)[0]
        predictions.append(label)

    # Calculate percentage for each class_name
    total_predictions = len(predictions)
    percentage_dict = {}
    for class_name in class_names:
        class_count = predictions.count(class_names.index(class_name))
        percentage = (class_count / total_predictions) * 100
        percentage_dict[class_name] = percentage

    # Calculate confidence percentage (Happy + Neutral + Surprise + Angry)
    confidence_percentage = (percentage_dict["Happy"] + percentage_dict["Neutral"] + percentage_dict["Surprise"] + percentage_dict["Angry"])
    # Calculate stress percentage (Disgust + Fear + Sad)
    stress_percentage = (percentage_dict["Disgust"] + percentage_dict["Fear"] + percentage_dict["Sad"])

    return confidence_percentage, stress_percentage


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    health_condition = request.form.get('health_condition')

    if file.filename == '':
        return "No file selected"

    if file and file.filename.endswith('.mp4'):
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        num_frames_to_extract = 100
        try:
            confidence_percentage, stress_percentage = predict_emotion_percentages(video_path, num_frames_to_extract)
        except Exception as e:
            os.remove(video_path)
            return f"Error processing video: {e}"

        os.remove(video_path)
        return render_template('result1.html', confidence_percentage=confidence_percentage, stress_percentage=stress_percentage)
    else:
        return "Unsupported file format. Please upload an MP4 video file."


if __name__ == '__main__':
    app.run(debug=True)


