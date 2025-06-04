from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load your trained model
model = load_model('sequential_ecg_model.h5')
# model = load_model('ecg_heart_risk_model.h5')

# Create upload folder if not exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels = {
    0: 'History of MI',
    1: 'Myocardial Infarction',
    2: 'Abnormal heartbeat',
    3: 'Normal heartbeat'
}

# Confidence threshold
THRESHOLD = 0.6

# Improved Rule-based check for ECG-like image structure
def is_likely_ecg(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    height, width = img.shape
    aspect_ratio = width / height
    if aspect_ratio < 1.0:
        return False  # ECGs are usually landscape

    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    if edge_density < 0.01 or edge_density > 0.3:
        return False

    # Detect horizontal lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=width // 2, maxLineGap=10)
    horizontal_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # nearly horizontal line
                horizontal_lines += 1

    if horizontal_lines < 3:
        return False

    return True

# Image preprocessing function
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    avg_brightness = np.mean(img_array)
    print(f"Average brightness: {avg_brightness}")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for uploading and predicting
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                if not is_likely_ecg(filepath):
                    pred_label = "Invalid image — not a recognizable ECG structure"
                else:
                    img = prepare_image(filepath)
                    preds = model.predict(img)[0]
                    confidence = np.max(preds)
                    pred_class = np.argmax(preds)
                    second_highest = np.partition(preds, -2)[-2]
                    margin = confidence - second_highest

                    print(f"Prediction scores: {preds}")
                    print(f"Top confidence: {confidence}, Margin: {margin}")

                    if confidence < THRESHOLD or margin < 0.15:
                        pred_label = "Invalid image or not a recognizable ECG"
                    else:
                        pred_label = class_labels[pred_class]

            except Exception as e:
                pred_label = f"Error: {str(e)}"

            return render_template('result.html', prediction=pred_label)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
