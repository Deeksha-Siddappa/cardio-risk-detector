# 🫀 Cardio Risk Detector

A Flask-based web app that classifies ECG images to detect possible heart conditions using a trained deep learning model.

## 🔍 Features

- Upload ECG images and get a prediction
- Validates image structure before classification
- Predicts:
  - History of MI
  - Myocardial Infarction
  - Abnormal heartbeat
  - Normal heartbeat

## 🧠 Model

- Trained Keras model (`sequential_ecg_model.h5`)
- Input size: 224x224
- Uses image preprocessing and rule-based validation

## 📊 Datasets

- ECG Images: [Mendeley Dataset](https://data.mendeley.com/datasets/gwbz3fsgp8/2)  
- This dataset contains ECG signals from both healthy individuals and people with cardiovascular issues.

## 🖥️ Run Locally

```bash
git clone https://github.com/your-username/cardio-risk-detector.git
cd cardio-risk-detector
pip install -r requirements.txt
python app.py
Then open: http://127.0.0.1:5000
```
## 📂 Project Structure

app.py – Flask backend
templates/ – upload.html, result.html
uploads/ – Stores uploaded images
sequential_ecg_model.h5 – Trained model

## 📦 Requirements

Make sure you have the following Python packages installed:

- 🧪 `flask` – For creating the web server
- 🧠 `tensorflow` – For loading and running the trained deep learning model
- 🔢 `numpy` – For numerical computations
- 📷 `opencv-python` – For image validation and preprocessing

To install all at once:
```bash
pip install flask tensorflow numpy opencv-python
