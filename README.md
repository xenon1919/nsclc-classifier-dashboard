
# NSCLC Classifier Dashboard 🧬

A Flask-based web application to predict Non-Small Cell Lung Cancer (NSCLC) subtypes—LUAD, LUSC, or NORMAL—from histopathology images using a deep learning model.

## 🌟 Features

- Upload histopathological lung tissue images
- Predict cancer subtypes instantly using a trained CNN
- Bootstrap-based responsive and modern UI
- Visual display of prediction confidence
- Educational tool for understanding AI-assisted diagnosis

## 🧠 How It Works

1. Upload a microscope image (histopathology scan).
2. The model processes the image through a deep CNN.
3. Outputs a prediction: LUAD / LUSC / NORMAL with confidence.
4. See the prediction and confidence via progress bar and image preview.

## 🛠️ Tech Stack

- Python (Flask)
- Bootstrap 5
- HTML/CSS
- PyTorch or TensorFlow (model backend)
- Font Awesome (icons)

## 🚀 Setup Instructions

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask server:
   ```bash
   python app.py
   ```

3. Open in browser:  
   `http://127.0.0.1:5000`

## ⚠️ Disclaimer

This tool is intended for academic and educational purposes only and is **not approved for clinical or diagnostic use**.

---

© 2025 NSCLC Classifier
