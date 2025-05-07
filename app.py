from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model("model/lung_model.h5")
class_names = ['LUAD', 'LUSC', 'NORMAL']

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    file_path = ''

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file:
            # Generate a unique filename to avoid conflicts
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Load and preprocess image
            img = image.load_img(file_path, target_size=(224, 224))
            img_tensor = image.img_to_array(img) / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)

            # Make prediction
            result = model.predict(img_tensor)
            predicted_class = class_names[np.argmax(result)]
            prediction = f"{predicted_class} (Confidence: {round(np.max(result) * 100, 2)}%)"

            return render_template('index.html', prediction=prediction, image='/' + file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
