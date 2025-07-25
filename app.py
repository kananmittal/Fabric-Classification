import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for scripts and servers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model on server start
MODEL_PATH = 'best_model_final.h5'
model = load_model(MODEL_PATH)

CLASS_MAP = {0: 'No Defect', 1: 'Defect'}

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_histogram(img_path, save_path):
    img = Image.open(img_path).convert('RGB')
    r, g, b = img.split()

    plt.figure(figsize=(6, 3))
    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(np.array(r).ravel(), bins=256, color='red', alpha=0.6, label='Red')
    plt.hist(np.array(g).ravel(), bins=256, color='green', alpha=0.6, label='Green')
    plt.hist(np.array(b).ravel(), bins=256, color='blue', alpha=0.6, label='Blue')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Image preparation
            img_tensor = prepare_image(filepath)
            preds = model.predict(img_tensor)[0]
            class_idx = np.argmax(preds)
            confidence = float(preds[class_idx]) * 100
            label = CLASS_MAP.get(class_idx, "Unknown")

            # Image analysis
            img_pil = Image.open(filepath).convert('RGB')
            width, height = img_pil.size
            mean_intensity = round(np.array(img_pil).mean(), 2)

            # Histogram
            hist_filename = f"hist_{filename}"
            hist_path = os.path.join(app.config['UPLOAD_FOLDER'], hist_filename)
            generate_histogram(filepath, hist_path)

            return render_template(
                'result.html',
                filename=filename,
                label=label,
                confidence=round(confidence, 2),
                width=width,
                height=height,
                mean_intensity=mean_intensity
            )

        except Exception as e:
            flash(f"Error during prediction: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg, gif')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

@app.route('/clear')
def clear():
    # Optional: implement logic to clear files
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
