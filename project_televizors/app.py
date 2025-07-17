
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Предобученная модель YOLOv8

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Обработка нейросетью
    results = model(img)
    output_img = results[0].plot()

    # Сохраняем результат
    os.makedirs('static', exist_ok=True)
    cv2.imwrite('static/result.jpg', output_img)

    return jsonify({'count': len(results[0].boxes)})

@app.route('/result')
def result_image():
    return send_file('static/result.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
