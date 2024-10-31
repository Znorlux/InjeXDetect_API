from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import onnxruntime as ort
import os

# Crear el directorio de "uploads" si no existe
os.makedirs("uploads", exist_ok=True)

app = Flask(__name__)

# Cargar el modelo ONNX
session = ort.InferenceSession("model.onnx")
window_size = 25

def predict_human_behavior(csv_file_path):
    dataset = pd.read_csv(csv_file_path)
    dataset = dataset.drop(columns=['Timestamp', 'Key'], errors='ignore')
    dataset = dataset.replace(',', '.', regex=True).astype(float)
    
    X_data = dataset[["Interval_ms", "Chars_per_Second", "Error_Count"]].values

    # Crear secuencias de entrada según el tamaño de ventana
    X_seq = []
    for i in range(len(X_data) - window_size + 1):
        X_seq.append(X_data[i:i + window_size])

    X_seq = np.array(X_seq).astype(np.float32)
    
    # Realizar predicciones con ONNX
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: X_seq})[0]
    
    # Calcular el promedio de las predicciones
    mean_prediction = np.mean(predictions)
    result = "Human" if mean_prediction > 0.5 else "Not Human (you are being hacked, dude)"
    return result

@app.route('/')
def index():
    return render_template('index.html')

# Ruta principal para subir el archivo y obtener la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        result = predict_human_behavior(file_path)
    finally:
        os.remove(file_path)

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
