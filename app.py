from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import json
import subprocess
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import base64
import io

# Configurar Matplotlib para que use el backend 'Agg'
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Inicializar MediaPipe para la detección de rostros
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Función para seleccionar puntos clave relevantes
def extract_key_points(landmarks, image_shape):
    selected_indices = {
        'left_eye_center': 159,
        'right_eye_center': 386,
        'left_eye_inner_corner': 133,
        'left_eye_outer_corner': 33,
        'right_eye_inner_corner': 362,
        'right_eye_outer_corner': 263,
        'left_eyebrow_inner_end': 70,
        'left_eyebrow_outer_end': 105,
        'right_eyebrow_inner_end': 336,
        'right_eyebrow_outer_end': 334,
        'nose_tip': 1,
        'mouth_left_corner': 61,
        'mouth_right_corner': 291,
        'mouth_center_top_lip': 13,
        'mouth_center_bottom_lip': 14,
        'mouth_center_upper_lip': 0,  # Centro del labio superior
        'mouth_center_lower_lip': 17   # Centro del labio inferior (ajusta el índice si es necesario)
    }
    
    key_points = {}
    for key, index in selected_indices.items():
        x = int(landmarks[index].x * image_shape[1])
        y = int(landmarks[index].y * image_shape[0])
        key_points[key] = (x, y)
    return key_points

def puntos(img, img_str):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            return 'No se detectaron rostros en la imagen.'

        face_landmarks = results.multi_face_landmarks[0]
        key_points = extract_key_points(face_landmarks.landmark, img.shape)

    coordinates = {
        'left_eye_center_x': key_points['left_eye_center'][0],
        'left_eye_center_y': key_points['left_eye_center'][1],
        'right_eye_center_x': key_points['right_eye_center'][0],
        'right_eye_center_y': key_points['right_eye_center'][1],
        'left_eye_inner_corner_x': key_points['left_eye_inner_corner'][0],
        'left_eye_inner_corner_y': key_points['left_eye_inner_corner'][1],
        'left_eye_outer_corner_x': key_points['left_eye_outer_corner'][0],
        'left_eye_outer_corner_y': key_points['left_eye_outer_corner'][1],
        'right_eye_inner_corner_x': key_points['right_eye_inner_corner'][0],
        'right_eye_inner_corner_y': key_points['right_eye_inner_corner'][1],
        'right_eye_outer_corner_x': key_points['right_eye_outer_corner'][0],
        'right_eye_outer_corner_y': key_points['right_eye_outer_corner'][1],
        'left_eyebrow_inner_end_x': key_points['left_eyebrow_inner_end'][0],
        'left_eyebrow_inner_end_y': key_points['left_eyebrow_inner_end'][1],
        'left_eyebrow_outer_end_x': key_points['left_eyebrow_outer_end'][0],
        'left_eyebrow_outer_end_y': key_points['left_eyebrow_outer_end'][1],
        'right_eyebrow_inner_end_x': key_points['right_eyebrow_inner_end'][0],
        'right_eyebrow_inner_end_y': key_points['right_eyebrow_inner_end'][1],
        'right_eyebrow_outer_end_x': key_points['right_eyebrow_outer_end'][0],
        'right_eyebrow_outer_end_y': key_points['right_eyebrow_outer_end'][1],
        'nose_tip_x': key_points['nose_tip'][0],
        'nose_tip_y': key_points['nose_tip'][1],
        'mouth_left_corner_x': key_points['mouth_left_corner'][0],
        'mouth_left_corner_y': key_points['mouth_left_corner'][1],
        'mouth_right_corner_x': key_points['mouth_right_corner'][0],
        'mouth_right_corner_y': key_points['mouth_right_corner'][1],
        'mouth_center_top_lip_x': key_points['mouth_center_top_lip'][0],
        'mouth_center_top_lip_y': key_points['mouth_center_top_lip'][1],
        'mouth_center_bottom_lip_x': key_points['mouth_center_bottom_lip'][0],
        'mouth_center_bottom_lip_y': key_points['mouth_center_bottom_lip'][1],
        'mouth_center_upper_lip_x': key_points['mouth_center_upper_lip'][0],  # Coordenadas del labio superior
        'mouth_center_upper_lip_y': key_points['mouth_center_upper_lip'][1],  # Coordenadas del labio superior
        'mouth_center_lower_lip_x': key_points['mouth_center_lower_lip'][0],  # Coordenadas del labio inferior
        'mouth_center_lower_lip_y': key_points['mouth_center_lower_lip'][1],  # Coordenadas del labio inferior
        'image': img_str
    }
    return coordinates, key_points

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    img_data = None
    if 'file' not in request.files:
        print("No se encontró 'file' en request.files")
        return 'No se ha proporcionado ningún archivo.'

    file = request.files['file']

    if file.filename == '':
        print("El archivo no tiene un nombre")
        return 'No se ha seleccionado ningún archivo.'

    try:
        img = Image.open(file)
        img = img.convert('RGB')  # Mantener la imagen en color
        img = np.array(img)  # Convertir la imagen a un array de NumPy
        img_str = img.flatten().tolist()  # Convertir la matriz a lista plana

        # Obtener la matriz de puntos clave de la imagen
        key_points, key_points2 = puntos(img, img_str)

        # Convertir el diccionario a una cadena JSON
        key_points_json = json.dumps(key_points)

        # Llamar a procesar.py y pasarle los datos JSON en el argumento
        result = subprocess.run(
            ['python3', 'procesar.py'],
            input=key_points_json,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Error en la ejecución de procesar.py:", result.stderr)
            return f"Error en la ejecución: {result.stderr}"

        # Mostrar salida de procesar.py en la consola
        datos = result.stdout
        print("resultado:", result.stdout)

        # INICIO GRAFICAR
        # Dibujar las cruces rojas en la imagen
        for point in key_points2.values():
            cv2.drawMarker(img, point, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, 
                        markerSize=15, thickness=3, line_type=cv2.LINE_AA)  # Aumentar tamaño y grosor

        # Convertir la imagen a PIL para graficar
        img_pil = Image.fromarray(img)

        # Crear la figura de Matplotlib
        buf = io.BytesIO()
        plt.imshow(img_pil)  # Mantener la imagen en color
        plt.axis('off')  # Ocultar los ejes
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        # Convertir la imagen a base64 para poder mostrarla en el HTML
        img_data = base64.b64encode(buf.getvalue()).decode('ascii')
        # FIN GRAFICAR

    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
        return f"Error al procesar la imagen: {str(e)}"

    # Renderizar resultado.html y pasar la imagen generada
    return render_template('resultado.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
