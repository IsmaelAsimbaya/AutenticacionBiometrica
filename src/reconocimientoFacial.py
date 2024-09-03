import math
import os
import os.path
import pickle
import base64
import io
import numpy as np
import requests
import face_recognition
import cv2
import imutils
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, UnidentifiedImageError
#ibreria para la prueba d vida anti-spoof
from silent_face_anti_spoofing.test import test

EXTENSIONES_PERMITIDAS = {'png', 'jpg', 'jpeg'}

def codificacion_rostro (rostro_array):

    # Se aplica el algoritmo HOG sobre la clasificacion de JV Haar Cascade Classifier para aumentar la precision en la deteccion.
    rostro_hog = face_recognition.face_locations(rostro_array)

    if len(rostro_hog) != 1:
        print("La imagen no es valida para entrenamiento: {}".format(
            "No se encontro una cara" if len(rostro_hog) < 1 else "Se encontro mas de una cara"))
        return  None
    else:
        # retornamos la codificacion de la cara actual al conjunto de entrenamiento
        print('codificacion realizada')
        return face_recognition.face_encodings(rostro_array, known_face_locations=rostro_hog)[0]
    
def captura_codificacion_rostros_video(persona_id, videourl):

    # almacenamos la ruta del directorio donde se encuentra el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # se crean las rutas para los directorios de train
    dataPath = os.path.join(script_dir, 'data', 'train')
    personaPath = os.path.join(dataPath, persona_id)

    if not os.path.exists(personaPath):
        print('Directorio train creado: ', personaPath)
        os.makedirs(personaPath)

    cap = cv2.VideoCapture(videourl)
    
    if not cap.isOpened():
        print("Error al abrir el archivo de video")
        return False

    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    margin = 120
    X_codificacionrostro = []
    y_personaid = []

    print('Iniciando captura')

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5,
                                             minSize=(200, 200),
                                             maxSize=(450, 450))
        
        for (x, y, w, h) in faces:
             
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(frame.shape[1], x + w + margin)
            y_end = min(frame.shape[0], y + h + margin)
        
            rostro_haar = auxFrame[y_start:y_end, x_start:x_end]
            rostro_haar = cv2.resize(rostro_haar, (150, 150), interpolation=cv2.INTER_CUBIC)

            rostro_codificado = codificacion_rostro(rostro_haar)
            if rostro_codificado is not None:
                X_codificacionrostro.append(rostro_codificado)
                y_personaid.append(persona_id)

            count = count + 1

        if count >= 300:
            print('Captura terminada')
            break

    data_rostro_persona = {'codificaciones':X_codificacionrostro, 'persona_id':y_personaid}
    path_filname_data = os.path.join(personaPath, persona_id + '.pkl')
    with open(path_filname_data, 'wb') as file:
        pickle.dump(data_rostro_persona, file)
    print('Datos guardados en {}'.format(path_filname_data))
    cap.release()
    return True

def recolectar_codificaciones(codificaciones_path):

    X_codificacionrostro_all = []
    y_personaid_all = []

    # Recorrer todas las carpetas y subcarpetas dentro del directorio principal
    for root, dirs, files in os.walk(codificaciones_path):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # Combinar las codificaciones e IDs
                    X_codificacionrostro_all.extend(data['codificaciones'])
                    y_personaid_all.extend(data['persona_id'])
    print('Datos en {} combinados'.format(codificaciones_path))

    return {'codificaciones': X_codificacionrostro_all, 'persona_id': y_personaid_all}

# entrena un clasificadro k vecinos mas crecanos para reconocimiento facial
def entrenamiento_knn(data_path , model_save_path=None, n_vecinos=None, km_algortm='ball_tree'):
    # model_save_path: directorio para guardar el modelo en el disco
    # n_neighbors: la estructura de datos subyacente para admitir knn.default es ball_tree

    combined_data = recolectar_codificaciones(data_path)

    if not combined_data['codificaciones'] and not combined_data['persona_id']:
        print('sin datos para entrenamiento')
        return False
    
    X_codificacionrostro = combined_data['codificaciones']
    y_personaid = combined_data['persona_id']

    # determinamos cuantos vecinos usar para el clasificador KNN
    if n_vecinos is None:
        print("Eligiendo n_vecinos automaticamnete:", n_vecinos)
        n_vecinos = int(round(math.sqrt(len(X_codificacionrostro))))

    # Crearmos y entrenamos el clasificador KNN
    knn_clsf = neighbors.KNeighborsClassifier(n_neighbors=n_vecinos, algorithm=km_algortm, weights='distance')
    knn_clsf.fit(X_codificacionrostro, y_personaid)

    # guardamos el clasificador KNN entrenado
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clsf, f)
    print('modelo entrenado')
    return True

def verificar_imagen_numpy(imagen_np):
    # Verificar que la imagen tiene 2 o 3 dimensiones (grayscale o RGB)
    if imagen_np.ndim not in [2, 3]:
        return False
    # Verificar que los valores de los píxeles están dentro del rango válido
    if not (0 <= imagen_np).all() and (imagen_np <= 255).all():
        return False
    # Verificar que el tipo de datos sea uint8
    if imagen_np.dtype != np.uint8:
        return False
    return True
    
def redimension(imagen_original, nuevo_ancho = 500):
    ancho_original, alto_original = imagen_original.size
    nuevo_alto = int(alto_original * nuevo_ancho / ancho_original)
    return np.array(imagen_original.resize((nuevo_ancho, nuevo_alto)))

def carga_imagen_base64(img_base64):
    try:
        imagen_decod = Image.open(io.BytesIO(base64.b64decode(img_base64)))
        imagen_np = redimension(imagen_decod)
        if verificar_imagen_numpy(imagen_np):
            return imagen_np
        else:
            raise ValueError("Imagen corrupta o no válida.")
    except Exception as e:
        print(f"Error al cargar la imagen desde Base64: {e}")
        return None

def carga_imagen_url(img_url):
    try:
        imagen_load = Image.open(io.BytesIO(requests.get(img_url).content))
        imagen_np = redimension(imagen_load)
        if verificar_imagen_numpy(imagen_np):
            return imagen_np
        else:
            raise ValueError("Imagen corrupta o no válida.")
    except Exception as e:
        print(f"Error al cargar la imagen desde URL: {e}")
        return None

def anti_spoof(img_nparray):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spoof = test(img_nparray,
                 model_dir=os.path.join(script_dir, 'silent_face_anti_spoofing', 'resources','anti_spoof_models'),
                 device_id=0)
    if spoof != 1:
        raise Exception("Intento de suplantacion de identidad")

# reconoce una imagen dadda usando un clasificadro KNN entrenado 0.47
def predict_img(auth_img_nparray, modelknn_path=None, distance_threshold=0.47):
    # auth_img_nparray: nparray de la imagen a procesar
    # modelknn_path: camino a un clasificador knn pre entrnado. si no se especifica, model_save_path debe ser knn_clsf.
    # distance_threshold: Umbral de distancia para la clasificación de rostros. cuanto más grande es, más posibilidades
    #                     de clasificar erróneamente a una persona desconocida como conocida.
    # retornamos una lista de nombres y locaciones de caras para las caras reconocidas en la imagen
    # para las caras no reconocidas se retorna el nombre de "unknown"

    if modelknn_path is None:
        raise Exception("Debe proporcionar el clasificador knn")

    with open(modelknn_path, 'rb') as f:
        knn_clsf = pickle.load(f)

    # cargamos la imagen y encontramos la posicion de las caras
    X_face_locations = face_recognition.face_locations(auth_img_nparray)

    # si no encontramos caras en la imagen, retornamos una lista vacia
    if len(X_face_locations) == 0:
        return []
    
    # encontramos las codificaciones para las caras en la imagen de test
    faces_encodings = face_recognition.face_encodings(auth_img_nparray, known_face_locations=X_face_locations)
    # usamos el modelo KNN para encontrar las mejores coincidencias para la iamgen de test
    closest_distances = knn_clsf.kneighbors(faces_encodings, n_neighbors=1)
    print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(are_matches)

    # predecimos las clases y removemos las clasificaiones que no estan en el humbral
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clsf.predict(faces_encodings), X_face_locations, are_matches)]

def face_rec(image_source, id_predict, source_type = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Iniciando reconocimiento")
    prediccion = None

    if source_type == 'url':
        prediccion = predict_img(carga_imagen_url(image_source), 
                                modelknn_path=os.path.join(script_dir,'knn_modelo_entrenado.clf'))
    elif source_type == 'base64':
        prediccion = predict_img(carga_imagen_base64(image_source), 
                                modelknn_path=os.path.join(script_dir,'knn_modelo_entrenado.clf'))
    else:
        print("Tipo de origen no soportado. Usa 'url' o 'base64'.")
        return None

    for name, (top, right, bottom, left) in prediccion:
        print("- Found {} at ({}, {})".format(name, left, top))
        if name == id_predict:
            return True
    return False

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # urlvideofirebase = ('https://firebasestorage.googleapis.com/v0/b/srab-d7052.appspot.com/o/videos%2Fisma.mp4?alt=media&token=86c67320-df84-4824-98b2-8ec6a176f6b1')
    # captura_codificacion_rostros_video('1774', urlvideofirebase)

    # print("Entrenando clasificador KNN...")
    # entrenamiento_knn(data_path = os.path.join(script_dir,'data','train'),
    #               model_save_path = os.path.join(script_dir,'knn_modelo_entrenado.clf'),
    #               n_vecinos = 5)
    # print("Entrenamiento Completado!")

    # imagenurl = 'https://firebasestorage.googleapis.com/v0/b/srab-d7052.appspot.com/o/validaciones%2F1774%2Fimagen_2024-08-19_032513648.png?alt=media&token=1af12bd3-0435-45da-b33b-22d0bb758b2b'

    # test = face_rec(image_source=imagenurl,
    #          id_predict='1774',
    #          source_type='url')
    # print(test)