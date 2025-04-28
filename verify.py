import dlib
import cv2
import mysql.connector
import json
import numpy as np
import sys
import time
from scipy.spatial import distance

def verify_face(image_path):
    # Load models
    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
    sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print("NO_IMAGE_LOADED")
        sys.exit(1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize if too large
    if rgb_image.shape[0] > 800:
        scale = 800 / rgb_image.shape[0]
        rgb_image = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)

    # Detect faces
    start = time.time()
    faces = detector(rgb_image, 1)
    if len(faces) == 0 or (time.time() - start) > 0.2:
        cnn_faces = cnn_detector(rgb_image, 1)
        if len(cnn_faces) == 0:
            print("NO_FACE_DETECTED")
            sys.exit(1)
        face = cnn_faces[0].rect
    else:
        face = faces[0]

    shape = sp(rgb_image, face)
    face_descriptor = face_encoder.compute_face_descriptor(rgb_image, shape)
    face_vector = np.array(face_descriptor)

    # Connect to database
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_recognition"
        )

        cursor = connection.cursor()
        cursor.execute("SELECT user_id, encoding FROM face_encodings")
        rows = cursor.fetchall()

        min_dist = float('inf')
        matched_user_id = None

        for row in rows:
            stored_user_id = row[0]
            stored_vector = np.array(json.loads(row[1]))
            dist = distance.euclidean(face_vector, stored_vector)

            if dist < min_dist:
                min_dist = dist
                matched_user_id = stored_user_id

        if min_dist < 0.6:
            print(f"User {matched_user_id} authenticated successfully! (Distance: {min_dist:.4f})")
        else:
            print("Face not recognized!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        sys.exit(1)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_face.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    verify_face(image_path)

sys.exit(0)
