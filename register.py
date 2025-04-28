import dlib
import cv2
import mysql.connector
import json
import numpy as np
import sys
import time

def register_face(user_id, image_path):
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

    # Resize if image too large
    if rgb_image.shape[0] > 800:
        scale = 800 / rgb_image.shape[0]
        rgb_image = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)

    # Try frontal detector first
    start = time.time()
    faces = detector(rgb_image, 1)
    if len(faces) == 0 or (time.time() - start) > 0.2:  # fallback to CNN if no faces or timeout > 200ms
        cnn_faces = cnn_detector(rgb_image, 1)
        if len(cnn_faces) == 0:
            print("NO_FACE_DETECTED")
            sys.exit(1)
        face = cnn_faces[0].rect
    else:
        face = faces[0]

    # Get facial landmarks and encoding
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

        encoding_json = json.dumps(face_vector.tolist())

        # Check if user exists
        cursor.execute("SELECT COUNT(*) FROM face_encodings WHERE user_id = %s", (user_id,))
        (count,) = cursor.fetchone()

        if count > 0:
            cursor.execute(
                "UPDATE face_encodings SET encoding = %s WHERE user_id = %s",
                (encoding_json, user_id)
            )
            print(f"UPDATED user {user_id}")
        else:
            cursor.execute(
                "INSERT INTO face_encodings (user_id, encoding) VALUES (%s, %s)",
                (user_id, encoding_json)
            )
            print(f"INSERTED user {user_id}")

        connection.commit()

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        sys.exit(1)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python register_face.py <user_id> <image_path>")
        sys.exit(1)

    user_id = sys.argv[1]
    image_path = sys.argv[2]
    register_face(user_id, image_path)

    sys.exit(0)
