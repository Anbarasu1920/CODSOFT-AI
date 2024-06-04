import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def extract_face_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    face_features = []
    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_features.append(np.array(face_descriptor))

    return face_features, faces


known_faces = []
known_names = []
image_paths = ['person1.jpg', 'person2.jpg']
names = ['Person 1', 'Person 2']

for image_path, name in zip(image_paths, names):
    features, _ = extract_face_features(image_path)
    known_faces.append(features[0])
    known_names.append(name)


def recognize_faces(image_path):
    image = cv2.imread(image_path)
    face_features, faces = extract_face_features(image_path)
    for face, face_feature in zip(faces, face_features):
        distances = np.linalg.norm(known_faces - face_feature, axis=1)
        min_distance = np.argmin(distances)
        if distances[min_distance] < 0.6:
            name = known_names[min_distance]
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(image, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_image_path = 'test_image.jpg'
recognize_faces(test_image_path)
