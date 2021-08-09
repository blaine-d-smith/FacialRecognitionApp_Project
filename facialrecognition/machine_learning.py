from django.conf import settings
import os
import numpy as np
import cv2
import pickle

# Face Detection Model
face_detector_model = cv2.dnn.readNetFromCaffe('FacialRecognitionApp/static/models/deploy.txt',
                                               'FacialRecognitionApp/static/models/face_detector_model.caffemodel')
# Feature Extraction Model
face_feature_model = cv2.dnn.readNetFromTorch('FacialRecognitionApp/static/models/face_feature_model.t7')

# Face Recognition Model
face_recognition_model = pickle.load(open('FacialRecognitionApp/static/models/machinelearning_face_recognition_model.pkl', mode='rb'))

# Emotion Recognition Model
emotion_recognition_model = pickle.load(open('FacialRecognitionApp/static/models/machinelearning_emotion_recognition_model.pkl', mode='rb'))

# Pipeline Model
def pipeline_model(path):
    img = cv2.imread(path)
    image = img.copy()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)

    # Set Input
    face_detector_model.setInput(blob)
    detections = face_detector_model.forward()

    # Results by Machine
    machine_results = dict(face_detect_score=[],
                           face_name=[],
                           face_name_score=[],
                           emotion_name=[],
                           emotion_name_score=[],
                           count=[])

    count = 1
    if len(detections) > 0:
        for i, confidence in enumerate(detections[0, 0, :, 2]):
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startx, starty, endx, endy) = box.astype('int')

                # Bounding Box
                cv2.rectangle(image, (startx, starty), (endx, endy), (242, 74, 0), 2)

                # Feature Extraction
                face_roi = img[starty:endy, startx:endx]
                face_blob = cv2.dnn.blobFromImage(face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)

                # Set Input
                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()

                # Predict Face
                face_name = face_recognition_model.predict(vectors)[0]
                face_score = face_recognition_model.predict_proba(vectors).max()
                # print(face_name, face_score)

                # Predict Emotion
                emotion_name = emotion_recognition_model.predict(vectors)[0]
                emotion_score = emotion_recognition_model.predict_proba(vectors).max()
                # print(emotion_name, emotion_score)

                # Result Images
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'outputs/result.jpg'), image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'outputs/roi_{}.jpg'.format(count)), face_roi)

                # Machine Results
                machine_results['count'].append(count)
                machine_results['face_detect_score'].append(confidence)
                machine_results['face_name'].append(face_name)
                machine_results['face_name_score'].append(("{:.2f}".format(face_score*100)))
                machine_results['emotion_name'].append(emotion_name)
                machine_results['emotion_name_score'].append(("{:.2f}".format(emotion_score*100)))

                count += 1
    return machine_results
