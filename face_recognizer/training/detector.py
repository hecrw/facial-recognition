from pathlib import Path
import face_recognition as fr
import pickle
from collections import Counter
import cv2 as cv
import threading as thread

location = "output/encodings.pkl"

def encode_faces(model="hog"):
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = fr.load_image_file(filepath)

        face_location = fr.face_locations(image, model=model)
        face_enco = fr.face_encodings(image, face_location)
        for e in face_enco:
            names.append(name)
            encodings.append(e)
    name_encodings = {"names": names, "encodings": encodings}
    with open(location, "wb") as file:
        pickle.dump(name_encodings, file)

def recognize_faces(image, model="hog"):
    with open(location, "rb") as file:
        loaded_encodings = pickle.load(file)
    input_image = image
    input_face_location = fr.face_locations(input_image, model=model)
    input_face_encod = fr.face_encodings(input_image, input_face_location)

    for box, encoding in zip(input_face_location, input_face_encod):
        name = _recognize_face(encoding, loaded_encodings)
        if not name:
            name = "unknown"
        top, right, bottom, left = box
        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv.imshow("Face Recognition", image)

def _recognize_face(encoding, loaded_encodings):
    bool_matches = fr.compare_faces(loaded_encodings["encodings"], encoding)
    votes = Counter(name for match, name in zip(bool_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]

# encode_faces()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480), cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    counter += 1
    if counter % 30 == 0:
        recognize_faces(frame)
    
    key = cv.waitKey(1)
    
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
