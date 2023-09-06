from pathlib import Path
import face_recognition as fr
import pickle
from collections import Counter

location = "output/encodings.pkl"
def encode_faces(model = "hog"):
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

def recognize_faces(image_location, model = "hog"):
    with open(location, "rb") as file:
        loaded_encodings = pickle.load(file)
    input_image = fr.load_image_file(image_location)
    input_face_location = fr.face_locations(input_image, model=model)
    input_face_encod = fr.face_encodings(input_image, input_face_location)
    for box, encoding in zip(input_face_location, input_face_encod):
        name = _recognize_face(encoding, loaded_encodings)
        if not name:
            name = "unknown"
        
def _recognize_face(encoding, loaded_encodings):
    bool_matches = fr.compare_faces(loaded_encodings["encodings"], encoding)
# encode_faces()
