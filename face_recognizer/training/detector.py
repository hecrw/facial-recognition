from pathlib import Path
import face_recognition as fr
import pickle
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

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
    
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    for box, encoding in zip(input_face_location, input_face_encod):
        name = _recognize_face(encoding, loaded_encodings)
        if not name:
            name = "unknown"
        display_face(draw, box, name)
    del draw
    pillow_image.show()
        
def _recognize_face(encoding, loaded_encodings):
    bool_matches = fr.compare_faces(loaded_encodings["encodings"], encoding)
    votes = Counter(name for match, name in zip(bool_matches,loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]
    
def display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline="blue", width=4)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    font = ImageFont.truetype("training/pixel.ttf",24)
    draw.text((text_left, text_top),name,fill="white",font=font)
    
# def validate():
#     for file in Path("validation").rglob("*"):
#         if file.is_file():
#             recognize_faces(str(file.absolute()))
    
# encode_faces()
# validate()
