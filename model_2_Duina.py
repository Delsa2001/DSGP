
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np

model = load_model("balcony_identification_model.h5")

def preprocess(image_path):
    image = load_img(image_path, target_size=(128,128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def predict(image_path, model):
    preprocessed_image = preprocess(image_path)
    prediction = model.predict(preprocessed_image)

    if prediction[0]>0.5:
        return "This is a Indoor space"
    else:
        return "This is Balcony space"
    




