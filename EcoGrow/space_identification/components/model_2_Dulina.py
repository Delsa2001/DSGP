
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.utils import img_to_array, load_img # type: ignore
import numpy as np

model = load_model(r"EcoGrow\space_identification\Models\resnet50_balcony_model.h5")

def preprocess(image_path):
    image = load_img(image_path, target_size=(128,128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def predict(image_path, model, threshold = 0.2):
    preprocessed_image = preprocess(image_path)
    prediction = model.predict(preprocessed_image)[0][0]


    if prediction < (0.5 - 0.49) :
        predicted_class =  "This is a Balcony space"
    elif prediction > (0.5 + threshold):
        predicted_class =  "This is Indoor space"
    else:
        predicted_class = "Not a Balcony or Indoor Space"
 

    confidence_score = float(np.max(prediction)) 
    print(confidence_score)


    return predicted_class, confidence_score





