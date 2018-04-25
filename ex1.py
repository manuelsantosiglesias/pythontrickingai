import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

# Carga el modelo
model = inception_v3.InceptionV3()

# Cargamos la imagen a un array
img = [image.load_img("pelican.jpg", target_size=(300, 300)),image.load_img("pelican2.jpg", target_size=(300, 300)),image.load_img("pelican3.jpg", target_size=(300, 300)),image.load_img("result.jpg", target_size=(300, 300))]

for f in img:
    input_image = image.img_to_array(f)
    # Escalamos la imagen
    input_image /= 255.
    input_image -= 0.5
    input_image *= 2.

    # Ponemos la imagen como keras espera
    input_image = np.expand_dims(input_image, axis=0)

    # Ejecutamos la predicción
    predictions = model.predict(input_image)

    # Imprimimos en pantalla la predicción
    predicted_classes = inception_v3.decode_predictions(predictions, top=5)
    print(predicted_classes)
    imagenet_id, name, confidence = predicted_classes[0][0]
    print("!Es {} con {:.4}% de probabilidad!".format(name, confidence * 100))