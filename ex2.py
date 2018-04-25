import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

# Carga el modelo
model = inception_v3.InceptionV3()

model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

object_type_to_fake = 151

# Cargamos imagen
img = image.load_img("pelican.jpg", target_size=(300, 300))
original_image = image.img_to_array(img)

# reescalamos los valores de la imagen en el array están entre 1 y -1
original_image /= 255.
original_image -= 0.5
original_image *= 2.

original_image = np.expand_dims(original_image, axis=0)

# cambios máximo por arriba y por abajo
max_change_above = original_image + 0.01
max_change_below = original_image - 0.01

# Create a copy of the input image to hack on
final = np.copy(original_image)

learning_rate = 0.1
cost_function = model_output_layer[0, object_type_to_fake]
gradient_function = K.gradients(cost_function, model_input_layer)[0]
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])
cost = 0.00

# Bucle para calcular la aimgen
while cost < 0.80:
    #print(grab_cost_and_gradients_from_model([final,0]))
    cost, gradients = grab_cost_and_gradients_from_model([final, 0])
    final += gradients * learning_rate
    final = np.clip(final, max_change_below, max_change_above)
    final = np.clip(final, -1.0, 1.0)

    #print(grab_cost_and_gradients_from_model)
    print("Predicción del modelo (chihuahua): {:.8}%".format(cost * 100))

# deescalado
img = final[0]
img /= 2.
img += 0.5
img *= 255.

im = Image.fromarray(img.astype(np.uint8))
im.save("result.jpg")