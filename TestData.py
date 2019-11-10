import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set i to test entry in dataset
testEntry = 9

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.models.load_model("model.h5")

prediction = model.predict(test_images)

plt.grid(False)
plt.imshow(test_images[testEntry], cmap=plt.cm.binary)
plt.xlabel("Actual: " + class_names[test_labels[testEntry]])
plt.title("Prediction " + class_names[np.argmax(prediction[testEntry])])
plt.show()