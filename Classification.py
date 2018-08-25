import tensorflow as tf
from tensorflow import keras    #"tf.keras" is a high level api to build and train model in TensorFlow
import numpy as np
import matplotlib.pyplot as plt

hello = tf.constant('hello, Tensorflow')
sess = tf.Session()
print(sess.run(hello))

print("TensorFlow Version: ", tf.__version__)  #To print the tensorflow version

fashion_mnist = keras.datasets.fashion_mnist    #here we are taking/importing fashion_mnist dataset from keras database.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  #here we are loading the dataset

#we could have also written "from keras.datasets import fashion_mnist" and then "fashion_mnist.load_data()" (same as up)


#Since the class names was not included with the dataset, store them here to use later when plotting the images.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("train_image shape: ", train_images.shape , "test_image shape: ", test_images.shape) #output says We will
# use 60,000 images to train the network. The images are 28x28 NumPy arrays, with pixel values ranging
# between 0 and 255. same for test dataset.

print("train_image length: ", len(train_images), "test_image length:", len(test_images))
print("training data labels: ", train_labels) #The labels are an array of integers, ranging from 0 to 9.
#These correspond to the class of clothing (which we made above as class_names) the image represents.

#Preprocessing of Data
plt.figure()
plt.imshow(train_images[0])  #we have our images in numpy.array. therefore imshow() is used to display it.
plt.colorbar()  #It’s helpful to have an idea of what value a color represents. We can do that by adding color bars.
plt.gca().grid(False)  #"gca" returns the axes associated with the figure. And grid is grid.
#plt.show()

train_images = train_images / 255
test_images = test_images / 255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])  #Disable xticks
    plt.yticks([])  #Disable yticks
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)  #cmap means color maps. set to binary means 2 colors.
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#Building the neural networks require configuring the layers and then compiling the model.
#Layers of neural network extract representations from the data fed into them.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#In Keras, you assemble layers to build models.
# The most common type of model is a stack of layers: the tf.keras.Sequential model.
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images
# from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
# This layer has no parameters to learn; it only reformats the data.
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
# These are densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
#  Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Once you defined the ,model it needs to be compiled (using compile()).
#The optimizer is the search technique used to update the weights in your model.
#Some popular optimizer are :
#1. SGD: stochastic gradient descent, with support for momentum.
#2. RMSprop: adaptive learning rate optimization method proposed by Geoff Hinton.
#3. Adam: Adaptive Moment Estimation (Adam) that also uses adaptive learning rates.
#Some common loss fuctions are:
#1. mse: for mean squared error.
#2. binary_crossentropy: for binary logarithmic loss (logloss).
#3. categorical_crossentropy: for multi-class logarithmic loss (logloss).
#Metrics are evaluated by the model during training. Only one metric is supported at the moment and that is accuracy.

model.fit(train_images, train_labels, epochs=2)
#Here we are training the model. As the model trains, the loss and accuracy metrics are displayed.
#Epoch is the number of times the model is exposed to the training data.
#Batch size is the number of training instances shown to the model before a weight update is performed.

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)
#Here we are comparing the model preformance of the model on the test dataset.

predictions = model.predict(test_images)
#With the model trained we can now make predictions on the test data.

print(predictions[0])
print(np.argmax(predictions[0]))  #As we have 10 neurons in the output layer we will take the maximum value.(predicted)
print(test_labels[0])  #Checking what was the actual value.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
plt.show()

