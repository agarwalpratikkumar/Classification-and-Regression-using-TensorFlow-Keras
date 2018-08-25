import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#Shuffle the training set
#print(train_labels.shape)
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))
print("Testing set:  {}".format(test_data.shape))
#It is given that these features are stored in different values.
#So this calls for normalizing all the values into similar scale.

print(train_data[0])

#All the features
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

#Create a dataframe to represent in table format.
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())
#We can see that values are in different ranges. Hence Normalize.

#given that the labels are the house prices in thousands of dollars.
print(train_labels[0:10])

#Normalization:
#For each feature, subtract the mean of the feature and divide by the standard deviation.

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std
#Training data after normalization
print(train_data[0])

#Create the model
#we will make a sequential model with 2 fully connected hidden layers (64 neurons )and a output layer
# with 1 neuron whch will return a single continuous value.
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])
    return model
#Mean Squared Error (MSE) is a common loss function used for regression problems
#Similarly, evaluation metrics used for regression differ from classification.
#A common regression metric is Mean Absolute Error (MAE).
#A metric is a function that is used to judge the performance of your model.
#A metric function is similar to a loss function, except that the results from evaluating a metric
#are not used when training the model.

model = build_model()
model.summary() #We can see the number of weights which needs to be trained.

#Train the model
EPOCHS = 500

# We are displaying training progress by printing a single dot for each completed epoch
class printDots(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[printDots()])

#We can visualize the model's training progress using the stats stored in the history object.
# We want to use this data to determine how long to train before the model stops making progress.
#By setting 'verbose' 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
#verbose=0 will show you nothing (silent); verbose=1 will show you an animated progress bar like [=====] and
#verbose 2 will mention the number of epoch like : Epoch 1/10


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    #plt.show()

plot_history(history)
#We can see from the graph that validation loss is not decreasing after a while.
#So we need to stop at that point to prevent overfitting.

#So we need to update our model.fit

model = build_model()

#Stop training when a monitored quantity has stopped improving.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
#The patience parameter is the amount of epochs to check for improvement.
#'monitor' is the quantity to be monitored
#'patience' is the number of epochs with no improvement after which training will be stopped.


history = model.fit(train_data, train_labels, epochs=EPOCHS,
          validation_split=0.2, verbose=0,
          callbacks=[early_stop, printDots()])
plot_history(history)

#Lets check the performance on test data.
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
#print(loss, mae)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

#Prediction
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

#To get an overoview of the error
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()




