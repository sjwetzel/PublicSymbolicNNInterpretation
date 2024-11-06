# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:03:18 2023

@author: sebas
"""


from __future__ import print_function

import tensorflow as tf

#tf.enable_eager_execution()


import tensorflow.keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Subtract,Input,Lambda, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D, LocallyConnected1D, Add
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers


import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

np.random.seed(seed=2)
features=3
noise=0.01
l2=0.0001

def probe_function(x):
    return np.exp(x[0]-x[1])-np.pi*x[2]
def target_function(x):
    if probe_function(x)*np.random.normal(1,noise)> 1:
        return 1
    else:
        return 0

def explicit_gradient(x):
    return [np.exp(x[0]-x[1])   ,  - np.exp(x[0]-x[1])   ,   -np.pi]

print('data generation')

x=np.random.rand(10000,features)*2-1
y=np.array([target_function(sample) for sample in x])

randomized_order=np.random.permutation(len(x))


x=x[randomized_order]
y=y[randomized_order]

x_train=x[:int(0.8*len(x))]
y_train=y[:int(0.8*len(x))]

x_test=x[int(0.8*len(x)):]
y_test=y[int(0.8*len(x)):]




batch_size = 100
epochs = 1#1000

input_layer=Input(shape=(x_train.shape[-1],),name='input_layer')
layer_a1=Dense(1000,activation='elu',name='layer_a1',kernel_regularizer=regularizers.L2(l2),bias_regularizer=regularizers.L2(l2))(input_layer)  ## does sigmoid give smoother gradients then relu?
layer_a1=Dropout(0.2)(layer_a1)
layer_a2=Dense(1000,activation='elu',name='layer_a2',kernel_regularizer=regularizers.L2(l2),bias_regularizer=regularizers.L2(l2))(layer_a1)
layer_a2=Dropout(0.2)(layer_a2)
output_latent=Dense(1,name='output')(layer_a2)
output=Activation(activation='sigmoid')(output_latent)

model = Model(inputs=input_layer, outputs=output)
model_latent = Model(inputs=input_layer, outputs=output_latent)

model.summary()

   

# Let's train the model 
model.compile(loss='binary_crossentropy',
              #optimizer=tensorflow.keras.optimizers.Adadelta(lr=0.0001),
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['acc'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=20, verbose=1,min_lr=0)


early_stop= EarlyStopping(monitor='loss', patience=40, verbose=1)

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=[reduce_lr,early_stop])


# Plot training & test loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



#### data analysis
## inverse sigmoid

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
    #return x
    return np.log(x/(1-x))


predictions=model.predict(x)
predictions_latent=model_latent.predict(x)


y_analysis = [inverse_sigmoid(prediction) for prediction in predictions]




y_probe = [probe_function(sample) for sample in x]


plt.scatter(predictions_latent,y_probe,label="latent model")
#plt.scatter(y_analysis,y_probe,label="inverse sigmoid")

# plt.xlim(-5, 5)
# plt.ylim(-0.1, 0.1)

plt.title('prediction correlation with true function')
plt.ylabel('true function')
plt.xlabel('model prediction')
plt.legend(loc='upper left')
plt.show()


x_unsure=x[np.where((predictions > 0.0001) & (predictions < 0.9999))[0]]


## neural net gradients

data_set=x_unsure


input_variable = tf.Variable(data_set, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(input_variable)
    preds = model_latent(input_variable)

nn_gradients = tape.gradient(preds, input_variable)

## explicit gradients


explicit_gradients = np.array([explicit_gradient(sample) for sample in data_set])

def normalize(x):
    norm=np.sqrt(np.sum(x**2))
    return x/norm

normalized_nn_gradients=np.array([normalize(sample) for sample in nn_gradients])
normalized_explicit_gradients=np.array([normalize(sample) for sample in explicit_gradients])


dot_products=[np.dot(normalized_explicit_gradients[i],normalized_nn_gradients[i]) for i in range(len(normalized_explicit_gradients))]
MSE=[np.average((normalized_explicit_gradients[i]-normalized_nn_gradients[i])**2) for i in range(len(normalized_explicit_gradients))]


print("average dot produt between grads of nn and true function: ",np.average(dot_products))
print("median dot produt between grads of nn and true function: ",np.median(dot_products))

print("average MSE between grads of nn and true function: ",np.average(MSE))
print("median MSE produt between grads of nn and true function: ",np.median(MSE))

print("length x_unsure", len(x_unsure))

np.savetxt('X.out', x, delimiter=',')
np.savetxt('Y.out', y, delimiter=',')