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
features=2
noise=0.01
l2=0.0001

def probe_function(x):
    return x[0]**2 +2* x[1]**2
def target_function(x):
    if probe_function(x)*np.random.normal(1,noise)> 1:
        return 1
    else:
        return 0

def explicit_gradient(x):
    return [2*x[0],4*x[1]]

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


##plot data
plt.figure(dpi=120)
x_0=x[np.where(y > 0.5)[0]]
x_1=x[np.where(y < 0.5)[0]]


plt.scatter(x_0[:,0],x_0[:,1], label='Class y=0', alpha=0.4, s=6, c='orange')
plt.scatter(x_1[:,0],x_1[:,1], label='Class y=1', alpha=0.4, s=6, c='cyan')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Scatter Plot of Data with 2 Features', fontsize=14)
# Plot contour line for the function x1**2 + 2*x2**2 = 1
x1_contour = np.linspace(-1, 1, 400)
x2_positive = np.sqrt(0.5 - 0.5 * x1_contour**2)
x2_negative = -np.sqrt(0.5 - 0.5 * x1_contour**2)

x2_contour = np.linspace(-1, 1, 400)
x1_contour, x2_contour = np.meshgrid(x1_contour, x2_contour)
contour_line = x1_contour**2 + 2 * x2_contour**2 - 1

plt.contour(x1_contour, x2_contour, contour_line, levels=[0], colors='red', linewidths=0.6)

# Add description
plt.text(0.32, 0.22, 'Decision Boundary:\n'+r' $g(x)=x_1^2 + 2x_2^2 = 1$', color='red', fontsize=12, ha='center')
plt.text(-0.75, 0.7, 'Normalized\nGradients', color='black', fontsize=12, ha='center')

# Choose a subset of points on the contour line
selected_indices = np.arange(0, 400, 51)


x1_contour = np.linspace(-1, 1, 400)

# Compute gradient at selected contour line points
grad_x1 = 2 * x1_contour[selected_indices]
grad_x2 = 4 * x2_positive[selected_indices]

# Normalize the gradient to get unit vectors
norm = np.sqrt(grad_x1**2 + grad_x2**2)
unit_vectors_x1 = grad_x1 / norm
unit_vectors_x2 = grad_x2 / norm

# Plot vectors starting at the contour line
scale_factor = 0.2
plt.quiver(x1_contour[selected_indices], x2_positive[selected_indices],
           scale_factor * unit_vectors_x1, scale_factor * unit_vectors_x2,
           angles='xy', scale_units='xy', color='black', scale=1, width=0.005)

# Compute gradient at selected contour line points
grad_x1 = 2 * x1_contour[selected_indices]
grad_x2 = 4 * x2_negative[selected_indices]

# Normalize the gradient to get unit vectors
norm = np.sqrt(grad_x1**2 + grad_x2**2)
unit_vectors_x1 = grad_x1 / norm
unit_vectors_x2 = grad_x2 / norm

# Plot vectors starting at the contour line
scale_factor = 0.2
plt.quiver(x1_contour[selected_indices], x2_negative[selected_indices],
           scale_factor * unit_vectors_x1, scale_factor * unit_vectors_x2,
           angles='xy', scale_units='xy', color='black', scale=1, width=0.005)
plt.show()

batch_size = 100
epochs = 1000#1000

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


plt.scatter(y_probe,predictions_latent,label="latent model")
plt.scatter(y_probe,predictions,label="full model")

#plt.scatter(y_analysis,y_probe,label="inverse sigmoid")

# plt.xlim(-5, 5)
# plt.ylim(-0.1, 0.1)

plt.title('prediction correlation with true function')
plt.ylabel('true function')
plt.xlabel('model prediction')
plt.legend(loc='upper left')
plt.show()



# Generate two different datasets
x_range = y_probe
data1 = predictions
data2 = predictions_latent

# Create scatter plot
fig, ax1 = plt.subplots(figsize=(8, 8),dpi=120)

# Plot dataset 1 on the left y-axis
ax1.scatter(x_range, data1, label='Neural Network F Prediction', alpha=0.4, s=6, c='blue')
ax1.set_xlabel('$g(x)=x_1^2+2x_2^2$',fontsize=18)
ax1.set_xticklabels([-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0],fontsize=18)

ax1.set_ylabel('Neural Network F Prediction', color='blue',fontsize=18)
ax1.tick_params('y', colors='blue',labelsize=18)


# Create a secondary y-axis for dataset 2 on the right
ax2 = ax1.twinx()
ax2.scatter(x_range, data2, label='Latent Model f Prediction', alpha=0.4, s=6, c='red')
ax2.set_ylabel('Latent Model f Prediction', color='red',fontsize=18)
ax2.tick_params('y', colors='red',labelsize=18)
# Set title
plt.title('Correlations Between Neural\nNetwork and True Function',fontsize=24)

# Display legend
#fig.legend(loc='lower right')

# Display the plot
plt.grid(True)
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

np.savetxt('circle_X.out', data_set, delimiter=',')
np.savetxt('circle_gradients.out', normalized_nn_gradients, delimiter=',')