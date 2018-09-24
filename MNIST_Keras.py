import tensorflow as tf
import cv2
#images of digits 28*28
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()



for img in x_train:
    img = cv2.Canny(img,100,200)
    
x_train = tf.keras.utils.normalize(x_train,axis=1)    

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show

for img in x_test:
    img = cv2.Canny(img,100,200)
    
x_test = tf.keras.utils.normalize(x_test,axis=1)    

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

val_loss, val_accu = model.evaluate(x_test,y_test)

print(val_accu,val_loss)

model.save(filepath= /home/pinakpani/Machine_Learning_Projects/MNIST-with-keras)