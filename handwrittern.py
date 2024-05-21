import numpy as np 
import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt 


mist = tf.keras.datasets.mnist 
(x_train,y_train),(x_test,y_test) = mist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 
model.fit(x_train,y_train,epochs=5)
loss , accuracy = model.evaluate(x_test,y_test) 
print(loss)
print(accuracy)


img = cv2.imread('./test.png')[:,:,0]
img = np.invert(np.array([img])) 
prediction = model.predict(img)
print(f"The digit is probably a {np.argmax(prediction)}")
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()