import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
datasets=tf.keras.datasets
layers=keras.layers
models=tf.keras.models
l2=tf.keras.regularizers.l2
(train, trainl), (test, testl) = datasets.cifar10.load_data()
print(testl)
train=train.astype("float32")/255
test=test.astype("float32")/255

model=keras.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32,3,padding="same", activation=tf.nn.relu,kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,3, activation=tf.nn.relu,kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    # layers.Conv2D(32,3, activation=tf.nn.relu),
    layers.Flatten(),
    # layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dropout(0.3),
    layers.Dense(10, activation=tf.nn.softmax)
])
# print(model.summary())
model.compile(optimizer=tf.optimizers.Adam(9e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train,trainl,batch_size=64,epochs=10)
testloss=model.evaluate(test,testl,batch_size=64)
pred=model.predict(test)
n=10000
for i in range(n):
    print(pred[i])
    print(testl[i])
    plt.imshow(test[i, :])
    plt.show()
    print(np.where(pred[i]==max(pred[i]))[0])
    a=input("enter")
    if True:
        continue