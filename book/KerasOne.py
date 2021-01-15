import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(4, input_shape=(2,)),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['accuracy'])

training_number = 100
training_data = np.random.random((training_number, 2))
labels = np.array([(1 if data[0] < data[1] else 0) for data in training_data])
print(type(training_data))
print(type(labels))
model.fit(training_data, labels, epochs=20, batch_size=32)

test_number = 100
test_data = np.random.random((test_number, 2))
expected = [(1 if data[0] < data[1] else 0) for data in test_data]
error = 0
for i in range(0, test_number):
    data = test_data[i].reshape(1, 2)
    pred = 0 if model.predict(data) < 0.5 else 1

    if (pred != expected[i]):
        error += 1

print("total errors:{},accuracy:{}".format(error, 1.0 - error / test_number))

# 画图
# plot_model(model,to_file='KerasOne.png',show_shapes=True)
