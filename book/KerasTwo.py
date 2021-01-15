from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model

model1 = Sequential()
model1.add(Dense(32, input_shape=(32,), activation='sigmoid'))
plot_model(model1, to_file='m1.png', show_shapes=True)

a = Input(shape=(32,))
b = Dense(1, activation='sigmoid')(a)
model2 = Model(inputs=a, outputs=b)
plot_model(model2, to_file='m2.png', show_shapes=True)

input1 = Input(shape=(2,))
h1 = Dense(3, activation='sigmoid')(input1)
output1 = Dense(1, activation='sigmoid')(h1)

input2 = Input(shape=(3,))
new_input = concatenate([output1, input2])
h2 = Dense(4, activation='sigmoid')(new_input)
output2 = Dense(2, activation='sigmoid')(h2)
model3 = Model(inputs=[input1, input2], outputs=[output1, output2])
plot_model(model3, to_file='m3.png', show_shapes=True)
