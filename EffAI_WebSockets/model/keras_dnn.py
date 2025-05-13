from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim):
    h1 = (input_dim + 1) // 2
    h2 = (h1 + 1) // 2
    h3 = (h2 + 1) // 2

    model = Sequential()
    model.add(Dense(h1, input_dim=input_dim, activation='tanh'))
    model.add(Dense(h2, activation='tanh'))
    model.add(Dense(h3, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model
