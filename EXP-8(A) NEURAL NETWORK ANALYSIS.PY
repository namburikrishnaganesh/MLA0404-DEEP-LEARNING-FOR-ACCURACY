import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, size=(10,))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(10, input_dim=5, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
learning_rate = 0.01
sgd = SGD(learning_rate=learning_rate)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
