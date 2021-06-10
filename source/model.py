# 1. Thêm các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Lambda
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# load du lieu
data = np.load("data.npy")
target = np.load("target.npy")

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

size = 224
# Xay dung model
model = Sequential()
model.add(Input(shape=(size, size, 1)))

model.add(Lambda(lambda x: x/255.0))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compile model: chi ro ham loss_function nao dc su dung
# va phuong thuc dung de toi uu ham loss function
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# luu weights
checkpoint = ModelCheckpoint("best_weights.hdf5", monitor="val_accuracy", verbose=1,
                             save_best_only=True, mode="max")

# thuc hien train model voi data
H = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16,
              epochs=15, callbacks=[checkpoint], verbose=1)

# # Ve do thi loss va accuracy cua tap train va tap val
fig = plt.figure()
numOfEpoch = 15
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

# Danh gia model voi du lieu test
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

# luu model file h5
model.save("predict-covid-19.h5")

