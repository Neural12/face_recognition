import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2


def new_learning_rate(epoch, lr):
    if epoch < 20:
        return 0.001
#     elif epoch < 10:
#         return 0.001
#     else:
#         return 0.0001

person1_folder = r"path"

person2_folder = r"path"

person3_folder = r"path"

person4_folder = r"path"

person5_folder = r"path"

person6_folder = r"path"

name_to_image = {}


def load_images(folder):
    global name_to_image
    files = os.listdir(folder)
    print("Number of files in", os.path.basename(folder), "folder:", len(files))
    for image_file in files:
        if image_file.endswith(".jpg"):
            name = os.path.basename(folder) + "_" + image_file.split(".")[0]
            name_to_image[name] = os.path.join(folder, image_file)


load_images(person1_folder)
load_images(person2_folder)
load_images(person3_folder)
load_images(person4_folder)
load_images(person5_folder)
load_images(person6_folder)

X = []
y = []

for name, image_path in name_to_image.items():
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)

    X.append(img_array)
    if name.startswith("person1"):
        y.append(0)  
    elif name.startswith("person2"):
        y.append(1)
    elif name.startswith("person3"):
        y.append(2)
    elif name.startswith("person4"):
        y.append(3)
    elif name.startswith("person5"):
        y.append(4)
    elif name.startswith("person6"):
        y.append(5)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.60),
    BatchNormalization(),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
            # a)
              loss='sparse_categorical_crossentropy',
            # b)
              metrics=['accuracy'])
# c)
lr_scheduler = LearningRateScheduler(new_learning_rate)



# d)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# e)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# f)
history = model.fit(X_train, y_train, epochs=15, batch_size=22, 
            validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])



# 1
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



def predict_person(image_path, model):
    img = cv2.imread(image_path)
    
    img = cv2.resize(img, (150, 150))

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(" No face detected.")
        return "No face detected", 0.0

    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
        face_img = rgb_img[y:y+h, x:x+w]  
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)  
        plt.imshow(face_img)
        plt.axis('off')
        plt.show()

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array.astype('float32') / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    if confidence < 0.5:
        return "Unknown", confidence
    elif predicted_class == 0:
        return "Person1", confidence
    elif predicted_class == 1:
        return "Person2", confidence
    elif predicted_class == 2:
        return "Person3", confidence
    elif predicted_class == 3:
        return "Person4", confidence
    elif predicted_class == 4:
        return "Person5", confidence
    elif predicted_class == 5:
        return "Person6", confidence


image_path = r"your_image"
person, confidence = predict_person(image_path, model)
print("Identified person: ", person)
print("Confidence score: ", confidence)


