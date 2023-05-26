import matplotlib.pyplot as plt
import numpy as np
import cv2

# %matplotlib inline

class_names = ['CUP', 'SPOON', 'FORK', 'MOUSE']

# Creating Realtime Dataset

CAMERA = cv2.VideoCapture(0)
camera_height = 500

raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []

while CAMERA.isOpened():

    # Read a new camera frame
    
    ret, frame = CAMERA.read()
    
    # Flip
    frame = cv2.flip(frame, 1)
    
    # Rescale the images output
    aspect = frame.shape[1]/float(frame.shape[0])
    res = int(aspect * camera_height) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))
    
    # The greean reactangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)
    
    # Show the Frame
    cv2.imshow("Capturing", frame)
    
    # Controls 1 = quit / s = capturing
    key = cv2.waitKey(1)
    
    if key & 0xff == ord('q'):
        break
    elif key & 0xff == ord('1'):
        # Save the raw frames to frame
        raw_frames_type_1.append(frame)
    elif key & 0xff == ord('2'):
        # Save the raw frames to frame
        raw_frames_type_2.append(frame)
    elif key & 0xff == ord('3'):
        # Save the raw frames to frame
        raw_frames_type_3.append(frame)
    elif key & 0xff == ord('4'):
        # Save the raw frames to frame
        raw_frames_type_4.append(frame)
        
    # Preview
    plt.imshow(frame)
    plt.show()
    
# Camera
CAMERA.release()
cv2.destroyAllWindows()

save_width = 339
save_height = 400

import os
from glob import glob

reval = os.getcwd()
print ("Current working directory %s" % reval)

print('img1: ', len(raw_frames_type_1))
print('img2: ', len(raw_frames_type_2))
print('img3: ', len(raw_frames_type_3))
print('img4: ', len(raw_frames_type_4))

# Crop the images 

for i, frame in enumerate(raw_frames_type_1):
    
    # Get roi
    roi - frame[75+2:425-2, 300+2:650-2]
    
    # Parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    # Save
    cv2.imwrite('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    
for i, frame in enumerate(raw_frames_type_2):
        
        # Get roi
        roi - frame[75+2:425-2, 300+2:650-2]
        
        # Parse BRG to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # resize to 224 x 224
        roi = cv2.resize(roi, (save_width, save_height))
        
        # Save
        cv2.imwrite('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
for i, frame in enumerate(raw_frames_type_3):
            
        # Get roi
        roi - frame[75+2:425-2, 300+2:650-2]
        
        # Parse BRG to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # resize to 224 x 224
        roi = cv2.resize(roi, (save_width, save_height))
        
        # Save
        cv2.imwrite('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
for i, frame in enumerate(raw_frames_type_4):

        # Get roi
        roi - frame[75+2:425-2, 300+2:650-2]
        
        # Parse BRG to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # resize to 224 x 224
        roi = cv2.resize(roi, (save_width, save_height))
        
        # Save
        cv2.imwrite('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
from glob import glob
from keras import preprocessing

width = 96
height = 96

images_type_1 = []
images_type_2 = []
images_type_3 = []
images_type_4 = []

for image_path in glob('img_1/*.*'):
    image = preprocessing.image.load_img(image_path, target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    
    images_type_1.append(x)

for image_path in glob('img_2/*.*'):
    image = preprocessing.image.load_img(image_path, target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    
    images_type_2.append(x)
    
for image_path in glob('img_3/*.*'):
    image = preprocessing.image.load_img(image_path, target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    
    images_type_3.append(x)

for image_path in glob('img_4/*.*'):
    image = preprocessing.image.load_img(image_path, target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    
    images_type_4.append(x)
    
plt.figure(figsize=(12, 8))

for i, x in enumerate(images_type_1[:5]):
    
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()

for i, x in enumerate(images_type_2[:5]):
    
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))

plt.show()

for i, x in enumerate(images_type_3[:5]):
        
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[2]))

plt.show()

for i, x in enumerate(images_type_4[:5]):
    
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[3]))
    
plt.show()

# Prepare Image to Tensor

X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)

# Check the shape using .shape() check the images count

print (X_type_1.shape)
print (X_type_2.shape)
print (X_type_3.shape)
print (X_type_4.shape)

(13, 96, 96, 3)
(23, 96, 96, 3)
(14, 96, 96, 3)
(22, 96, 96, 3)

X_type_2

X = np.concatenate((X_type_1, X_type_2), axis=0)

if len(X_type_3):
    X = np.concatenate((X, X_type_3), axis=0)

if len(X_type_4):
    X = np.concatenate((X, X_type_4), axis=0)
    
# Scaling the data to 1 - 0

X = X / 255.0

X.shape

(72, 96, 96, 3)

from keras.utils import to_categorical

y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [1 for item in enumerate(X_type_2)]
y_type_3 = [2 for item in enumerate(X_type_3)]
y_type_4 = [3 for item in enumerate(X_type_4)]

y = np.concatenate((y_type_1, y_type_2), axis=0)

if len(y_type_3):
    y = np.concatenate((y, y_type_3), axis=0)

if len(y_type_4):
    y = np.concatenate((y, y_type_4), axis=0)
    
y = to_categorical(y, num_classes=len(class_names))

y.shape

(72, 4)

# CNN Config

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

# Default Parameters

# Situational - values, you may not adjust these

conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2

# Values you can adjust
lr = 0.001
epochs = 5
batch_size = 10
color_channels = 3

def build_model(conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop,
                dense_1_n = dense_1_n, dense_1_drop = dense_1_drop,
                dense_2_n = dense_2_n, dense_2_drop = dense_2_drop,
                lr = lr):
    
    model = Sequential()
    
    model.add(Convolution2D(conv_1, (3, 3),
                            input_shape = (width, height, color_channels),
                            activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(conv_1_drop))
    
    # ---
    
    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))
    
    # ---
    model.add(Flatten())
    
    # ---
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))
    
    # ---
    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))
    
    # ---
    model.add(Dense(len(class_names), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue=0.5),
                  metrics=['accuracy'])
    
    return model

# model parameter

model = build_model()

model.summary()

# Do not run yet

history = model.fit(X, y, validation_split=0.10, epochs=10, batch_size=5)

print(history) 

# Model evaluation
scores = model.evaluate(X, y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss and accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Prediction

import seaborn as sns 
from sklearn.metrics import confusion_matrix

def plt_show(img):
    plt.imshow(img)
    plt.show()
    
cup = 'img_1/10.png'
spoon = 'img_2/10.png'
fork = 'img_3/10.png'
mouse = 'img_4/10.png'

imgs = [cup, spoon, fork, mouse]

# def predict_(img_path):

classes = None
predicted_classes = []

for i in range(len(imgs)):
    type_ = preprocessing.image.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_)
    plt.show()
    
    type_x = np.expand_dims(type_, axis=0)
    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])
    
cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=predicted_classes, annot=True)

type_1 = preprocessing.image.load_img('img_1/10.png', target_size=(width, height))

plt.imshow(type_1)
plt.show()

type_1_x = np.expand_dims(type_1, axis=0)
predictions = model.predict(type_1_x)
index = np.argmax(predictions)

print(class_names[index])

type_2 = preprocessing.image.load_img('img_2/10.png', target_size=(width, height))

plt.imshow(type_2)
plt.show()

type_2_x = np.expand_dims(type_2, axis=0)
predictions = model.predict(type_2_x)

index = np.argmax(predictions)
print(class_names[index])

# Live Predictions using camera

from keras.applications import inception_v3
import time

CAMERA = cv2.VideoCapture(0)
camera_height = 500

while(True):
    _, frame = CAMERA.read()
    
    # Flip
    frame = cv2.flip(frame, 1)
    
    # Rescale the images output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame, (res, camera_height))
    
    # Get roi
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Adjust alignment
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]
    
    # The green rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)
    
    # Predictions / Labels
    type_1_txt = '{}: {}%'.format(class_names[0], int(type_1_x*100))
    cv2.putText(frame, type_1_txt, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_2_txt = '{}: {}%'.format(class_names[1], int(type_2_x*100))
    cv2.putText(frame, type_2_txt, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    
    type_3_txt = '{}: {}%'.format(class_names[2], int(type_3_x*100))
    cv2.putText(frame, type_3_txt, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    
    type_4_txt = '{}: {}%'.format(class_names[3], int(type_4_x*100))
    cv2.putText(frame, type_4_txt, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    
    cv2.imshow('Real time object detection', frame)
    
    # Conrols q = quit / s = capturing
    key = cv2.waitKey(1)
    
    if key & 0xff == ord('q'):
        break
    
    # Preview
    plt.imshow(frame)
    plt.show()
    
# Camera
CAMERA.release()
cv2.destroyAllWindows()