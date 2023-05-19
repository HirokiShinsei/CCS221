import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
from glob import glob
from keras import preprocessing
# from keras.preprocessing import image
from keras.utils import load_img, img_to_array, array_to_img, to_categorical
import os
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from PIL import Image

# %matplotlib inline

# class_names = ['CUP','SPOON','FORK','MOUSE']
# class_names = ['KNIFE','WATER_BOTTLE','PHONE','GLASS']
class_names = ['FORK','GLASSES','PLATE','SPOON']

#creating realtime dataset

CAMERA = cv2.VideoCapture(0)
camera_height = 500

raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []

while CAMERA.isOpened():
    # read a new camera frame
    ret, frame = CAMERA.read()
    
    # flip
    frame = cv2.flip(frame, 1)
    
    # rescale the image output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)
    frame = cv2.resize(frame, (res, camera_height))
     
    # Calculate the center of the bounding box
    center_x = int((150 + 650) / 2)
    center_y = int((50 + 425) / 2)

    # Calculate the new coordinates for the centered bounding box
    box_width = 650 - 150
    box_height = 425 - 50
    
    rectangle_x1 = center_x - int(box_width / 2)
    rectangle_y1 = center_y - int(box_height / 2)
    rectangle_x2 = center_x + int(box_width / 2)
    rectangle_y2 = center_y + int(box_height / 2)
    
    # Calculate the offset for centering the bounding box
    offset_x = int((339 - box_width) / 2)
    offset_y = int((400 - box_height) / 2)

    # Adjust the coordinates based on the offset
    rectangle_x1 += offset_x
    rectangle_y1 += offset_y
    rectangle_x2 += offset_x
    rectangle_y2 += offset_y

    # Draw the centered bounding box
    cv2.rectangle(frame, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), (0, 255, 0), 2)

    # Draw the centered bounding box
    cv2.rectangle(frame, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), (0, 255, 0), 2)
    
    # show the frame
    cv2.imshow('Capturing', frame)
    
    # controls q = quit/ s = capturing
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('1'):
        # save the raw frames to frame
        raw_frames_type_1.append(frame)
        print("Captured type 1 frame.")
    elif key == ord('2'):
        raw_frames_type_2.append(frame)
        print("Captured type 2 frame.")
    elif key == ord('3'):
        raw_frames_type_3.append(frame)
        print("Captured type 3 frame.")
    elif key == ord('4'):
        raw_frames_type_4.append(frame)
        print("Captured type 4 frame.")

# camera
CAMERA.release()
cv2.destroyAllWindows()


save_width = 339
save_height = 400

retval = os.getcwd()
print ("Current working directory %s" % retval)

print ('img1: ', len(raw_frames_type_1))
print ('img2: ', len(raw_frames_type_2))
print ('img3: ', len(raw_frames_type_3))
print ('img4: ', len(raw_frames_type_4))

#crop the images

for i, frame in enumerate(raw_frames_type_1):
    
    #get roi
    roi = frame[rectangle_y1:rectangle_y2, rectangle_x1:rectangle_x2]
    # roi = frame[50:425, 150:650]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()

for i, frame in enumerate(raw_frames_type_2):
    
    #get roi
    roi = frame[rectangle_y1:rectangle_y2, rectangle_x1:rectangle_x2]
    # roi = frame[50:425, 150:650]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()


for i, frame in enumerate(raw_frames_type_3):
    
    #get roi
    roi = frame[rectangle_y1:rectangle_y2, rectangle_x1:rectangle_x2]
    # roi = frame[50:425, 150:650]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()

for i, frame in enumerate(raw_frames_type_4):
    
    #get roi
    roi = frame[rectangle_y1:rectangle_y2, rectangle_x1:rectangle_x2]
    # roi = frame[50:425, 150:650]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    plt.imshow(roi)
    plt.axis('off')
    plt.show()
    
width = 96
height = 96

# Initialize empty lists to store images of each type
images_type_1 = []
images_type_2 = []
images_type_3 = []
images_type_4 = []


for image_path in glob('img_1/*.*'):
    image = load_img(image_path, target_size=(width, height))
    x = img_to_array(image)
    
    images_type_1.append(x)

for image_path in glob('img_2/*.*'):
    image = load_img(image_path, target_size=(width, height))
    x = img_to_array(image)
    
    images_type_2.append(x)

for image_path in glob('img_3/*.*'):
    image = load_img(image_path, target_size=(width, height))
    x = img_to_array(image)
    
    images_type_3.append(x)

for image_path in glob('img_4/*.*'):
    image = load_img(image_path, target_size=(width, height))
    x = img_to_array(image)
    
    images_type_4.append(x)

print('Shape of images_type_1:', images_type_1[0].shape)
print('Shape of images_type_2:', images_type_2[0].shape)
print('Shape of images_type_3:', images_type_3[0].shape)
print('Shape of images_type_4:', images_type_4[0].shape)

plt.figure(figsize=(12,8))

for i, x in enumerate(images_type_1[:5]):
    
    plt.subplot(1,10,i+1)
    image = array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()

plt.figure(figsize=(12,8))

for i, x in enumerate(images_type_2[:10]):
    
    plt.subplot(1,10,i+1)
    image = array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))
    
plt.show()
    
plt.figure(figsize=(12,8))
for i, x in enumerate(images_type_3[:10]):
    
    plt.subplot(1,10,i+1)
    image = array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[2]))
    
plt.show()

plt.figure(figsize=(12,8))
for i, x in enumerate(images_type_4[:10]):
    
    plt.subplot(1,10,i+1)
    image = array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[3]))
    
plt.show()
    

# Prepare image to tensor
X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)

X_type_1 = X_type_1.reshape(-1, width, height, 3)
X_type_2 = X_type_2.reshape(-1, width, height, 3)
X_type_3 = X_type_3.reshape(-1, width, height, 3)
X_type_4 = X_type_4.reshape(-1, width, height, 3)


X = np.concatenate((X_type_1, X_type_2), axis=0)

if len(X_type_3):
    X = np.concatenate((X, X_type_3), axis=0)

if len(X_type_4):
    X = np.concatenate((X, X_type_4), axis=0)
    
#Scaling the data to 1 - 0

X = X/255.0

X = X.reshape(-1, width, height, 3)
# X.shape=(72, 96, 96, 3)

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

#Default Parameters

#situational - values, you may not adjust these

conv_1 =16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_n_drop = 0.2
dense_2_n = 512
dense_2_n_drop = 0.2

#values you can adjust

lr = 0.001
epochs = 15
batch_size = 15
color_channels = 3

def build_model( conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop,
                dense_1_n = dense_1_n, dense_1_n_drop = dense_1_n_drop,
                dense_2_n = dense_2_n, dense_2_n_drop = dense_2_n_drop,
                lr=lr):
    
    model = Sequential()
    
    model.add(Convolution2D(conv_1, (3,3),
                            input_shape = (width, height, color_channels),
                            activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(conv_1_drop))
    
    #---
    
    model.add(Convolution2D(conv_2, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(conv_1_drop))
    
    # ---
    
    model.add(Flatten())
    
    # ---
    
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_n_drop))
    
    # ---
    
    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_n_drop))
    
    # ---
    
    model.add(Dense(len(class_names), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue=0.5),
                  metrics=['accuracy'])
    
    return model

# model parameter

model = build_model()

model.summary()

history = model.fit(X, y, validation_split=0.10, epochs=epochs, batch_size=batch_size)

print (history)

# Model evaluation
scores = model.evaluate(X, y, verbose=1)
print ("Accuracy: %.2f%%" %(scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('loss and accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#prediction
import seaborn as sns 
from sklearn.metrics import confusion_matrix

def plt_show(img):
    plt.imshow(img)
    plt.show()
    
#learning data
fork = "img_1/10.png"
glasses = "img_2/16.png"
plate = "img_3/09.png"
spoon = "img_4/10.png"

imgs = [fork, glasses, plate, spoon]

# def predict_(img_path):

classes = None
predicted_classes = []
true_labels = []

for i in range(len(imgs)):
    type_ = load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_)
    plt.show()
    
    type_x = np.expand_dims(type_, axis=0)
    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])
    
    true_labels.append(class_names[i % len(class_names)])  # Append the true class to the true_labels list
    
cm = confusion_matrix(true_labels, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=predicted_classes, annot=True)

type_1 = load_img('img_1/10.png', target_size=(width, height))

plt.imshow(type_1)
plt.show()

type_1_x = np.expand_dims(type_1, axis=0)
predictions = model.predict(type_1_x)
index = np.argmax(predictions)

print(class_names[index])

type_2 = load_img('img_2/16.png', target_size=(width, height))

plt.imshow(type_2)
plt.show()

type_2_x = np.expand_dims(type_2, axis=0)
predictions = model.predict(type_2_x)
index = np.argmax(predictions)

print(class_names[index])

type_3 = load_img('img_3/09.png', target_size=(width, height))

plt.imshow(type_3)
plt.show()

type_3_x = np.expand_dims(type_3, axis=0)
predictions = model.predict(type_3_x)
index = np.argmax(predictions)

print(class_names[index])

type_4 = load_img('img_4/10.png', target_size=(width, height))

plt.imshow(type_4)
plt.show()

type_4_x = np.expand_dims(type_4, axis=0)
predictions = model.predict(type_4_x)
index = np.argmax(predictions)

print(class_names[index])


# Live predictions using camera

from keras.applications import inception_v3
import time 

CAMERA = cv2.VideoCapture(0)
camera_height = 500

while True:
    _, frame = CAMERA.read()

    # Flip
    frame = cv2.flip(frame, 1)

    # Rescale the image output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)  # Landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # Get ROI
    roi = frame[rectangle_y1:rectangle_y2, rectangle_x1:rectangle_x2]
    # roi = frame[50:425, 150:650]

    # Parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Adjust alignment
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    # Green rectangle
    # Calculate the center of the bounding box
    center_x = int((150 + 650) / 2)
    center_y = int((50 + 425) / 2)

    # Calculate the new coordinates for the centered bounding box
    box_width = 650 - 150
    box_height = 425 - 50
    
    rectangle_x1 = center_x - int(box_width / 2)
    rectangle_y1 = center_y - int(box_height / 2)
    rectangle_x2 = center_x + int(box_width / 2)
    rectangle_y2 = center_y + int(box_height / 2)
    
    # Calculate the offset for centering the bounding box
    offset_x = int((save_width - box_width) / 2)
    offset_y = int((save_height - box_height) / 2)

    # Adjust the coordinates based on the offset
    rectangle_x1 += offset_x
    rectangle_y1 += offset_y
    rectangle_x2 += offset_x
    rectangle_y2 += offset_y

    # Draw the centered bounding box
    cv2.rectangle(frame, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), (0, 255, 0), 2)

    # cv2.rectangle(frame, (150, 50), (650, 425), (0, 255, 0), 2)

    # Predictions/Labels
    type_1_text = '{} - {}%'.format(class_names[0], int(type_1_x * 100))
    cv2.putText(frame, type_1_text, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_2_text = '{} - {}%'.format(class_names[1], int(type_2_x * 100))
    cv2.putText(frame, type_2_text, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_3_text = '{} - {}%'.format(class_names[2], int(type_3_x * 100))
    cv2.putText(frame, type_3_text, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_4_text = '{} - {}%'.format(class_names[3], int(type_4_x * 100))
    cv2.putText(frame, type_4_text, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    cv2.imshow('Real-time object detection', frame)

    # Controls q = quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release the camera
CAMERA.release()
cv2.destroyAllWindows()
