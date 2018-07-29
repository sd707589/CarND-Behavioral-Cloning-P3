import csv
import cv2
import numpy as np
lines=[]
images=[]
measurements=[]

def process_image(inImg):
    return cv2.cvtColor(inImg,cv2.COLOR_BGR2RGB)

with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "./data/IMG/" # the path of your training IMG directory
        img_center = process_image(cv2.imread(path + row[0].split('/')[-1]))
        img_left = process_image(cv2.imread(path + row[1].split('/')[-1]))
        img_right = process_image(cv2.imread(path + row[2].split('/')[-1]))
        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])

#Data Augmentation
augmented_images, augmented_measurements=[],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

#Neural network structure
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.1,
          shuffle=True,nb_epoch=6,verbose=1)
#show network structure
model.summary()
#save the model
model.save('model.h5')

import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()