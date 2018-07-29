# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./imgs/center.jpg "Center Lane Driving"
[image2]: ./imgs/recover_left.jpg "Recover frome Left"
[image3]: ./imgs/recover_right.jpg "Recover from Right"
[image4]: ./imgs/augment.jpg "Normal Img"
[image5]: ./imgs/augment_flip.jpg "Flipped Image"
[image6]: ./imgs/errorLoss.png "Loss Graph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 51-55) 

The model includes RELU layers to introduce nonlinearity (code lines 51-63), and the data is normalized in the model using a Keras lambda layer (code line 48). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 58-64). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 68). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to input a graph into the neural network and a prediction number comes out.

My first step was to use a convolution neural network model similar to the NVIDIA's LeNet. I thought this model might be appropriate because they had used this net to control a real vehicle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added some dropout layers to the model so that the model would be more robust to more training epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the turning corners. To improve the driving behavior in these cases, I gathered more training data at the turning corners especially when the vehicle recovered from the road sides to the road center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 47-65) consisted of a convolution neural network.
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

|Layer (type)  |Output Shape  | Connected to |
| ---- | ---- | ---- |
|lambda_1 (Lambda)| (None, 160, 320, 3) |lambda_input_1[0][0]|
|cropping2d_1 (Cropping2D)|(None, 65, 320, 3)| lambda_1[0][0]|
|convolution2d_1 (Convolution2D)|(None, 31, 158, 24)|cropping2d_1[0][0]|
|convolution2d_2 (Convolution2D)|(None, 14, 77, 36)|convolution2d_1[0][0]|
|convolution2d_3 (Convolution2D)|(None, 5, 37, 48)|convolution2d_2[0][0]|
|convolution2d_4 (Convolution2D)|(None, 3, 35, 64)|convolution2d_3[0][0]|
|convolution2d_5 (Convolution2D)|(None, 1, 33, 64)|convolution2d_4[0][0]|
|flatten_1 (Flatten)|(None, 2112)|convolution2d_5[0][0]|
|dropout_1 (Dropout)|(None, 2112)|flatten_1[0][0]|
|dense_1 (Dense)| (None, 100)|dropout_1[0][0]|
|dropout_2 (Dropout)|(None, 100)|  dense_1[0][0]|
|dense_2 (Dense)| (None, 50) |dropout_2[0][0] |
|dropout_3 (Dropout)|(None, 50)|dense_2[0][0]|
|dense_3 (Dense)| (None, 10)|dropout_3[0][0] |
|dropout_4 (Dropout)|(None, 10)|dense_3[0][0]|
|dense_4 (Dense)|(None, 1)|dropout_4[0][0]|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center autonomously. These images show what a recovery looks like starting from the left or the right side lane:

![alt text][image2]
![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would weken the possibility when the vechicle would just turn left. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 60,324 number of data points. I then preprocessed this data by normalization.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by that validation loss rate got the least value at the sixth epoch, which shown as below graph.

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
