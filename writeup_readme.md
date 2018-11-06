# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/data_histogram.jpg "Visualization"
[image2]: ./writeup_images/gray_img.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Traffic_signs_from_web/4.jpg "Traffic Sign 1"
[image5]: ./Traffic_signs_from_web/14.jpg "Traffic Sign 2"
[image6]: ./Traffic_signs_from_web/22.jpg "Traffic Sign 3"
[image7]: ./Traffic_signs_from_web/23.jpg "Traffic Sign 4"
[image8]: ./Traffic_signs_from_web/31.jpg "Traffic Sign 5"
[image9]: ./Traffic_signs_from_web/33.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that simplifes the input to the neural net and reduces the amount of parameters and the training time required.

The grayscaling was done through dot product function in numpy, the three channels were dotted with the vector [0.299, 0.587, 0.114]. 

The image then had to be reshaped to ensure it can be accepted by tensorflow.

The gray channel was then normalized by subracting the channel by 128 and dividing by 128. Normalization helps with ensuring that the model doesn't get atuck at local minima and ensures the model converges smoothly through gradient descent.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer 1: Input layer -> 2D convolotuional layer - input image is a 32x32x1 gray image, this layer outputs 6 feature maps each is 28x28 (valid padding & 5x5 filter size). Relu activation was used, and a stride of 1 in all directions.

Layer 2: pooling layer with filter size of 2x2 and stride size if 2x2 (output is 6 feature maps each is 14x14)

Layer 3: 2D convolution with filter size of 5x5, valid padding, and stride of 1. The output of this layer is 16 feature maps each 10x10. Relu activation is applied.

Layer 4: Pooling layer with a filter size of 2x2 and stride size of 2x2. Output of this layer is 16 feature maps each is 5x5.

Layer 5: Flatten layer to flatten the output from layer 4, output is 400 neurons.

Layer 6: Fully connected layer with an  output of 120 neurons and relu activation. Dropout is utilized in this layer with keep probability of 0.5 when training, and 1 when using the network.

Layer 7: Fully connected layer with an output of 84 neurons and relu activation. Dropout is utilized in this layer with keep prob of 0.5 when training and 1 when using the network.

Layer 8: Output layer (fully connected) with 43 output neurons.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, tuned the following hyperparameters:

Epochs - > 20
Batch size -> 100
Learning rate -> 0.001

All the above hyper-parameters where tuned to produce the best validation accuracy.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 95.4%
* test set accuracy of 93.5%

If an iterative approach was chosen:
* The inital architecture chosen was based on LeNet's arcitecture, it was chosen since it was proven to be effective in identifying basic features in images.

* However just the basic LeNet architecture wasn't sufficient in providing the desired output, the reason was that the model was overfitting the data. 

* The model was then adjusted by adding dropout to layers 6&7 of the model, the keep probability used was 0.5 during training, this helped the model generalize and produce higher validation accuracy.

* Epochs number, batch size and learning rate were also tuned. A smaller epoch number was initally selected but it was proven that using higher epochs number helps the model achieve higher accuracy. Also the batch size was optimized as well as the learning rate, the final values (100 and 0.001 respectively) produced the best results. In addition the number of neurons in the fully connected layers at the end of the network were also tuned to produce the best output.
 
* I believe the most important design choices I made were first selecting to start off with a network architecture that is known to be effective (LeNet), this helped speed up the development process of the network since I didn't try to re-invent the wheel coming up with a new network architecture. The second important design choice was to use dropout in layers 6 & 7 of the models, which really helprf the model generalize and produced higher validation accurary.

If a well known architecture was chosen:
* What architecture was chosen?
	LeNet
* Why did you believe it would be relevant to the traffic sign application?
 	Since Lenet is known to be effective at extracting simple features from an image ( proven for handwritten characters), and since traffic signs have simple features that charactirze them, then LeNet is a good starting point.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The final accuracies for training, validation and testing all are relitevly close to each other, indicating that model has generalized well and is not extremely overfitting the training data.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Some of the difficulties that can cause these images hard to classify are:

Image1:  The background is split between sky and trees which can affect accuracy of classification.

Image2: The stop sign is not dead perpendicular to the camera but is on a bit of an angle.

Image3: There is a bit of a sun glare on the sign which might introduce difficulty in classifying this image.

Image4: The image has a bit of blur on the bottom right section.

Image5: The sign is rotated a bit counter-clockwise in the image

Image6: The blue sky background has a similar color to the sign itself. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h        		| 70 km/h   									| 
| Stop Sign    			| Stop Sign										|
| Bumpy road			| Bumpy road									|
| Slippery Road    		| Slippery Road					 				|
| Wildlife crossing		| Wildlife crossing    							|
| Right Turn			| Right Turn           							|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.5 %, which is a relatively high accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

The softmax probabilites for each of the pictures are displayed below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999       			| 70 km/h   									| 
| .65     				| Stop Sign										|
| .84					| Bumpy road									|
| .999	      			| Slippery Road					 				|
| .999				    | Wildlife crossing   							|
| .80				    | Right Turn          							|

As it can be seen the probabilities are close to 100% for most images except for the stop sign (which has the lowest probability of 64%) followed by the right turn sign and the bumpy road sign, which has 0.80 and 0.84 probabilies respectively. Some of the reasons why these images have low probabilities are outlined above in question 1.

