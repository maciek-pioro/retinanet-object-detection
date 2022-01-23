# RetinaNet object / digit detection and classification
Detect multiple rotated and overlapping digits with bounding boxes by implementing RetinaNet variant.
High performance is achieved by using Focal loss and using Convolutional Neural Nets. 
These techniques are helpful in this problem because of high proportion of anchors not containing any digits and the fact that the task is translation-independent.


![image](https://user-images.githubusercontent.com/70006947/150699924-41f8656c-14c9-4321-810c-468a62b3cb4e.png)


## Anchor boxes selection
![image](https://user-images.githubusercontent.com/70006947/150699596-fba36725-ddf6-4790-85fb-a0f2b1003623.png)

Based on the heatmap of box sizes in the dataset I concluded that a small set of 11 anchors will do sufficiently good. At first I used more anchors but it seems that enlarging the set does not improve performance by much and slows down learning process.

## Model architecture

My initial attempts at creating a working model architecture were pretty bad, because at first I tried using fully connected layers after backbone. It turned out (which seems obvious now) that such model would have really hard time learning and working, since our task is translation-independent. Later I tried combining ConvNets and fully connected layers with the same result. My last (and successful) attempt was to mimic the final layers described in RetinaNet paper. In the beginning the net had some troubles with learning because I did not reshape the output correctly, but with proper reshaping the model could finally learn well and fast.

All heads are convnets consisting of 5 layers with the first layer being of kernel size 5 and further ones being of kernel size 3. Activation function used is ReLU.

## Training process

After noticing how time-consuming target creation was, I decided to create a finite train set about 10 times the size of the test set and precompute target boxes for each of the train examples. To make learning process stable I tried batch sizes of 16 and 32. SGD (i. e. batch size = 1) was pretty erratic while learning. I stuck with 32 since it worked quite well. Because I was trying out various batch sizes, I made the learning rate related to the batch size. Some research I did later suggests that was a bad idea (learning rate should be increased not decreased when batch size is increased), but I did not change the formula once I got the model to work.

Training took less than two hours on a laptop PC utilising a GeForce GTX 1650Ti GPU.

## Getting predictions from model output
### Score - model confidence

To order model predictions from best to worst I compute a score for each of the anchors.

Let ![image](https://user-images.githubusercontent.com/70006947/150699695-4c5c97c8-fb83-4e75-b5e9-ab8d4e88a6db.png) be an anchor. Let ![image](https://user-images.githubusercontent.com/70006947/150699676-117ae403-19c6-41bb-b388-626ba760b780.png) be the model's output predicting the digit for anchor ![image](https://user-images.githubusercontent.com/70006947/150699695-4c5c97c8-fb83-4e75-b5e9-ab8d4e88a6db.png) and ![image](https://user-images.githubusercontent.com/70006947/150699731-25896d67-12e6-4ea4-89c5-50bb555c40a3.png) the model's output predicting the rotation (all before softmax). I set score of anchor ![image](https://user-images.githubusercontent.com/70006947/150699695-4c5c97c8-fb83-4e75-b5e9-ab8d4e88a6db.png) to be

![image](https://user-images.githubusercontent.com/70006947/150699754-1a0e396c-e3bd-4d3b-985a-89fa7e95a2f1.png)

Maximal possible score would thus be 2.

### Algorithm

I started out with a simple algorithm for selecting final predictions - I would not allow any overlapping rectangles and always take the six most confident (highest score) predictions. I also rejected any malformed anchors (e. g. ![image](https://user-images.githubusercontent.com/70006947/150699806-c76eaf54-90c3-43ec-ad29-6a453f386018.png) or ![image](https://user-images.githubusercontent.com/70006947/150699818-415326e3-8a70-4309-8f8c-6bdde4b1b657.png) This turned out not to give good enough results because the images do not always contain six numbers and can overlap. Still, this was a good starting point.

After that I set the maximum IOU overlap to 0.1. It helped my net a little bit, but I was still losing a lot of accuracy on examples with less than 6 images.

I examined a couple of randomly generated images and how high the scores were for correctly and incorrectly predicted boxes. It turned out that if I set the score threshold to 1.9999 (I would reject boxes with score < 1.9999), my net would do much better.

This is a working solution, but I also suspect the score threshold might need to be changed if, for example, the net architecture or some learning parameters are changed. In other words, prediction selecting algorithm is fitted to a particular model.
