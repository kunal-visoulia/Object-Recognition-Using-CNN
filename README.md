# Object-Recognition-Using-CNN
Using the All-CNN network with state-of-the-art performance at object recognition on the CIFAR-10 image dataset published in the 2015 ICLR paper, "Striving For Simplicity: The All Convolutional Net".  This paper can be found at the following link:

https://arxiv.org/pdf/1412.6806.pdf

The CNN layers:<br/>
```
 model=Sequential()#we will be adding one layer after another
    
    #not the input layer but need to tell the conv. layer to accept input
    model.add(Conv2D(96,(3,3),padding='same',input_shape=(32,32,3)))#32x32x3 channels
    model.add(Activation('relu'))#required for each conv. layer
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1),padding='valid'))
    
    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
```

### DATASET
The dataset used is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The images are blurry(32x32 pixels only), Humans were only 94% accurate in classifying.

### DATASET PREPROCESSING
We need to preprocess the dataset so the images and labels are in a form that Keras can ingest
- normalize the images.
- convert our class labels to one-hot vectors. This is a standard output format for neural networks(*The class labels are a single integer value (0-9). What we really want is a one-h***we'll save some time by loading pre-trained weights for the All-CNN network. Using these weights, we can evaluate the performance of the All-CNN network on the testing dataset.***
16
ot vector of length ten because it is easier for CNN to output and it avoids biases to higher numbers; CNN is numerical,ex , a class value of 6 would skew the weights a lot differently than class label 1 . For example, the class label of 6 should be denoted [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]. We can accomplish this using the np_utils.to_categorical() function*).

>The original paper mentions that it took approximately 10 hours to train the All-CNN network for 350 epochs using a modern GPU, which is considerably faster (several orders of magnitude) than it would take to train on CPU.**So I used pre-trained weights**

weight_decay=1e-6
momentum=0.9
sgd;nesterov


**[softmax vs sigmoid](http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/)**

**[pre-trained neural networks](https://stats.stackexchange.com/questions/193082/what-is-pre-training-a-neural-network)**

### [DEEP LEARNING](https://www.kaggle.com/dansbecker/intro-to-deep-learning-and-computer-vision)
Tensorflow is the most popular tool for deep learning and Keras is a popular API/interface for specifying deep learning models. Keras started as a standalone library for specifying deep learning models which then can be run in Tensorflow, Theano, and other deep learning computation engines. The standalne Keras library still exists but tensorflow has become the dominant engine from deep learning. So the creator of Keras implemented a version of keras that is built into tensorflow , this allows us to specify models with the elegant of keras while taking advantage of some powerful tensorflow features.

A tensor is a matrix(matrix of pixel intensities,etc) with any number of dimensions. Today's, deep learning models applies ,something called, Convolutions to this type of tensor. A Convolution is a small tensor that can be multiplied over little sections of the main image.<br/>
example: [ 1.5   1.5<br/>
          -1.5  -1.5]<br/>
**This convulaion is a horizontal line detector**
          
Convolutions are also called filters because depending on the values in that convulation array it can pick out specific patterns from the image. If you multiply(the above convulation ex.) on a part of the image that has a hoz. line, you get a large value. If you mutiply it on a part of image without a hoz. line, you get value close to 0.

Example: say, we have a image with all pixel intensities = 200.<br/>
[ 200  200 .....<br/>
  200  200 .....<br/>
  ...  ... ..... <br/>
  ...  ... .....<br/>
                ]
 
mutiplying the upper left values of each tensor and upper right value of each tensor, so on  and we add them.<br/>
1.5 * 200 + 1.5 * 200 +(-1.5) * 200 + (-1.5) * 200 = 0

Say, we have black pixels above white pixels( a hoz. line):<br/>
[ 200  200 .....(black pixels)<br/>
   0    0 ..... (white pixels)<br/>
  ...  ... ..... <br/>
  ...  ... .....<br/>
                ]
<br/> 1.5 * 200 + 1.5 * 200 +(-1.5) * 0 + (-1.5) * 0 = 600 (a large value)

Size of a Convolution can vary but it is the value inside it that determines what type of patterns it detects.<br/>
You don't directly choose the numbers to go into your convolutions for deep learning, instead the deep learning technique determines what convolutions will be useful from the data (as part of model-training using gradient descent and back propagation). Using these way to automatic create filters means we can have a large number of filters, to capture many different patterns. Each convulation we apply to the original 2D tensor(pixel intensity matrix) creates new 2D tensors a new
and we stack all these 2D tensors into a 3D tensor, this gives a the representation of image in 3D(1st dimension for hoz rows of pixels,2nd dim. for vertical columns and 3rd dim.(**the channel dimension**) for different convulation's outputs ).<br/>
Now we apply a new convulation layer(each set of convulations that get applied at the same time is called a layer) to this output 3D tensor(this contains patterns like hoz and vert. lines, dark spots locations, etc). This 2nd layer of convulations takes this map of pattern locations as input and multiplies it with 3D layer of convulations to find more complex patterns(black and white pixels combination => hoz and vert. lines combination => shapes(boxes,etc); concentric circles to wheels, etc and this may help us detect a car) and so on.After a number of layers and filters, we have a 3Dtensor with a rich summary of image content.

![Imgur](https://i.imgur.com/op9Maqr.png)<br/>
Once you create a filter, we apply it to each part of the image and map the output to an output tensor. This gives us a map showing where the associated pattern shows up in the image.

> While any one convolution measures only a single pattern, there are more possible convolutions that can be created with large sizes. So there are also more patterns that can be captured with large convolutions.<br/>For example, it's possible to create a 3x3 convolution that filters for bright pixels with a dark one in the middle. There is no configuration of a 2x2 convolution that would capture this.On the other hand, anything that can be captured by a 2x2 convolution could also be captured by a 3x3 convolution.<br/>Does this mean powerful models require extremely large convolutions? Not necessarily. In the next lesson, you will see how deep learning  model=Sequential()#we will be adding one layer after another
    
    #not the input layer but need to tell the conv. layer to accept input
    model.add(Conv2D(96,(3,3),padding='same',input_shape=(32,32,3)))#32x32x3 channels
    model.add(Activation('relu'))#required for each conv. layer
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))#drop neurons randomly;helps the network generalize(prevent overfitting on training data) better so instead of having individual neurons 
    #that are controlling specific classes/features, the features are spread out over the entire network
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1),padding='valid'))
    
    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))models put together many convolutions to capture complex patterns... including patterns to complex to be captured by any single convolution.

## [Convolutional Neural Networks(CNN or ConvNets)](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
images recognition, images classifications. Objects detections, recognition faces etc., are some of the areas where CNNs are widely used.

https://cdn-images-1.medium.com/max/800/1*2SWb6CmxzbPZijmevFbe-g.jpeg

CNN image classifications takes an input image, process it and classify it under certain categories (Eg., Dog, Cat, Tiger, Lion). Based on the image resolution, it will see h x w x d( h = Height, w = Width, d = Dimension ). Eg., An image of 6 x 6 x 3 array of matrix of RGB (3 refers to RGB values) and an image of 4 x 4 x 1 array of matrix of grayscale image.

Technically, deep learning CNN models to train and test, each input image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1. The below figure is a complete flow of CNN to process an input image and classifies the objects based on values

### Convolution Layer
Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel.

Consider a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3 as shown in below

https://cdn-images-1.medium.com/max/800/1*4yv0yIH0nVhSOv3AkLUIiw.png

Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output shown in below<br/>
https://cdn-images-1.medium.com/max/800/1*MrGSULUtkXc0Ou07QouV8A.gif

Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels).

![](https://cdn-images-1.medium.com/max/800/1*uJpkfkm2Lr72mJtRaqoKZg.png)

### [Conv2D](https://keras.io/layers/convolutional/)
2D convolution layer (e.g. spatial convolution over images).

When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".


### [Strides](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)
Stride controls how the filter convolves around the input volume. In the example we had in part 1, the filter convolves around the input volume by shifting one unit at a time. The amount by which the filter shifts is the stride. In that case, the stride was implicitly set at 1. Stride is normally set in a way so that the output volume is an integer and not a fraction. Let’s look at an example. Let’s imagine a 7 x 7 input volume, a 3 x 3 filter (Disregard the 3rd dimension for simplicity), and a stride of 1. <br/>
https://adeshpande3.github.io/assets/Stride1.png

 what will happen to the output volume as the stride increases to 2.<br/>
 https://adeshpande3.github.io/assets/Stride2.png
  
### Padding
Sometimes filter does not fit perfectly fit the input image. We have two options:
- Pad the picture with zeros (zero-padding) so that it fits(padding = "same" results in padding the input such that the output has the same length as the original input)
- Drop the part of the image where the filter did not fit. This is called **valid padding(means no padding)** which keeps only valid part of the image(reuduce the dimension of the image).

### Pooling Layer/Downsampling Layer
Pooling layers section would reduce the number of parameters when the images are too large. Spatial pooling also called subsampling or downsampling which reduces the dimensionality of each map but retains the important information. Spatial pooling can be of different types:
- **Max pooling** take the largest element from the rectified feature map.
- Taking the largest element could also take the **average pooling**. 
- Sum of all elements in the feature map call as **sum pooling**. 

https://adeshpande3.github.io/assets/MaxPool.png<br/>
sourc: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

https://cdn-images-1.medium.com/max/800/1*gags_WLu961iw6I0ZX6iQA.png<br/>
source: https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8










