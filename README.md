# Object-Recognition-Using-CNN
Using the ALL-CNN network(2015) with state-of-the-art performance at object Recognition in CIFAR-10 image dataset


### [DEEP LEARNING](https://www.kaggle.com/dansbecker/intro-to-deep-learning-and-computer-vision)
Tensorflow is the most popular tool for deep learning and Keras is a popular API/interface for specifying deep learning models. Keras started as a standalone library for specifying deep learning models which then can be run in Tensorflow, Theano, and other deep learning computation engines. The standalne Keras library still exists but tensorflow has become the dominant engine from deep learning. So the creator of Keras implemented a version of keras that is built into tensorflow , this allows us to specify models with the elegant of keras while taking advantage of some powerful tensorflow features.

A tensor is a matrix(matrix of pixel intensities,etc) with any number of dimensions. Today's, deep learning models applies ,something called, Convulations to this type of tensor. A convulation is a small tensor that can be multiplied over little sections of the main image.<br/>
example: [ 1.5   1.5<br/>
          -1.5  -1.5]<br/>
**This convulaion is a horizontal line detector**
          
Convulations are also called filters because depending on the values in that convulation array it can pick out specific patterns from the image. If you multiply(the above convulation ex.) on a part of the image that has a hoz. line, you get a large value. If you mutiply it on a part of image without a hoz. line, you get value close to 0.

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

Size of a convulation can vary but it is the value inside it that determines what type of patterns it detects.<br/>
You don't directly choose the numbers to go into your convolutions for deep learning, instead the deep learning technique determines what convolutions will be useful from the data (as part of model-training using gradient descent and back propagation). Using these way to automatic create filters means we can have a large number of filters, to capture many different patterns. Each convulation we apply to the original 2D tensor(pixel intensity matrix) creates new 2D tensors a new
and we stack all these 2D tensors into a 3D tensor, this gives a the representation of image in 3D(1st dimension for hoz rows of pixels,2nd dim. for vertical columns and 3rd dim.(**the channel dimension**) for different convulation's outputs ).<br/>
Now we apply a new convulation layer(each set of convulations that get applied at the same time is called a layer) to this output 3D tensor(this contains patterns like hoz and vert. lines, dark spots locations, etc). This 2nd layer of convulations takes this map of pattern locations as input and multiplies it with 3D layer of convulations to find more complex patterns(black and white pixels combination => hoz and vert. lines combination => shapes(boxes,etc); concentric circles to wheels, etc and this may help us detect a car) and so on.After a number of layers and filters, we have a 3Dtensor with a rich summary of image content.

![Imgur](https://i.imgur.com/op9Maqr.png)<br/>
Once you create a filter, we apply it to each part of the image and map the output to an output tensor. This gives us a map showing where the associated pattern shows up in the image.

> While any one convolution measures only a single pattern, there are more possible convolutions that can be created with large sizes. So there are also more patterns that can be captured with large convolutions.<br/>For example, it's possible to create a 3x3 convolution that filters for bright pixels with a dark one in the middle. There is no configuration of a 2x2 convolution that would capture this.On the other hand, anything that can be captured by a 2x2 convolution could also be captured by a 3x3 convolution.<br/>Does this mean powerful models require extremely large convolutions? Not necessarily. In the next lesson, you will see how deep learning models put together many convolutions to capture complex patterns... including patterns to complex to be captured by any single convolution.
