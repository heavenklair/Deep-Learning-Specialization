# Convolveutional Neural Networks

## Computer Vision Edge Detection

In computer vision, we need to find out the edges in pictures to classify different objects and "detect objects" as we say. We simply convolve the image by a 3 x 3 matrix to get a new image which shows the edges more clearly. We can obtain multiple new pictures like this from the original picture to detect the edges. Some of the well know filter matrices are Sobel filter, Scharr filter, etc. 

There can be times we do not have to hand pick the 9 numbers in the matrix, rather we can use them as parameters and learn the optimal value of those using back prop.

The rule of convolveing the image is as follows: If the image is a nxn matrix, and the matrix  convolveving matrix is fxf, then the dimension of the output will be a matrix of (n+f-1, n+f-1). There are two downsides to this: 

1. Every time we apply a convolveutional operator, the image shrinks. So after a short while, we will be left with an image which is really small, maybe it **shrinks down** to one by one matrix. We don't want our image to shrink every time we detect edges or to set other features on it. 
2. Second downside is that, if we look the pixel at the corner or the edge, that pixel is only in one of the outputs, because this touches the three adajcent regions in comparision to a pixel in the middle of the picture. Thus we are **throwing some useful information** in this process.

To solve this, we can use **Padding**, which means adding some extra dimensions on the original picture. Using this, we will have able to preserve the original size of the picture. We can pad the original image with zeros, and we will denote the padding amount as p. The newer dimension of the original image becomes (n+2p-f+1, n+2p-f+1).

## Strided Convolutions

This basically is a convolveuational method but we are moving jumping the rows and columns by 2 instead of 1. Then it is called a **Stride = 2**. The input and output dimensions are governed by the following formula: If we have an (n, n) image and a (f, f) filter and we use padding p, stride s, we end up with a output of dimension ((n+2p-f)/s + 1 x (n+2p-f)/s + 1)). If we end up with a fractical dimensional in the output, not an integer, then we will use the floor operation to get an integer dimension of the output image.

## Convolutions Over Volume

Until now, we were talking about convolveution in a 2D images. If we are using RGB images, then we know that we have 3 layers of colors in that image. How will we convolve on these layers of images. The dimensions of the input image becomes 3 here. The dimension of the output image remains the 2 dimensionsal. The process is exactly the same in the 3 dimensional case. We will convolveve using the 3D over the input image. In this case, we will have 27 numbers to multiple to get the (1,1) output color of the output image.

If we want to detect multiple edges, we will use multiple filters. The second output image will become the second layer of the first output image. We are going to use **channels** to denote the layers in the output image.

## One Layer of a Convolutional Network

One layer of forward propagation of standard Neural Network is as follows: 
$$z^{[1]} = W^{[1]} a^{[0]} + b^{[1]} $$

$$ a^{[1]} = g(z^{[1]}) $$

The input image is the $a^{[0]}$, also denoted as $X$. The filter layers become $W^{[1]}$. The operations same the same, but the input becomes a different dimensional images.

If we have 10 filters that are $3 \times 3 \times 3$ in one layer of a neural network, how many parameters does that layer have?

So one filter has 27 + 1 = 28 parameters, and we have 10 such layers, thus we have $28 \times 10$ parameters. We can use these filters to detect features, vertical edges, horizontal edges and other features in the images. We can detect many features, even in a very very large image by using less number of parameters. This make the CNN makes less prone to overfitting.

Summary of notation: 

* If layer $l$ is a convolution layer:

$$f^{[l]} = \text{filter size}, \qquad p^{[l]} = \text{padding}, \qquad s^{[l]} = \text{stride}$$

* Input: 

$$ n^{[l-1]}_H \times n^{[l-1]}_W \times n^{[l-1]}_c $$

where $n_c$ is the number of channels in the previous layer, and H W are height and width.

* Output: 

$$ n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_c  $$

* The output volume of this layer is given by

$$ n^{[l]} = \lfloor \frac{n^{[l-1]} + 2p^{[l]} -f^{[l]} }{s^{[l]} } + 1 \rfloor$$

for height and width respectively.

If the output volume has the depth $n^{[l]}_c$, then we konw from the previous examples that thats equal to number of filters in that layer.

$$ n^{[l]}_c  = \text{number of filters} $$

* Each filter is: 

$$ f^{[l]} \times f^{[l]} \times n^{[l-1]}_c $$

* Activations: 

$$ a^{[l]} \rightarrow (n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_c  ) $$

If we are implementing the activations in a vectorized form, then

$$ A^{[l]} \rightarrow m \times n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_c $$

* Weights: 

$$ f^{[l]} \times f^{[l]} \times n^{[l-1]}_c \times n^{[l]}_c$$

It is dimension of one filter times the number of filters.
* Bias: 

$$ n^{[l]}_c $$

## Types of Layers Convolutional Network

* Convolutional Layer (CONV)
* Pooling layer (POOL)
* Fully connected layer (FC)

## Pooling Layer

Other than convolutional layer, ConvNets often use pooling layers to reduce the size of the representation, to speed up computation, as well as make some of the features detect a bit more robust features.

**Max Pooling**

![Max Pooling](images/max_pooling.jpeg)

We split the input image into four part, four squares and the output will be the max from the corresponding reshaded region. 

This is as if we are applying the filter size of 2 because we are taking the 2 x 2 regions and stride of 2. These become the hyperparameters of Max Pooling. One advantage of using Max pooling is that it doesnt use any parameters, instead only use hyperparameters.

We can also use **Average Pooling** where we will take the average of the numbers instead of taking max of numbers. **Max Pooling** is more common than **average pooling**.

To summarize the hyerparameters: 

* f : filter size
* s : stride 
* Max or Average Pooling

The most common choice of hyperparameters are f = 2, s = 2. This has the effect of roughly shrinking the height and width by a factor of two.

* Input: volume of size:

$$n_H \times n_W \times n_c$$

* Output: Volume of size 

$$ \lfloor \frac{n_H - f}{s} + 1  \rfloor  \times \lfloor \frac{n_W - f}{s} + 1  \rfloor \times n_c $$

## Convolutional Neural Network

Here is an example of Neural Network, espifically LeNet-5.

![Max Pooling](images/NN_Example_LeNet-5.jpeg)

As we go deeper in the network, we can see that height ($n_H$) and width ($n_W$) decreases and the depth ($n_C$) is increase. We will notice that a lot these properties.

## Why Convoltions? 

Two main advantages of convolutional layers over just using fully connected layers are: 

1. Parameter Sharing: A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.

2. Sparsity of Connections: In each layer, each output value depends only on a small number of inputs.