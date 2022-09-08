# Special Applications: Face Recognition and Neural Style Transfer

One of the challenges of face recognition is the need to solve the one-shot learning problem. It means that for most face recognition applications we need to be able to recognize a person's face with just one single image. But, deep learning algorithms don't work well if we have only one training example. So instead, to make this work, what we do is use a **similarity function**.

$$ d(\text{img1, img2}) = \text{degree of difference between images} $$

If the two images are of the same person, we want this to output a small number. And if the two images are of two very different people we want it to output a large number. 

## Siamese Network 

The idea of running two identical convolutional neural networks on two different inputs (images) and then comparing them, is a Siamese neural network architecture.

## Triplet Loss Function 

One way to learn the parameters of the neural network is to define and apply gradient descent on the triplet loss function. We want the distance between an anchor image and a positive image to be smaller than the distance between an anchor image and a negative image. We are always looking at three images at a time: an anchor image, a positive image, and a negative image

$$ || f(A) - f(P)||^2 \leq || f(A) - f(N)||^2  $$

$$ d(A,P) \leq d(A,N)  $$

where d is a distance function.

$$ || f(A) - f(P)||^2 - || f(A) - f(N)||^2  \leq 0 $$

A way for the neural network to give a trivial output (non-zero) when the encoding for every image was identical to the encoding to every other image, is to add $\alpha$ to the equation like 

$$ || f(A) - f(P)||^2 - || f(A) - f(N)||^2 + \alpha \leq 0 $$

where $\alpha$ is a hyperparameter.

Given: 3 images, A, P, and N.

$$ \text{Loss}(A, P, N) = \text{max} \{ || f(A) - f(P)||^2 - || f(A) - f(N)||^2 + \alpha, \ 0 \}$$

The cost function becomes

$$ J = \sum_{i=1}^{m} \text{Loss}(A, P, N)   $$

For this arc to work we need to have multiple images of one person, but if we don't have that, this system will not work for us. After training a system on the above architecture, we can apply it to our one-shot learning problem. But for the training set, we do need to make sure we have multiple images of the same person, at least for some people in your training set, so that we can have pairs of anchor and positive images.

**How to choose the Triplets?**

We want to choose the triplet images A, P, and N so that they're hard to train the model on. So we should not randomly choose the images to train the model on. 

## Face Verification and Binary Classification

The **Triplet Loss** is one good way to learn the parameters of a continent for face recognition. There's another way to learn these parameters using binary classification. We can train the model by taking a pair of neural networks (Siamese Network) and have them both compute $f(x^{(i)})$, which can be 128 dimensional, or higher. And then have these be input to a logistic regression unit to then just make a prediction. Below is a figure to illustrate it: 

## INSERT AN IMAGE

We can set the output as following: 0 if the images are of the same person, and 1 if different. This is an alternative to the triplet loss for training a system like this.

The output, $\hat{y}$, will be: 

$$  \hat{y} = \sigma (\sum_{i=1}^{128} | f(x^{(i)}_k) - f(x^{(j)}_k) |) $$

where $f(x^{(i)})$ is the encoding of the image, and $k$ represents the $k^{th}$ component of the vector. In whole, it is the elementwise difference in absolute values between two encodings. We can think of these 128 numbers as features that get feed into logistic regression as follows:

$$  \hat{y} = \sigma (\sum_{i=1}^{128} w_i | f(x^{(i)}_k) - f(x^{(j)}_k) |  + b ) $$


## What are Deep ConvNets learning?

When we build a ConvNet, what does different layers of the ConvNet really learning about the images we train it on? A way to find it out is by visualizing the patterns seen in the different layers, or small patches of images in different layers. So we pick a hidden unit in layer 1. We scan through the training set images and try to find out what are the different patches that maximize that unit's activations. If we plot those patches, we will see patterns for example algorithm detecting horizontal lines, vertical lines, etc. As we go deeper in the ConvNet, we will see more sophisticated patterns for example, human faces, car tires, trees etc. 

Thus this is what the ConvNets actually learn when we implement them on an image dataset. 

## Neural Style Transfer and Cost funcion

We will define a cost that will compute how good a particular generated image is and we will use gradient descent to minimize the cost function. The first part is the cost function is called content cost, and is a function of the content image and the generated image. It measures how similar is the contents of the generated image to the content of the content image. The second function is style cost, and it measures how similar is the style of the image G to the style of the image S.  We will weight these with two hyper parameters $\alpha$ and $\beta$ to specify the relative weighting between the content costs and the style cost. 

The way the algorithm would run is as follows. 

-  Initialize the generated image G randomly (100x100x3 or 500x500x3 etc.)
- Specify the cost function.
- Use gradient descent to minimize the above cost function as 

$$ G:= G -  \frac{ \alpha}{\alpha G} J(G) $$

**Content Cost Function**

Say that you use hidden layer l to compute the content cost. The value of $l$ is chosen to somewhere in between the middle of the layers of the neural network. We can use a pre-trained ConvNet like VGG network, AlexNet etc. Now, given a content image and given a generated image, we want to measure how similar are they in content.

Let $a^{[l](C)}$ and $a^{[l](G)}$ be the activations of the layer $l$ on the images. If $a^{[l](C)}$ and $a^{[l](G)}$ are similar, then that would mean that both images are similar. So we will define $J_{\text{content}(C,G)}$ as a measure on how different the activations of these two layers are. We will use the element-wise difference between these hidden unit activations in layer $l$.

$$ J_{\text{content}(C,G)} = || a^{[l](C)} - a^{[l](G)} ||^2 $$

**Style Cost Function**

Take a hidden unit from layer $l$, and find how correlated the activations are in different channels  in that unit. This gives us an intution of measurement in computing style. We can compare the activations in the original units and the generated image and see how correlated the activations and that is what style means in this regard.

Let $a^{[l]}_{i,j,k}$  = activations at $(i,j,k)$ where i refers to the height, k refers to the width and k refers to the indexes across different channels. Now we compute a **Style Matrix** denoted by $G^{[l]}$ of dimension $n^{[l]}_c\ x\ n^{[l]}_c$. We are going to compute $G^{[l]}_{kk'}$ which will measure how correlated the activations are in channel k compared to activations in channel k', where $k=1,\ldots , n^{[l]}_c$.

$$ G^{[l](S)}_{kk'} = \sum^{n^{[l]}_{h}}_i \sum^{n^{[l]}_{w}}_j a^{[l](S)}_{i,j,k} \ a^{[l](S)}_{i,j,k'}  $$

The above is for Style image. We also do this for the generated image.

$$ G^{[l](G)}_{kk'} = \sum^{n^{[l]}_{h}}_i \sum^{n^{[l]}_{w}}_j a^{[l](G)}_{i,j,k} \ a^{[l](G)}_{i,j,k'}  $$

Finally, the style cost function becomes 

$$  J^{\text{style}}_{[l]}(S,G) =  || G^{[l](S)}_{kk'} - G^{[l](G)}_{kk'} ||^2_F $$

We can simplify this and add normalization to above and it will become

$$  J^{\text{style}}_{[l]}(S,G) =  \frac{1}{ (2n^{[l]}_{H} n^{[l]}_{W} n^{[l]}_{C})^2 }  \sum_k \sum_{k'} (G^{[l](S)}_{kk'} - G^{[l](G)}_{kk'} )^2  $$

We get better results if we use the sum of above function over different layers as 

$$  J^{\text{style}}(S,G) = \sum_l \lambda  J^{\text{style}}_{[l]}(S,G) $$

where $\lambda$ is a weigthed hyperparameter.

The above and final style cost function allows us to use different layers in a neural network where the early ones measure relatively simpler low level features like edges and some later layers measure high level features and cause a neural network to take both low level and high level correlations into account when computing the style.

$$  J(G) = \alpha J_{\text{content}}(C,G) + \beta J_{\text{style}}(S,G)  $$