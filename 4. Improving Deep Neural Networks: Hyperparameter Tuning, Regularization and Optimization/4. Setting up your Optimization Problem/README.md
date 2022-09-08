# Setting up your Optimization Problem

## Normalizing Inputs

One of the techniques to speed up the algorithm while training is if we normalize our inputs. Normalizing has 2 steps: 

$\quad \circ$ Subtract mean: 

$$ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} $$

$$ x:= x - \mu $$

$\quad \circ$ Normalize Variance: 

$$ \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} ** 2 $$

$$ x = x/ \sigma $$

The above is an element-wise squaring. Now we will end up with the variance of $x_1$ and $x_2$ being equal to 1.

If we used to $\mu$ and $\sigma$ to scale our training data, then we should use the same quantities to normalize the test set. 

## Vanishing/Exploding Gradient

When we are training a very deep neural network, our derivatives or our slopes can sometimes get either very big or very small. This makes training difficult. 

Let the weights, $w^{[l]} > I$ where $I$ is the identity matrix. If we choose $b=0$ then we will get a result at the end where $\hat{y} = w^{[l]} \cdot \text{a matrix here} \cdot X $.  The values in the matrix will be larger than 1, let's say 1.5, and thus we will get $\hat{y} = 1.5^{L} $. This number will grow exponetially, and the gradients will explode.

The gradients will vanish if $w^{[l]} < I$.

## Weight Initialization for Deep Networks

Let us say that there are 4 inputs to one node. Suppose $b=0$, then 
$$ z = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 $$


We can see that if $n$ is large, then we want the $w_i's$ to be smaller in order to not let the z explode. One thing that we can do is set the variance of w: $\text{var}(w_i) = \frac{1}{n}$.

In Python, $w^{[l]} = \text{np.random.randn(shape)} \times \text{np.sqrt}(\frac{1}{n^{[l-1]}})$

In english, it means that square root of the numbers of features that are being fed into each Neuron in layer l, and that is $n^{[l-1]}$.

If we are using a ReLU acitivation function, rather than $\frac{1}{n}$, the variance turns out to be $\frac{2}{n}$. 

If we are using a tanh acitivation function, rather than $\frac{1}{n}$, the good choice of variance turns out to be $\frac{1}{n}$. 

So if the input features of activations are roughly mean 0 and standard variance 1, then this would cause z to also take on similar scale. This doesnt solve but helps in reducing the exploding and vanishing gradient problem.

## Numerical Approximation of Gradients and Implemetation

When we implement back propagation, there is test called gradient checking that can help us make sure that our implmentation of back prop is correct.

**Gradient Checck for a Neural Netowrk** 

$\quad \circ$ Take $W^{[i]},\ b^{[i]}$ and reshape them into a big vector $\theta.$ Since $W^{[i]}$ will be a matrix, it will be transformed into a vector. 

$\quad \circ$ Doing so will give us the cost function, $J$ in terms of $\theta$.
 
$\quad \circ$ Take $dW^{[i]},\ db^{[i]}$ and reshape them into a big vector $d\theta.$ Since $dW^{[i]}$ will be a matrix, it will be transformed into a vector. 


The question: Is $d\theta$ the graident/slope of the cost function $J$?

**Gradient Checking**

We are going to use the for loop in python to implement gradient checking.

$$  \text{for each }  i: $$

$$ d \theta_{approx}[i] = \frac{ j(\theta_1, \theta_2, \dots, \theta_i + \epsilon, \dots) -  j(\theta_1, \theta_2, \dots, \theta_i - \epsilon, \dots) }{ 2 \epsilon} \\
\approx d\theta[i] = \frac{\partial J}{\partial \theta_i} $$

After computing the above "for loop" with respect to each i, we will end up with: $d\theta_{approx}[i]$ which is of same dimension as $d\theta$, and $\theta$. Now, we need to check if $d\theta_{approx} \approx? d\theta$

How do we check it? 

We would compute the distance between the these two vectors and compute the $L_2$ norm of that: $||d\theta_{approx} - d\theta||_2$. This is the sum of squares of elements of the differences and the square root of that. We get the Euclidean distance. To normalize by the lengths of these vectors, we divide by the following:

$$ \frac{||d\theta_{approx} - d\theta||_2}{||d\theta_{approx}||_2 + ||d\theta||_2 } $$

In practice, we can use $\epsilon = 10^{-7}$. With this value of $\epsilon$ if we get a value of $10^{-7}$ for the above formula, then it is a great approximation. Similarly, $10^{-5}$ is okay, and $10^{-3}$ is a worrisome value and it might indicate that there is a bug somewhere in the calculations.

## Gradient Checking Implementation Notes

$\quad \circ$ Do not use Gradient checking in training. Use it only to debug

$\quad \circ$ If an algorithm fails a grad check, look at components to try to identify the bug.

$\quad \circ$ Remember Regularization: During grad check, remember your regularization term if youre using regularization

$\quad \circ$ Grad check doesnt work with dropout. One thing we can do is peform grad check without dropout, and if it is correct, then turn the dropouts afterwards.

$\quad \circ$ Run at random intialization; perhaps again after some training: 