# Deep Neural Networks 

Use the "Deep Neural Networks.ipynb" file as a subsitute for README file. 

Topics covered in the notes: 
* Deep L-Layer Neural Network
* Forward Propagation in a Deep Network
* Getting the Matrix Dimensions Right
* Building Blocks of Deep Neural Networks
* Forward and Backward Propagation
* Parameters vs Hyperparameters

###########################################

I am rewriting the readme ipynb to README.md file.

###########################################


## Deep Neural Network Notation
L = number of layers

$n^{[l]}$ = number of units in layers, $l$

$\quad \circ \ n^{[0]} = n_x =$ number of units in the input layer

$a^{[l]}$ = activations in layer $l$

$\quad \circ$ we compute $a^{[l]} = g^{[l]}(z^{[l]})$

$w^{[l]} = $weights for computing $z^{[l]}$ in layer $l$

$b^{[l]} = $ used to compute $z^{[l]}$

Input features are called $X$ and $X = a^{[0]}$. The activation of the final layer, $a^{[l]} = \hat{y}$.

## Forward Propagation in a Deep Neural Network

The general equation for forward propagation for a single training example looks like:

$$ z^{[l]} = w^{[l]} a^{[l-1]} + b^{[l]} $$

$$ a^{[l]} = g^{[l]}(z^{[l]}) $$

The general equation for forward propagation for vectorized sytem of equations of the entire training set looks like:

$$  Z^{[l]} = W^{[l]} A^{[l-1]} + ^{[l]} $$ 

$$ A^{[l]} = g^{[l]}(Z^{[l]}) $$

## Getting the matrix dimensions right

$\quad \circ \ $ The dimensions of parameter $W^{[l]}$ are: $(n^{[l]}, n^{[l-1]})$.

$\quad \circ \ $ The dimensions of parameter $b^{[l]}$ are: $(n^{[l]}, 1)$.

$\quad \circ \ $ If we are implementing the backward propagation, the dimensions of $dW^{[l]}$ are: $(n^{[l]},  n^{[l-1]})$ (same dimensions as $W^{[l]}$\quad \circ \ $$

$\quad \circ \ $ If we are implementing the backward propagation, the dimensions of $db^{[l]}$ are: $(n^{[l]}, 1)$ (same dimensions as $b^{[l]})$

Now lets check the dimensions of $a^{[l]},\ z^{[l]}$ and  $A^{[l]},\ Z^{[l]}$.

We know that equation

![Underbrace equation 1](images/underbrace_equ_1.png)

In the vectorized form, it beomes


![Underbrace equation 1](images/underbrace_equ_2.png)

$\quad \circ \ $ Thus dimension of ${Z^{[l]}}$ and $A^{[l]}$ are $(n^{[l]}, m)$.

$\quad \circ \ $ The dimensions of $A^{[0]} = X$ are $(n^{[0]}, m)$

$\quad \circ \ $ If we are implementing the backward propagation, the dimensions of $dZ^{[l]},\ dA^{[l]}$ are: $(n^{[l]}, m)$ (same dimensions as $Z^{[l]})$ and $A^{[l]}$.

## Building Blocks of Deep Neural Network

### Forward and Backward Propagation

Suppose we are at the layer $l$ in the network. Then we have: 

$\quad \circ \ W^{[l]},\ b^{[l]}$

$\quad \circ \ $ For forward Propagation: Input $a^{[l-1]}$, $\quad$ Output $a^{[l]}$

$\quad \circ \ $ For backward Propagation, Input $da^{[l]}$, cache$(z^{[l]})$, $\qquad$ ouput $da^{[l-1]}$, $dw^{[l]}$, $db^{[l]}$

## Backward and Forward Propagation

### Forward Propagation for layer $l$

$\quad \circ \ $  Input $a^{[l-1]}$

$\quad \circ \ $  Output $a^{[l]}$, cache($z^{[l]}$, along with $w^{[l]}$, and $b^{[l]}$)

Implentations for single training examples

$$  z^{[l]} = w^{[l]} \cdot a^{[l-1]} + b^{[l]} $$

$$ a^{[l]} = g^{[l]}(z^{[l]}) $$

and in Vecotrized Form

$$ Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]} $$
$$ A^{[l]} = g^{[l]}(Z^{[l]}) $$

### Backward Propagation for layer l

$\quad \circ \ $ Input $da^{[l]}$

$\quad \circ \ $  Output $da^{[l-1]},\ dW^{[l]},\ db^{[l]}$ along with $a^{[l-1]}$  in cache

Implentations for single training examples
$$ dz^{[l]} = da^{[l]} \times g^{[l]'}(z^{[l]}) $$

$$ dW^{[l]} = dz^{[l]} \cdot a^{[l-1]} $$

$$ db^{[l]} = dz^{[l]} \cdot a^{[l-1]}$$ 

$$ da^{[l-1]} = W^{[l]T} dz^{[l]}$$ 

$$ \text{ Repeat from above } $$

and in Vecotrized Form

$$ dZ^{[l]} = dA^{[l]} \times g^{[l]'}(Z^{[l]}) $$

$$ dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}
db^{[l]} = \frac{1}{m} \text{ np.sum } ( dZ^{[l]} , \text{ axis = 1}, \text{ keepdims =True})$$ 

$$dA^{[l-1]} = W^{[l]T} dZ^{[l]}$$

## Parameters vs Hyperparameters

Parameters: $W^{[l]},\ b^{[l]}, \dots$

Hyperparameters: 

$\quad \circ \ $ Learning rate $\alpha$

$\quad \circ \ $ Number of iterations

$\quad \circ \ $ Number of hidden layer ($l$)

$\quad \circ \ $ Hidden units $n^{[1]},\ n^{[2]}, \dots$

$\quad \circ \ $ Choice of activation function

Later, we will also see momentum term, mini batch size, various forms of regularization parameters, and so on.