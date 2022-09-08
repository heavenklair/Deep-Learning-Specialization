# Neural Network Basics

This section is the first course of the Deep Learning Specialization on Coursera. Find the "Treat me as a README.ipynb" file for further reading. 

Since `README.md` does not support latex, I created a alternative to write mathematical expressions.




#####################################################

Everything below is the README file content. Erase this and above when done with the converting ipynb to README

#####################################################

## Logistic Regression
Logistic Regreesion is an algorithm for binary classification

## Standard Notations for Logistics Regression
A single training example is represented by a pair, like $(x,y)$ where $x \in R^{n_x}$ ($x$ is an $R^{n_x}$) dimensional feature vector) and $n_x$ is the input size. $y \in (0,1)$ is the predicted output vector. 


$m$ denotes the examples in the dataset: $(x^{(1)}, y^{(1)}, x^{(2)},y^{(2)}, \dots , x^{(m)},y^{(m)})$ are the training examples.


$X \in R^{n_x \times m}$ is the input matrix. Here $n_x$ is the number of columns and $m$ is the number of rows.

$Y \in R^{1 \times m}$ is the label matrix.

## Logistic Regression

Given $x$, we want to find $\hat{y}$ where $\hat{y} = P(y=1|x)$.

Parameters: $w \in R^{n_x},\ b \in \R$

Output $\hat{y} = \sigma (w^T + b)$. We can also write $z = w^T x + b$ where z is the sigmoid function. The sigmoid function is represented as $\sigma(z) = \frac{1}{1+e^{-x}}$. 

If z is large, then $\sigma(z) \approx 1$.

If z is large negative number, then $\sigma(z) \approx 0$.

## Logistic Regression Cost Function 

We have $\hat{y} = \sigma(w^Tx+b)$, where $\sigma (z) = \frac{1}{1+1+e^{-z}}$.

Suppose we are given the training set: $\{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)}) \}$ and we want $\hat{y}^{(i)} = y^{(i)} $. We will use the cost function to analysis this approximation. The cost function of Logsitic Regression is given by:
$L(\hat{y}, y)  = -(y \text{log} \hat{y})  + (1-y) \text{log}(1-\hat{y})$

If $y=1:\  L(\hat{y}, y)  = -(\text{log} \hat{{y}})$

If $y=0:\ L(\hat{y}, y)  = (1-y) \text{log}(1-\hat{y})$

Toghther, it becomes: 

$$ J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)} ) = -\frac{1}{m} \sum_{i=1}^{m} \big[ y^{(i)} \text{log} \hat{y}^{(i)} + (1-y^{(i)}) \text{log}(1 - \hat{y}^{(i)}  ) \big] $$

## Gradient Descent

Check the Data Sciene Toolkit to learn more about usage of Gradient Descent.

After this, we would like to implement the gradient descent on m training examples.

We have the cost function

$J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(a^{(i)} , y^{(i)})$, where $a^{(i)} = \hat{y}^{(i)} = \sigma (w^Tx^{(i)} + b)$

Derivating the cost function w.r.t $`w_1`$, we get 

$\frac{\partial}{\partial w_1}\ J(w, b) = \frac{1}{m} \sum_{1}^{m} \sum_{i=1}^{m} \frac{\partial}{\partial} L(a^{(i)}, y^{(i)})$


## Algorithm for Gradient Descent

Initialize $J = 0,\ dw_1 = 0,\ dw_2 = 0,\ db = 0$.

$\text{For } i = 1 \text{ to }  m$

$$ z^{(i)} = w^T x^{(i)} + b$$ 

$$ a^{(i)} = \sigma(z^{(i)}) $$

$$ J+ = -[ y^{(i)} \text{log}a^{(i)} + (1-y^{(i)}) \text{log}(1-a^{(i)}) ]$$

$$ dz^{(i)} = a^{(i)} - y^{(i)}$$ 

$$ dw_1 += x^{(i)} dz^{(i)} $$

$$\vdots$$

$$ dw_n += x^{(i)} dz^{(i)}$$ 

$$ db += dz^{(i)}$$ 

$$ J = J/m $$

$$ dw_1 = dw_1/m ;\ dw_2 = dw_2/m;\ \dots dw_n = dw_n/m ; db = db/m$$

Writing a for loop in Python is not considered a good pracitce if we are performing a mathematical calculation which involves a heavy computation. So in our case, instead of writing,

$$dw_1 += x^{(i)} dz^{(i)}$$

$$\vdots$$

$$dw_n += x^{(i)} dz^{(i)}$$

in the for loop, we will use the `np.dot` function from the numpy library to perform the computations. So, we are going to get rid of $dw_1 =0$ and $dw_2 = 0$, we are going to initialize $dw$ by a vector as `dw = np.zeros(n_x, 1)`.

### Vectorizing the Gradient Descent Algorithm

Instead of the for loop we are going to use the vector value operation as $dw +=x^{(i)}\ dz^{(i)}$

and we will also get rid of the last line in the algorithm $dw_1 = dw_1/m ;\ dw_2 = dw_2/m;\ \dots dw_n = dw_n/m$ and replace it with $dw / = m$

Now we are also going to elliminate the first for loop that we have. Instead we are going to use the numpy arrays to carry out the mathematical operations.

So $X \in \R^{n_x \times m}$ matrix.

$w^T = \R^{(1, n_x)}$ will be row vector.

$z = [z^{(1)}\ \dots z^{(m)}] = w^T X [b \dots b]  = [w^T x^{(1)}+ b \ w^T x^{(2)}+b \dots w^T x^{(m)} +b ]$

Thus $z = \text{np.dot(w.T, x) + b}$ in Python.

At last, $A = [a^{(1)} \ a^{(2)} \dots a^{(m)} ] = \sigma(z)$

## Vectorizing Logsitic Regression

Now the final form of Logistic Regression in Linear Algebra becomes: 

$$ z = w^T X + b $$

$$  = \text{np.dot}(w.T \cdot X) + b $$

$$ A = \sigma (z)$$

$$ dz = A \cdot y$$

$$ dw = \frac{1}{m} X dz^T$$

$$ db = \frac{1}{m} \text{np.sum}(dz)$$

$$w := w - \alpha dw$$

$$b := b - \alpha db$$

The above is a single iteration of the Gradient Descent, we can repeat the above code to iterate the Gradient Descent.