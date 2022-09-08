## Hyperparamter Tuning 

Here are some guidelines, some tips on how to systematically organize your hyperparameter tuning process which hopefully will make it more efficient for us to converge on a good setting of the hyperparameter.

Out of all the hyperparameters, the **learning rate**, alpha is the most important hyperparameter, so the first one to be tuned is this. 

After tuning **alpha**, we can tune the momentum term (beta). A good starting point for beta is 0.9. Along with beta, we can also tune the **mini-batch size** to make that the optimization algorithm is working efficiently. Often we can fiddle with the** number of hidden units** int he neural nets.

Then the **number of layers** and **learning-rate-decay** come at the last. While using the Adam Algorithm, we do not need to change **beta1**, **beta2** or **epsilon**. Default values for **beta1** is 0.9, for **beta2** is 0.99 and for **epsilon** is 10e8.

If we wish to find the best values of hyperparameters for our model, the best way to do so is to plot the accuracies of the model with those hyperparameters. Make sure that the model is initialized with random values.


## Using an Appropriate Scale to Pick Hyperparameters

How can we find the best value for the hyperparemeters discussed above? How can we iterate through the different value for those hyperparameters? It turns out that sampling at random does not mean sampling uniformly at random, over the range of valid values. Instead it is important to pick an appropriate scale on which to explore the hyperparameters. 

Let's say we are searching for the learning rate hyperparameters, alpha. The possible values for alpha range from 0.0001 to 1. Instead of running the algorithm on all the possible values (linear) in that range, it is more reasonable to search for hyperparameters on a log scale. We can divide the number line from 0.0001 to 1 as follows: {0.0001, 0.001}, {0.001, 0.01}, {0.01, 0.1}, and {0.1, 1}. Then we can sample uniformly, at random, on this type of log scale. In Python it will look like something like: 

```python
r = -4 * np.random.rand()
alpha = 10 ** r
```

r will be a random number choosen between -4 and 0. alpha will be between 0.0001 and 1.

**Hyperparameters for exponentially weighted averages**

The case to find the best value for beta which ranges from 0.9 to 0.999 is similar to above. We can compute the 1-beta and get the values that range from 0.1 to 0.001. In this case, the values are reveresed. We can sample uniformly at random by choosing a value for r which would be between -3, -1. Then we can set 1-beta = 10 **r, and eventually find beta.

The one reason we are using log scale instead of linear, while trying to find beta, is that when beta is close to 1, the sensetivity of the results we get changes with every small changes to beta. With log scale, it doesnt happen !! 

## Batch Normalization in "Batch Normalization.ipynb"
Use the "Batch Normalization.ipynb" file to read more about it.

## Softmax Regression (Multi-Class Logistic Regression)
Use the "Softmax Regression (Multi-Class Logistic Regression).ipynb" file to read more about it.

## TensorFlow
Use the "TensorFlow Introduction.ipynb" file to read more about it.