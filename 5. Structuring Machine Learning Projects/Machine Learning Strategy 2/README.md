# Error Analysis

## Carrying out Error Analysis

What is Error Analysis? 

If our learning algorithm is not yet at the performance of a human, then we can manually examine the mistakes that the algorithm is making, giving us insight into what to do next. This process is called **Error Analysis**.

For example, if the cat classifier algorithm is incorrectly predicting some dog pictures as cats. In this case, should we try to write a different algorithm to classify dogs (can take couple of months)? Well, rather than spending a few months doing this only to risk finding out at the end that it wasnt that helpful, here is an error analysis procedure that can let you very quickly tell whether or not this could be worth the effort.

- Get approx 100 mislabeled dev set examples, then examine them manually. Count them up one at a time to see how many of these mislabeled examples in your dev set are actually pictures of dogs.

    - If the number of dog picture is tiny (suppose 5%) in the mislabled dataset, then we do not need to do anything. The reason is erradicating the dog picture miclassification will only solve 5% of the problem. What about remaining 95%? This problem gives us a new term, **Upper Bound** which means how much could we improve the performance by working on the dog problem. 
    - If the number of dog pictures are larger (suppose 50%), then we could be more optimistic about spending time on this problem.

**Evaluate Multiple Ideas in Parallel**

Count up the percentage of error contributions to the overall error of the dev test, and then try to tackle which error should you reduce first.

Same goes to when we encounter **Incorrectly Labeled Data**.

## Build your first system quickly, then iterate

One piece of advice is that one should build the system quickly and then iterate.


# Mismatched Training and Dev/Test Set

## Training and Testing on Different Distributions

The first thing that we need to make sure that the train/dev/test sets come from same distributions. We can think of what we want to achieve from the algorithm and set up the target in a way so that train distribution is matching the dev/test set distribution. 

If the distributions are different, and number of dev/test and train distributions are different, then 


## Bias and Variance with Mismatched Data Distributions

Estimating the bias and variance of our algorithm can really help us prioritize what to work on next. The way we analyze bias and variance changes when the training set come from different distribution than our dev and test set distribution. 

Suppose 
- Human error = 0%
- Training error = 1% 
- Dev error = 10%

If the training and dev set come from the same distribution, then we will conclude that we have a larger variance problem. That is, our algorithm is not generalizing well from the training set.

If the training data and dev data are coming from different distributions, we can no longer conclude the above. What if it is doing just fine on the dev set, and overfitting in the training set? Whatever the reason may be, we can say that two things changed going from training to dev set: 

1. The algorithm saw the data in the training set but not in the dev set.
2. The distribution of data in the dev set in different.

In order to tease out these two effects, we are going to define a new subset of data: 

**Training-dev set**: Same distribution as training set, but not used for training. 

Now, we are going to train the NN on the Training data, and evaluate the error on Training, Training-dev, and dev sets. If we see a jump in the error going from Training-dev to dev error, then we can be sure that we have a data mismatch problem. If we have a large gap between Bayes error and Training error, then we have Aviodable bias. If we have large gap between Training-dev and dev set, then we have data mismatch problem.

Key quantities to look at : 

- Human Level Error 
- Training set Error 
- Training-dev set Error 
- Dev Error 

The last thing, if we have a gap between Dev set and Test set, then that means we have a variance problem, or we have overfit to dev set.

## Addressing Data Mismatch

If we find that we have a data mismatch problem, then we can 

- Carry out manual error analysis to try to understand difference between training and dev/test sets.
- Make training data more similar; or collect more data similar to dev/test sets.

In order to get more data, we can use a technique called "artificial data synthesis". 


# Learning from Multiple Tasks

## Transfer Learning

One idea in deep learning is that sometimes we can just take knowledge that NN has learned from one task and apply that knowledge to a separate task. This is called Transfer Learning. 