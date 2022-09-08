## Structuring Machine Learning Projects

Let us say that we built a project that is giving back a 90% accuracy. It is still not good enough for deployment, so there are a lot of things that we can try. For example: 
- Collect more data
- Collect more diverse training set
- Train algorithm longer with gradient descent
- Try Adam instead of gradient descent
- Try bigger network
- Try smaller network
- Try Dropout
- Add L2 regularization
- Network Architecture
    - Activation functions
    - Number of hidden units

Suppose we choose one of the above method to try only to realize after months that it didn't do any good to us? For example, we choose to collect data and spend months doing it and at the end, it doesn't make any contribution to no improvement the performace of the system. What if we have quick and effective ways to figure out which of the all of the above ideas and others are worth pursuing and which ones you can safely discard. This repo contains a number of strategies and ways of analyzing a machine learning problem that will point us in the direction of the most promising things to try.

The sequence: 
* Machine Learning Strategy 1
    * **Setting up your Goal**
    * **Comparing the Human-level Performance**

* Machine Learning Strategy 2
    * **Error Analysis**
    * **Mismatched Training and Dev/Test Set**
    * **Learning from Multiple Tasks**
    * **End-to-End Deep Learning**