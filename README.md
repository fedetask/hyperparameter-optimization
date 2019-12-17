# Hyperparameter optimization
The goal of this repo is to create a library of tools to estimate hyperparameters of machine learning algorithms. It's often the case that a ML algorithm depends on a high number of parameters, and finding their best combination becomes hard due to the number of those and the time cost of evaluating a single instance.

## What this repo contains:
### Gaussian Process regression:
NOTE: This code was initially developed to help us estimating the best parameters in a [Reinforcement Learning assignment](https://github.com/OleguerCanal/KTH_RL-EL2805) but I and [Oleguer Canal](https://github.com/OleguerCanal) decided that it was going beyond the scope of that project and it could become useful for many others, as well as a good opportunity to learn.

Gaussian Process regression allows us to predict values of a function given some example pairs (x, y), where x is a N-dimensional input and y is a scalar value. Moreover, together with the value, we get the variance. In hyperparameter optimization, we often want to find the set of hyperparameters that maximizes a function (the score of a reinforcement learning agent, the accuracy of a classificatio, etc.). However, each evaluation is costly in terms of computational resources, and therefore we want to find this maximum using the lowest number of evaluations. With GP Regression, we can predict where this maximum is, and evaluate the function in that point. Moreover, by taking the variance into consideration, we can evaluate points that actually have the highest probability of being a maximum. 

#### Gaussian Process VS Grid Search
To test the effectiveness of the GP Regression search for hyperparameter optimization, we implemented a very simple classification task using the MNIST dataset. We use a Neural Network with a convolutional layer, a pooling layer, a dense layer and dropout. Therefore, one wants to find the hyperparameters [kernel\_size, pool\_size, dense\_units, drop\_rate] that maximize the accuracy on the test set. We therefore use Gaussian Process and Grid Search over the following parameter space:
- Kernel size: [3, 5, 7, 9]
- Pool size: [2, 3, 4]
- Dense units: [32, 64, 128, 256]
- Drop rates: [0.1, 0.2, 0.3, 0.4, 0.5]

We use Adam optimizer with Sparse Categorical Cross Entropy loss. The parameter space is given by the cartesian product of the lists above, has dimension 4 and the total number of hyperparameter combinations is 240. Each combination is effectively a vector in this parameter space. We start from the same vector [3, 2, 32, 0.1] for both Grid Search and Gaussian Process. Grid Search tries all the possible combinations in the order returned by itertools.product(). Gaussian Process instead always goes to the combination that has the highest predicted probability of being better than the ones already known. Every time we evaluate a combination of hyperparameters we train the model for one epoch. 

![alt text](https://github.com/fedetask/hyperparameter-optimization/blob/federico-dev/evaluation/figures/gp_vs_gs.png)

As the plot shows, Gaussian Process is able to find hyperparameters that maximize the accuracy from the very first iterations, while grid search takes about 150 evaluations to find an equivalently good one
