# Hyperparameter optimization
The goal of this repo is to create a library of tools to estimate hyperparameters of machine learning algorithms. It's often the case that a ML algorithm depends on a high number of parameters, and finding their best combination becomes hard due to the number of those and the time cost of evaluating a single instance.

## What this repo contains:
#### Gaussian Process regression:
NOTE: This code was initially developed to help us estimating the best parameters in a [Reinforcement Learning assignment](https://github.com/OleguerCanal/KTH_RL-EL2805) but I and [Oleguer Canal](https://github.com/OleguerCanal) decided that it was going beyond the scope of that project and it could become useful for many others, as well as a good opportunity to learn.

Gaussian Process regression allows us to predict values of a function given some example pairs (x, y), where x is a N-dimensional input and y is a scalar value. Moreover, together with the value, we get the variance. In hyperparameter optimization, we often want to find the set of hyperparameters that maximizes a function (the score of a reinforcement learning agent, the accuracy of a classificatio, etc.). However, each evaluation is costly in terms of computational resources, and therefore we want to find this maximum using the lowest number of evaluations. With GP Regression, we can predict where this maximum is, and evaluate the function in that point. Moreover, by taking the variance into consideration, we can evaluate points that actually have the highest probability of being a maximum. 
