# Asynchronous Methods For Deep Reinforcement Learning

## Contents

* [Paper](Paper.pdf)

## Summary

A conceptually simple and lightweight framework for Deep Reinforcement Learning that uses Asynchronous Gradient Descent for Optimization of Deep Neural Network Controllers.

The Asynchronous Variant of Actor-Critic surpasses the current State-of-the-art on Atari domain.

## Background

The standard reinforcement learning setting where an agents interacts with an environment ![e](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BE%7D) over a number of discrete time steps. At each time step t, the agent receives a state ![state](https://latex.codecogs.com/gif.latex?s_t) and selects an action ![action](https://latex.codecogs.com/gif.latex?a_t) from a set of possible actions ![actions](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D) according to the its policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi), where ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) is a mapping from states ![state](https://latex.codecogs.com/gif.latex?s_t) to actions ![action](https://latex.codecogs.com/gif.latex?a_t). In return the agent receives the next state ![state](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D) and receives a scalar reward ![reward](https://latex.codecogs.com/gif.latex?r_t). 

The return ![return](https://latex.codecogs.com/gif.latex?R_t%20%3D%20%5Csum_%7Bk%3D0%7D%5E%7B%5Cinfty%7D%20%5Cgamma%5Ek%20r_%7Bt&plus;k%7D) is the total accumulated return from the time step t with a discount factor ![gamma](https://latex.codecogs.com/gif.latex?%5Cgamma%20%5Cin%20%280%2C1%5D). The goal of the agent is to maximize the expected return from each state ![state](https://latex.codecogs.com/gif.latex?s_t).

The action value ![Q](https://latex.codecogs.com/gif.latex?Q%5E%7B%5Cpi%7D%5Cleft%20%28%20s%2Ca%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%5B%20R_t%7Cs_t%3Ds%2Ca%20%5Cright%20%5D) is the expected return for selecting an action a in the state s and following policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi). The optimal value function ![Q_optimal](https://latex.codecogs.com/gif.latex?Q%5E*%5Cleft%20%28%20s%2Ca%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Cpi%7D%20Q%5E%7B%5Cpi%7D%5Cleft%20%28%20s%2Ca%20%5Cright%20%29) gives the maximum action value for state s and action a achievable by any policy.

Similarly the value function of the state s under policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) is defined as ![value](https://latex.codecogs.com/gif.latex?V%5E%5Cpi%20%5Cleft%20%28s%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%5B%20R_t%7Cs_t%3Ds%20%5Cright%20%5D) and is the expected return for following the policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) from the state s.

### Value-Based Model-Free Reinforcement Learning

In value-based model-free reinforcement learning methods, the action value function is represented using a function approximator, such as a neural network. Let ![Q](https://latex.codecogs.com/gif.latex?Q%5Cleft%20%28%20s%2Ca%3B%5Ctheta%20%5Cright%20%29) be an approximate action-value function with parameters ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta). The updates to ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) can be derived from a variety of reinforcement learning algorithms. The parameters of the action value function are learned iteratively minimizing a sequence of loss functions ![loss](https://latex.codecogs.com/gif.latex?L_i%5Cleft%20%28%20%5Ctheta_i%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%28%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%5Cleft%20%28%20s%5E%5Cprime%2C%20a%5E%5Cprime%3B%20%5Ctheta_%7Bi-1%7D%20%5Cright%20%29%20-Q%5Cleft%20%28%20s%2Ca%3B%5Ctheta_i%20%5Cright%20%29%20%5Cright%20%29%5E2) where ![s'](https://latex.codecogs.com/gif.latex?s%5E%5Cprime) is encountered after state s.

In one-step methods is that obtaining a reward r only directly affects the value of the state action pair s, a that led to the reward. The values of other state action pairs are affected only indirectly through the updated value ![Q](https://latex.codecogs.com/gif.latex?Q%5Cleft%20%28%20s%2Ca%20%5Cright%20%29). This can make the learning process slow since many updates are required the propagate a reward to the relevant preceding states and actions.

One way of propagating rewards faster is by using n-step returns. In n-step Q-learning, ![Q](https://latex.codecogs.com/gif.latex?Q%5Cleft%20%28%20s%2Ca%20%5Cright%20%29) is updated toward the n-step return defined as ![nstep](https://latex.codecogs.com/gif.latex?r_t%20&plus;%20%5Cgamma%20r_%7Bt&plus;1%7D%20&plus;%20%5Cdots%20&plus;%20%5Cgamma%5E%7Bn-1%7D%20r_%7Bt&plus;n-1%7D%20&plus;%20%5Cmax_a%20%5Cgamma%5En%20Q%28s_%7Bt&plus;n%7D%2C%20a%29). This results in single reward directly affecting the values of n preceding state action pairs.

### Policy-Based Model-Free Reinforcement Learning

In policy-based model-free methods directly parameterize the policy ![policy](https://latex.codecogs.com/gif.latex?%5Cpi%5Cleft%20%28%20a%2Cs%3B%5Ctheta%20%5Cright%20%29) and update the parameters ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) by performing approximate gradient ascent on ![E](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D%5Cleft%20%5B%20R_t%20%5Cright%20%5D).

The standard Reinforce updates the policy parameters ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) in the direction ![direction](https://latex.codecogs.com/gif.latex?%5Cnabla_%5Ctheta%20%5Clog%20%5Cpi%20%5Cleft%20%28%20a_t%7Cs_t%3B%20%5Ctheta%20%5Cright%20%29%20R_t).


## Asynchronous RL Framework

A multi-threaded asynchronous variants of one-step Sarsa, one-step Q-learning, n-step Q-learning and advantage actor-critic.

* Asynchronous actor-learners are used on a single machine with multiple CPU threads, reducing the communication costs.

* Observations are made on multiple actor learners running in parallel, exploring different parts of the environment. 

By running different exploration policies, the changes made to the parameters by multiple actor learners applying online updates in parallel are likely to be less correlated that a single agent applying online updates. 

### Asynchronous One-Step Q-Learning

Each thread interacts with its own copy of the environment and at each step, computes a gradient of Q-learning loss. A shared and slowly changing target network in computing the Q-learning loss.

The gradients are accumulated over multiple timesteps before they are applied.

### Asynchronous One-Step Sarsa

Same as the asynchronous one-step Q-learning which uses a different target value ![target](https://latex.codecogs.com/gif.latex?r%20&plus;%20%5Cgamma%20Q%5Cleft%20%28%20s%5E%5Cprime%2C%20a%5E%5Cprime%3B%20%5Ctheta%5E-%20%5Cright%20%29).

### Asynchronous n-step Q-Learning

The algorithm operates in the forward view by explicitly computing n-step returns, as opposed to the more common backward view used by techniques like eligibility traces. The algorithm first selects actions using its exploration policy for up to ![tmax](https://latex.codecogs.com/gif.latex?t_%7Bmax%7D) steps. The process results in the agent receiving up to ![tmax](https://latex.codecogs.com/gif.latex?t_%7Bmax%7D) rewards from the environment.

### Asynchronous Advantage Actor-Critic

THe asynchronous advantage actor-critic maintains a policy ![policy](https://latex.codecogs.com/gif.latex?%5Cpi%28a_t%7Cs_t%3B%5Ctheta%29) and an estimate of the value function ![value](https://latex.codecogs.com/gif.latex?V%5Cleft%20%28%20s_t%3B%5Ctheta_v%20%5Cright%20%29).. The policy and the value function are updated after every ![tmax](https://latex.codecogs.com/gif.latex?t_%7Bmax%7D) actions. The update performed by the algorithm can be seen as ![update](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%5E%5Cprime%7D%20%5Clog%20%5Cpi%20%5Cleft%20%28%20a_t%7Cs_t%2C%20%5Ctheta%5E%5Cprime%20%5Cright%20%29%20A%20%5Cleft%20%28%20s_t%2C%20a_t%3B%20%5Ctheta%2C%20%5Ctheta_v%20%5Cright%20%29) where ![action](https://latex.codecogs.com/gif.latex?A%20%5Cleft%20%28%20s_t%2C%20a_t%3B%20%5Ctheta%2C%20%5Ctheta_v%20%5Cright%20%29) is an estimate of the advantage function ![advantage](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D0%7D%5E%7Bk-1%7D%20%5Cgamma%5Ei%20r_%7Bt&plus;i%7D%20&plus;%20%5Cgamma%5Ek%20V%5Cleft%20%28%20s_%7Bt&plus;k%7D%3B%20%5Ctheta_v%20%5Cright%20%29%20-%20V%5Cleft%20%28%20s_t%3B%20%5Ctheta_v%20%5Cright%20%29) where k can vary from state to state and is upper-bounded by ![tmax](https://latex.codecogs.com/gif.latex?t_%7Bmax%7D).

As with the value-based methods, a parallel actor-learners and accumulated updates for improving training stability. A convolutional neural network that has one softmax output for the policy ![policy](https://latex.codecogs.com/gif.latex?%5Cpi%20%5Cleft%20%28%20a_t%20%7Cs_t%3B%20%5Ctheta%20%5Cright%20%29) and one linear output for the value function ![value](https://latex.codecogs.com/gif.latex?V%5Cleft%20%28%20s_t%3B%20%5Ctheta_v%20%5Cright%20%29).

### Optimization

The standard non-centered RMSProp update is perormed elementwise.

![RMS](https://latex.codecogs.com/gif.latex?g%20%3D%20%5Calpha%20g%20&plus;%20%5Cleft%20%28%201-%5Calpha%20%5Cright%20%29g%20%5CDelta%20%5Ctheta%5E2%20%5Ctextup%7B%20and%20%7D%20%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%20%5Ceta%20%5Cfrac%7B%5CDelta%20%5Ctheta%7D%7B%5Csqrt%7Bg&plus;%5Cepsilon%7D%7D)