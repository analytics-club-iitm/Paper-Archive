# Playing Atari With Deep Reinforcement Learning

## Contents

* [Paper](Paper.pdf)

## Summary

Deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. 

The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

## Background

The agent interacts with an environment ![E](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BE%7D) in a sequence of actions, observations and reqards. At each time-step, the agent selects an action ![action](https://latex.codecogs.com/gif.latex?a_t) from the set of legal game actions, ![actions](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%20%3D%20%5Cleft%20%5C%7B%201%2C%20%5Cdots%2C%20K%20%5Cright%20%5C%7D). The action is passed to the emulator and modifies its internal state and game score. The emulator's internal state is not observed by the agent; instead it observes an image ![image](https://latex.codecogs.com/gif.latex?x_t%20%5Cin%20%5Cmathbb%7BR%7D%5Ed) from the emulator, which is a vector of raw pixel values representing the current screen.In addition it receives a reward ![reward](![action](https://latex.codecogs.com/gif.latex?r_t)representing the change in game score.

Since the agent only observes images of the current screen, the task is partially observed and many emulator states are perceptually aliased. Therefore a sequence of actions and observations ![history](https://latex.codecogs.com/gif.latex?s_t%20%3D%20x_1%2C%20a_1%2C%20%5Cdots%2C%20a_%7Bt-1%7D%2C%20x_t) and learn game strategies that depend upon these sequences.

The goal of the agent is to interact with the emulator by selecting actions in a way that maximises future rewards. The standard assumptions are that the future rewards are discounted by a factor of ![gamma](https://latex.codecogs.com/gif.latex?%5Cgamma) per time-step, and the discounted return at time t is ![reward](https://latex.codecogs.com/gif.latex?R_t%20%3D%20%5Csum_%7Bt%5E%5Cprime%3Dt%7D%5E%7BT%7D%20%5Cgamma%5E%7Bt%5E%5Cprime-t%7Dr_%7Bt%5E%5Cprime%7D). 

The optimal action-alue function ![Q](https://latex.codecogs.com/gif.latex?Q%5E*%28s%2Ca%29) is the maximum expected return achievable by following any strategy, and then taking some action a, ![Q](https://latex.codecogs.com/gif.latex?Q%5E*%28s%2Ca%29%20%3D%20%5Cmax_%5Cpi%20%5Cmathbb%7BE%7D%20%5Cleft%20%5B%20s_t%3Ds%2C%20a_t%3Da%2C%20%5Cpi%20%5Cright%20%5D), where ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) is a policy mapping sequences to actions.

The optimal action-value function obeys an important identity known as the Bellman equation. This is based on the following intution: if the optimal value ![Q](https://latex.codecogs.com/gif.latex?Q%5E*%28s%5E%5Cprime%2C%20a%5E%5Cprime%29) of the sequence ![s](https://latex.codecogs.com/gif.latex?s%5E%5Cprime) at the next time-step for all possile actions ![a](https://latex.codecogs.com/gif.latex?a%5E%5Cprime), then the optimal strategy is to select the action maximising the expected value of ![expected](https://latex.codecogs.com/gif.latex?r&plus;%5Cgamma%20Q%5E*%28s%5E%5Cprime%2C%20a%5E%5Cprime%29), ![bellman](https://latex.codecogs.com/gif.latex?Q%5E*%28s%5E%5Cprime%2C%20a%5E%5Cprime%29%20%3D%20%5Cmathbb%7BE%7D_%7Bs%5E%5Cprime%20%5Csim%20%5Cmathcal%7BE%7D%7D%20%5Cleft%20%5B%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%5E*%28s%5E%5Cprime%2C%20a%5E%5Cprime%29%20%7C%20s%2Ca%20%5Cright%20%5D).

The basic idea behind many reinforcement learning algorithms is to estimate the action-value function, by using the Bellman equation as an iterative update ![Q](https://latex.codecogs.com/gif.latex?Q_%7Bi&plus;1%7D%28s%5E%5Cprime%2C%20a%5E%5Cprime%29%20%3D%20%5Cmathbb%7BE%7D_%7Bs%5E%5Cprime%20%5Csim%20%5Cmathcal%7BE%7D%7D%20%5Cleft%20%5B%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q_i%28s%5E%5Cprime%2C%20a%5E%5Cprime%29%20%7C%20s%2Ca%20%5Cright%20%5D). Such value iteration algorithms converge to the optimal action- value function, ![converge](https://latex.codecogs.com/gif.latex?Q_i%20%5Crightarrow%20Q%5E*%20%5Ctextup%7B%20as%20%7D%20i%5Crightarrow%20%5Cinfty).

A function approximator is used to estimate the action-value function ![function](https://latex.codecogs.com/gif.latex?Q%28s%2Ca%3B%5Ctheta%29%20%3D%20Q%5E*%28s%2Ca%29). A neural network function approximator with weights ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) as a Q-neetwork can be trained by minimising a sequence of Loss function ![Loss](https://latex.codecogs.com/gif.latex?L_i%28%5Ctheta_i%29) that changes at each iteration i, ![Loss](https://latex.codecogs.com/gif.latex?L_i%28%5Ctheta_i%29%20%3D%20%5Cmathbb%7BE%7D_%7Bs%2C%20a%20%5Csim%20%5Crho%5Cleft%20%28%20%5Ccdot%20%5Cright%20%29%7D%20%5Cleft%20%5B%20%5Cleft%20%28%20y_i%20-%20Q%28s%2Ca%3B%20%5Ctheta_i%29%20%5Cright%20%29%5E2%20%5Cright%20%5D), where ![y](https://latex.codecogs.com/gif.latex?y_i%20%3D%20%5Cmathbb%7BE%7D_%7Bs%5E%5Cprime%20%5Csim%20%5Cmathcal%7BE%7D%7D%20%5Cleft%20%5Br%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%28s%5E%5Cprime%2C%20a%5E%5Cprime%3B%20%5Ctheta_%7Bi-1%7D%29%20-%20Q%28s%2C%20a%3B%20%5Ctheta_i%29%20%7C%20s%2Ca%20%5Cright%5D) is the target for iteration i and ![rho](https://latex.codecogs.com/gif.latex?%5Crho%28s%2Ca%29) is a probability distribution over sequences s and actions a that is refered to as behavious distribution.


The gardient is obtained by differentiating the loss function with respect to the weights, 
![gradient](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta_i%7DL_i%28%5Ctheta_i%29%20%3D%20%5Cmathbb%7BE%7D_%7Bs%2Ca%20%5Csim%20%5Crho%28%5Ccdot%29%3Bs%5E%5Cprime%20%5Csim%20%5Cmathcal%7BE%7D%7D%20%5Cleft%20%5B%20%5Cleft%20%28%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%28s%5E%5Cprime%2C%20a%5E%5Cprime%3B%20%5Ctheta_%7Bi-1%7D%29%20-%20Q%28s%2C%20a%3B%20%5Ctheta_i%29%20%5Cright%29%20%5Cnabla_%7B%5Ctheta_i%7D%20Q%28s%2C%20a%3B%20%5Ctheta_i%29%20%5Cright%20%5D).

## Approach

A technique known as experience replay where the agent’s experiences ![e](https://latex.codecogs.com/gif.latex?e_t%20%3D%20%28s_t%2C%20a_t%2C%20r_t%2C%20s_%7Bt&plus;1%7D%29) in ![D](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D%20%3D%20e_1%2C%20%5Cdots%2C%20e_n) at each time-step are stored and pooled over many episodes into a replay memory. 

During the inner loop of the algorithm, Q-learning updates are applied, or minibatch updates, to samples of experience, ![eD](https://latex.codecogs.com/gif.latex?e%20%5Csim%20%5Cmathcal%7BD%7D), drawn at random from the pool of stored samples. After performing experience replay, the agent selects and executes an action according to an ![epsilon](https://latex.codecogs.com/gif.latex?%5Cepsilon)-greedy policy.

### Advantages

* First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency.
* Second, learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randomizing the samples breaks these correlations and therefore reduces the variance of the updates.
* Third, when learning on-policy the current parameters determine the next data sample that the parameters are trained on.

### Architecture

The input to the neural network consists is an 84×84×4. The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity. The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity. The final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fully-connected linear layer with a single output for each valid action.