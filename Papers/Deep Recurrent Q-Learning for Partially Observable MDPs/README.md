# [Deep Recurrent Q-Learning For Partially Observable MDPs](https://arxiv.org/pdf/1507.06527.pdf)

## Summary

Investigating the effects of adding recurrency to a Deep Q-Network by replacing the first post-convolutional fully-connected layer with a recurrent LSTM resulting in Deep Recurrent Q-Network.

DRQN replicates DQNs performance on standard Atari games and partially observed equivalents featuring flickering game screens. 

DRQNs performance scales as a function of observability but when trained on full observations and evaluared with partial observations, DRQNs performance degrades less than DQNs.

## Background

Since DQN is trained using an input consisting of the last four states the agent has encountered, DQN will be unable to master games that require the player to remember events more distant than four screens in the past. 

Put differently, any game that requires a memory of more than four frames will appear non-Markovian because the future game states (and rewards) depend on more than just DQN’s current input. Instead of a Markov Decision Process (MDP), the game becomes a Partially-Observable Markov Decision Process (POMDP).

### Deep Q-Learning

Reinforcement Learning is concerned with learning control policies for agents interacting with unknown environments. The environments are often formalized as a Markov Decision Process described by a 4-tuple ![tuple](https://latex.codecogs.com/gif.latex?%5Cleft%20%28%20%5Cmathcal%7BS%2C%20A%2C%20P%2C%20R%7D%20%5Cright%20%29). At each timestep t, an agent interacting with the MDP observes a state ![state](https://latex.codecogs.com/gif.latex?s_t%20%5Cin%20%5Cmathcal%7BS%7D) and choses action ![action](https://latex.codecogs.com/gif.latex?a_t%20%5Cin%20%5Cmathcal%7BA%7D) which determines the reward ![reward](https://latex.codecogs.com/gif.latex?r_t%20%5Csim%20%5Cmathcal%7BR%7D%28s_t%2C%20a_t%29) and the next state ![next](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D%20%5Csim%20%5Cmathcal%7BP%7D%28s_t%2C%20a_t%29). 

Q-learning is a model-free off-policy algorithm for estimating long-term expected return of executing an action from a given state. A hight Q-value indicates an action a is judged to yeild better long-term results in a state s. Q-values estimate towards the observed reqard plus the max Q-value over all actions ![action](https://latex.codecogs.com/gif.latex?a%5E%5Cprime) in the resulting state ![state](https://latex.codecogs.com/gif.latex?s%5E%5Cprime):

![Q](https://latex.codecogs.com/gif.latex?Q%28s%2Ca%29%20%3A%3D%20Q%28s%2Ca%29%20&plus;%20%5Calpha%20%5Cleft%20%28%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7DQ%28s%5E%5Cprime%2Ca%5E%5Cprime%29%20-%20Q%28s%2Ca%29%20%5Cright%20%29)

The model is a neural network parameterized by weights and biases collectively denoted as ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta). Q-values are estimated by querying the output nodes of the network after performing a forward pass given a state input. Updates are made to the parameters of the network to minimize a differentiable loss function ![Loss](https://latex.codecogs.com/gif.latex?L%28s%2Ca%7C%5Ctheta_i%29%20%3D%20%5Cleft%20%28%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%28s%5E%5Cprime%2Ca%5E%5Cprime%7C%5Ctheta_i%29%20-%20Q%28s%2Ca%7C%5Ctheta_i%29%5E2%20%5Cright%20%29) as ![parameters](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bi&plus;1%7D%20%3D%20%5Ctheta_i%20&plus;%20%5Calpha%20%5Cnabla_%5Ctheta%20L%28%5Ctheta_i%29).

Deep Q-Learning uses three techniques to stabilize the learning
* Experiences ![experience](https://latex.codecogs.com/gif.latex?e_t%20%3D%20%28s_t%2C%20a_t%2C%20r_t%2C%20s_%7Bt&plus;1%7D%29) are recorded in a replay memory ![memory](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D) and then sampled uniformly. 
* A seperate target network ![Qhat](https://latex.codecogs.com/gif.latex?%5Chat%7BQ%7D) provides update targets to the main network
* An adaptive learning rate such as RMSProp which maintains a per-parameter learning rare and adjusts this to history of gradient updates.

### Partial Observability

A Partially Obervable Markov Decision Process (POMDP) captures the dynamics of many real-world environments by explicitly acknowledgubg the sensations received by the agent. A POMDP can be described as a 6-tuple ![tuple](https://latex.codecogs.com/gif.latex?%28%5Cmathcal%7BS%2C%20A%2C%20P%2C%20R%7D%2C%20%5COmega%2C%20%5Cmathcal%7BO%7D%29). ![states](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BS%2C%20A%2C%20P%2C%20R%7D) are the states, actions, transitions, and rewards as before, except now the agent is no longer privy to the true system state and instead receives an observation ![observation](https://latex.codecogs.com/gif.latex?o%20%5Cin%20%5COmega), which is generated from the underlying system state according to the probability distribution ![probability](https://latex.codecogs.com/gif.latex?o%20%5Csim%20%5Cmathcal%7BO%7D%28s%29).

Vanilla Deep Q-Learning has no explicit mechanisms for deciphering the underlying state of the POMDP.

## Architecture

![Layout](assets/Architecture.jpg)

The architecture of DQN is modified minimally by replacing the first fully-connected layer with a recurrent LSTM layer of the same size. The architecture of DRQN takes a single 84×84 processed image. The image is processed by three convolutional layers and the outputs are fed to the fully connected LSTM layer. Finally, a linear layer outputs a Q-Value for each action. 

During training, the parameters for both the convolutional and recurrent portions of the network are learned jointly from scratch. 

### Stable Recurrent Updates

Updating a recurrent, convolutional network requires each backward pass to contain many time-steps of game screens and target values. The LSTM's initial hidden states may be zero or carried from the previous values.

* **Bootstrapped Sequential Updates**: Episodes are selected randomly from the replay memory and updates begin at the beginning of the episode and proceed forward through time to the conclusion of the eposide. The targets at each timestep are generated from the target Q-network, ![Qhat](https://latex.codecogs.com/gif.latex?%5Chat%7BQ%7D). The RNN’s hidden state is carried forward throughout the episode.

* **Bootstrapped Random Updates**: Episodes are selected randomly from the replay memory and updates begin at random points in the episode and proceed for only unroll iterations timesteps (e.g. one backward call). The targets at each timestep are generated from the target Q-network, ![Qhat](https://latex.codecogs.com/gif.latex?%5Chat%7BQ%7D). The RNN’s initial state is zeroed at the start of the update.

Experiments indicate that both types of updates are viable and yield convergent policies with similar performance across a set of games.