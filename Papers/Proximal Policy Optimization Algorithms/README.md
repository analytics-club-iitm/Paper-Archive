# [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

## Summary

A new family of policy gradient methods for reinforcement learning which alternate between data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent.

Q-learning fails on many simple problems and is poorly understood, vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing.


Standard policy gradient methods perform one gradient update per data sample, but a novel objective function is proposed that enables multiple epochs of minibatch updates with clipped probability ratios. To optimize policies, sampling data from the policy and performed several epochs of optimizations are alternated.

## Background: Policy Optimization

### Policy Gradient Methods

Policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm. The most commonly used gradient estimator has the form ![policy](https://latex.codecogs.com/gif.latex?%5Chat%7Bg%7D%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cnabla_%5Ctheta%20%5Clog%20%5Cpi_%5Ctheta%20%28a_t%7Cs_t%29%20%5Chat%7BA%7D_t%20%5Cright%20%5D), where ![pi_theta](https://latex.codecogs.com/gif.latex?%5Cpi_%5Ctheta) is a stochastic policy and ![Ahat](https://latex.codecogs.com/gif.latex?%5Chat%7BA%7D) is an estimator of the advantage function at timestep t. The expectation indicates the empirical average over a finite batch of samples.

An objective function whose gradient is the policy gradient estimator is constructed and the estimator is obtained by differentiating the objective ![objective](https://latex.codecogs.com/gif.latex?L%5E%7BPG%7D%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Clog%20%5Cpi_%5Ctheta%20%28a_t%7Cs_t%29%20%5Chat%7BA%7D_t%20%5Cright%20%5D). 


### Trust Region Methods

In TRPOm an obective function is maximized subject to a constraint on the size of policy update. Specifically, ![objective](https://latex.codecogs.com/gif.latex?%5Cmax_%7B%5Ctheta%7D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cfrac%7B%5Cpi_%5Ctheta%20%28a_t%20%7C%20s_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%20%28a_t%20%7C%20s_t%29%7D%20%5Chat%7BA%7D_t%20%5Cright%20%5D) subject to ![inequality](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Ctextup%7BKL%7D%5Cleft%20%5B%20%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%7Cs_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%7Cs_t%29%20%5Cright%20%5D%20%5Cright%20%5D%20%5Cleq%20%5Cdelta).

The ![theta_old](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bold%7D) is the vector of policy parameters before the update. This problem can efficiently be approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint.

The theory justifying TRPO actually suggests using a penalty instead of a constraint, i.e.,
solving the unconstrained optimization problem, ![penalty](https://latex.codecogs.com/gif.latex?%5Cmax_%7B%5Ctheta%7D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cfrac%7B%5Cpi_%7B%5Ctheta%7D%28a_t%7Cs_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28a_t%7Cs_t%29%7D%20%5Chat%7BA%7D_t%20-%20%5Cbeta%20%5Ctextup%7BKL%7D%20%5Cleft%20%5B%20%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%7Cs_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%7Cs_t%29%20%5Cright%20%5D%20%5Cright%20%5D) for some coefficient ![beta](https://latex.codecogs.com/gif.latex?%5Cbeta). This follows from the fact that a certain surrogate objective forms a lower bound on the policy.

### Clipped Surrogate Objective

Let ![probability](https://latex.codecogs.com/gif.latex?r_t%28%5Ctheta%29) denote the probability ratio ![ratio](https://latex.codecogs.com/gif.latex?r_t%28%5Ctheta%29%20%3D%20%5Cfrac%7B%5Cpi_%7B%5Ctheta%7D%28a_t%7Cs_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28a_t%7Cs_t%29%7D). TRPO maximizes a "surrogate" objective:

![Objective](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BCPI%7D%7D%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cfrac%7B%5Cpi_%7B%5Ctheta%7D%28a_t%7Cs_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28a_t%7Cs_t%29%7D%20%5Chat%7BA%7D_t%20%5Cright%20%5D%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20r_t%28%5Ctheta%29%20%5Chat%7BA%7D_t%20%5Cright%20%5D)

The superscript CPI refers to conservative policy iteration. Without a constraint, maximization of ![L](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BCPI%7D%7D) would lead to a excessive large policy update. The main objective therefore is ![main](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BCPIP%7D%7D%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cmin%20%5Cleft%20%28%20r_t%28%5Ctheta%29%5Chat%7BA%7D_t%2C%20%5Ctextup%7Bclip%7D%28r_t%28%5Ctheta%29%2C%201-%5Cepsilon%2C%201&plus;%5Cepsilon%29%5Chat%7BA%7D_t%20%5Cright%20%29%5Cright%20%5D), where ![epsilon](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%3D%200.2) is a hyperparameter.

The clip modifies the surrogate objectibe by clipping the proabability ratio, which removes the incentive for moving ![reward](https://latex.codecogs.com/gif.latex?r_t) outside of the interval ![interval](https://latex.codecogs.com/gif.latex?r_t). Finally, we take the minimum of the clipped and unclipped objective, so the final objective is a lower bound on the unclipped objective.

### Adaptive KL Penalty Coefficient

A KL divergance can be used to adapt the penalty coefficient, to achieve target value of the KL divergance each policy update. 

* Use several epochs of minibatch SGD, optimize the KL-penalized objective ![objective](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BKLPEN%7D%7D%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%20%5B%20%5Cfrac%7B%5Cpi_%7B%5Ctheta%7D%20%28a_t%7Cs_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%20%28a_t%7Cs_t%29%7D%20%5Chat%7BA%7D_t%20-%20%5Cbeta%20%5Ctextup%7BKL%7D%20%5Cleft%5B%20%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%7Cs_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%7Cs_t%29%20%5Cright%20%5D%20%5Cright%20%5D)

* Compute ![d](https://latex.codecogs.com/gif.latex?d%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%5B%20%5Ctextup%7BKL%7D%20%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%7Cs_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%7Cs_t%29%20%5Cright%20%5D) 
	- If ![update](https://latex.codecogs.com/gif.latex?d%20%3C%20d_%7Btarg%7D/1.5%2C%20%5Cbeta%20%5Cleftarrow%20%5Cbeta/2)
	- If ![update](https://latex.codecogs.com/gif.latex?d%20%3E%20d_%7Btarg%7D%20%5Ctimes%201.5%2C%20%5Cbeta%20%5Cleftarrow%20%5Cbeta%20%5Ctimes%202)

The updated beta is used for next policy update.


## Algorithm

The surrogate losses from the previous sections can be computed and differentiated with a minor change to a typical policy gradient implementation. Most techniques for computing variance-reduced advantage-function estimators make use a learned state-value function ![value](https://latex.codecogs.com/gif.latex?V%28s%29). 

If using a neural network architecture that shares parameters between the policy and value function, a loss funciton that combines the policy surrogate must be used. This objective can further be augmented by adding an entropy bonus to ensure sufficient exploration. Combining the terms, the following objective is obained, which is (approximately) maximized each iteration: ![combined](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BCLIP%7D&plus;%5Ctextup%7BVF%7D&plus;%5Ctextup%7BS%7D%7D_t%20%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%5B%20L%5E%7B%5Ctextup%7BCLIP%7D%7D_t%20%28%5Ctheta%29%20-%20c_1%20L%5E%7B%5Ctextup%7BVF%7D%7D_t%20%28%5Ctheta%29%20&plus;%20c_2%20S%5B%5Cpi_%5Ctheta%5D%28s_t%29%20%5Cright%20%5D) where ![c](https://latex.codecogs.com/gif.latex?c_1%2C%20c_2) are coefficients and S denotes entropy bonus and ![LVF](https://latex.codecogs.com/gif.latex?L%5E%7B%5Ctextup%7BVF%7D%7D) is a squared-error loss ![loss](https://latex.codecogs.com/gif.latex?%5Cleft%20%28V_%5Ctheta%28s_t%29%20-%20V_t%5E%7Btarg%7D%20%5Cright%20%29%5E2).

The advantage estimator runs the policy for T timesteps and is given by ![estimator](https://latex.codecogs.com/gif.latex?%5Chat%7BA%7D_t%20%3D%20-V%28s_t%29%20&plus;%20r_t%20&plus;%20%5Cgamma%20r_%7Bt&plus;1%7D%20&plus;%20%5Cdots%20&plus;%20%5Cgamma%5E%7BT-t&plus;1%7D%20r_%7BT-1%7D%20&plus;%20%5Cgamma%5E%7BT-t%7D%20V%28s_T%29). Generalizing this choice to a truncated version of generalized advantage estimation, we get ![estimator](https://latex.codecogs.com/gif.latex?%5Chat%7BA%7D_t%20%3D%20%5Cdelta_t%20&plus;%20%28%5Clambda%20%5Cgamma%29%20%5Cdelta_%7Bt&plus;1%7D%20&plus;%20%5Cdots%20&plus;%20%28%5Clambda%20%5Cgamma%29%5E%7BT-t&plus;1%7D%20%5Cdelta_%7BT-1%7D) where ![delta](https://latex.codecogs.com/gif.latex?%5Cdelta_t%20%3D%20r_t%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%29%20-%20V%28s_t%29).

A proximal policy optimization (PPO) algorithm that uses fixed-length trajectory segments. Each iteration, each of N parallel actors collect T timesteps of data. A surrogate loss is constructed on these NT timesteps and optimized using SGD for K epochs.




