## Summary 

Function approximation is essential to reinforcement learning, but
the standard approach of approximating a value function and determining
a policy from it has so far proven theoretically intractable.

The limitations with the above methods are
Firstly , the policy that is obtained is a deterministic policy whereas in most cases , the policy needed is a stochastic one.

Secondly , an arbitary small change in the estimated value of an action can cause the action to be or not be selected.Such discontinous changes pose as an obstacle  for establishing convergence assurances.

An alternative approach to overcome the above difficulties is to make a function approximator(neural network) whose input is the state and the output is the action selection probablities and the weights are the weights are the policy parameters.

## Policy gradient theorm :

Let 

$$	\rho(\pi) =  \sum _{s}{d^\pi}(s)\sum_{a}{\pi}(s,a)\mathbb{R}^a_s $$

where  $$ \mathbb{R}^a_s = E(r_{t+1}|s_t = s,a_t = a) , \forall s,s' \in S, a \in A $$ is the average expected reward 

and 
 $$d^\pi(s) = lim_{t\to\infty}(Pr(s_t = s)|so,\pi)$$
is the stationary of states under $\pi$ which is independent of $so$ for all policies

Policy gradient theorm:
For any MDP, in either the average-reward or
start-state formulations,
 
$$ \frac{\partial \rho}{\partial \theta} = \sum_{s}d^{\pi}(s) \sum_{a}\frac{\partial {\pi(s,a)}}{\partial \theta}{Q^{\pi}}(s,a) $$
 
## Policy gradient approximation :

In the above equation , Q_{pi} is not normally and is approximated by a function approximator (neural network) 


Let
$$f_{w}: S \times A \to \mathbb{R}$$
If  

$$\frac{\partial f_{w}}{\partial w} = \frac{\partial {\pi(s,a)}}{\partial \theta}\frac{1}{\pi(s,a)}$$

then 

 $$\frac{\partial \rho}{\partial \theta} = \sum_{s}d^{\pi}(s) \sum_{a}\frac{\partial {\pi(s,a)}}{\partial \theta}{f_{w}(s,a)}$$
The above however doesnt necessarily mean ${f}_{w}$ should be approximately the same as $Q(s,a)$ , it can vary by a extra bias term which is inpendent of the action taken and hence the expectation of whose is equal to zero .
