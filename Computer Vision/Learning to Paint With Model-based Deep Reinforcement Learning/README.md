# Learning to Paint With Model-based Deep Reinforcement Learning

### Pytorch Implementation of the model : [Painting with RL](https://github.com/Vinayak-VG/My-Projects/tree/main/Reinforcement%20Learning/Painting%20with%20RL)

## Summary 

### Introduction

Teaching a machine how to paint using only strokes is a difficult task since it needs to decide both on past history as well as for future strokes. To make the model decide based on future rewards, the authors decide to use model-based Deep Reinforcement Learning since it helps the model to learn using the future outcomes as well as using the history. They use a neural renderer to simulate the canvas for the model so that it can look into every possible outcome for a single action. Basically on each stroke on the canvas, the agent has to predict the stroke location, shape, stroke thickness and colour. They define a differentiable neural renderer that can back-propagate gradients for the model to learn. It does not require any supervision since it learns on its own using the simulation. This paper is a combination of GAN + RL

### Previous Work

  * Synthesizing programs for images using reinforced adversarial learning: [Paper](https://arxiv.org/pdf/1804.01118.pdf)

### Model

<img width="770" alt="Screenshot 2021-08-04 at 10 50 13 PM" src="https://user-images.githubusercontent.com/80670240/128236682-2b5ce3e7-48de-4a2f-97c3-997da5ad9bee.png">

State space in this environment is made up of 3 parts: The current state of canvas, target Image and step number. 
The next state is defined by s(t+1) = trans(s(t), a(t)), where the trans function is the function which applies the action a(t) on the current canvas state s(t)

The action space of the agent is made up of 4 parameters namely colour, shape, thickness and location. Basically the agent acts according to a policy function ᴨ which maps from state space to action space. 

The reward is modelled such that it can measure the difference between current canvas and the next canvas. 
r(s(t),a(t))=L(t) −L(t+1) where L(t) is the loss between the current canvas and the target image and L(t+1) is the loss between the next canvas and the target image. To make the final canvas resemble the target image, the agent should be driven to maximize the cumulative rewards in the whole episode.

We use Deep Deterministic Policy Gradient(DDPG) to help the actor and the critic develop a best policy and also to predict the estimated reward correctly respectively. Usually the data is actually the data from the replay buffer but it would be tough for the agent to just learn from the replay buffer since it has to model complex structures and hence they design a neural renderer so that the agent can observe a modeled environment. Then it can explore the environment and improve its policy efficiently

<img width="776" alt="Screenshot 2021-08-04 at 10 50 35 PM" src="https://user-images.githubusercontent.com/80670240/128236508-58e4f06e-b8ff-4d24-bc56-b63b3eb4751b.png">

At step t, the critic takes s(t+1) as input rather than both of s(t) and a(t). The critic still predicts the expected reward for the state but no longer includes the reward caused by the current action. The new expected reward is a value function V(s(t)) trained using discounted reward:

`V (s(t)) = r(s(t), a(t)) + γV (s(t+1))`

Action Bundle: To improve accuracy and also to decrease computation cost, they propose action bundle. Rather than the agent observing the environment and perform the action each frame. We could let the agent observe the environment for k frames using the same previous action and then after k frames it acts. This practice encourages the exploration of the action space and action 
combinations. The renderer can easily render k strokes at a time rather than k strokes every single time individually. 

They use Wasserstein GAN loss to predict the reward for every time step. The objective or loss for WGAN is defined as follows

<img width="243" alt="Screenshot 2021-08-04 at 11 20 25 PM" src="https://user-images.githubusercontent.com/80670240/128236941-776d5091-b7c1-4155-8ab2-2c11129011e2.png">

The fake examples are the pairs of the canvas paintings by the agent and the target image whereas the real images are the pairs of same target image. They also constraint that D should be under the constraints of 1-Lipschitz.

<img width="295" alt="Screenshot 2021-08-04 at 11 20 43 PM" src="https://user-images.githubusercontent.com/80670240/128236950-7e47de60-018c-4d5f-9fba-2cee38a5eca0.png">
This is the loss used to calculate the reward which we saw earlier. 

Neural Render: The neural renderer is made up of some fc layers and sub pixel upsampling layers where the input is the stroke parameters and the output is the rendered stroke image. 

<img width="403" alt="Screenshot 2021-08-04 at 11 26 04 PM" src="https://user-images.githubusercontent.com/80670240/128237059-9ed077b7-ffb7-4025-92c0-b7aa6980b468.png">

The stroke design they choose is Quadratic Bezier curve(QBC) with thickness to simulate the effect of brush. The stroke is defined as:

<img width="599" alt="Screenshot 2021-08-04 at 11 35 05 PM" src="https://user-images.githubusercontent.com/80670240/128237066-c0fff45a-5e01-4f39-8f65-cd7c82e9f86b.png">

where (x0,y0,x1,y1,x2,y2) are the coordinates of the three control points of the QBC. (r0, t0), (r1, t1) control the thickness and transparency of the two endpoints of the curve, respectively. (R, G, B) controls the color. The formula of QBC is

<img width="626" alt="Screenshot 2021-08-04 at 11 35 27 PM" src="https://user-images.githubusercontent.com/80670240/128237093-5580ed37-3d34-48c3-82bf-b771a9c2903c.png">

### Dataset

  * MNIST
  * SVHN
  * CelebA
  * ImageNet

### Training

To learn more about the training paramerters and stuff check out the paper. They have nicely explained the architecture with neat diagrams and have given the hyperparameters very organised

### Results

<img width="1135" alt="Screenshot 2021-08-04 at 11 57 00 PM" src="https://user-images.githubusercontent.com/80670240/128237110-b0fd349b-1afb-4f01-926d-f5d7848a65f5.png">

### End Note

To know more about Painting with RL, check out the paper: [Painting with RL](https://arxiv.org/pdf/1903.04411.pdf)

To check out the Pytorch Implementation of the model, check out my GitHub Repo: [GitHub Repo](https://github.com/Vinayak-VG/My-Projects/tree/main/Reinforcement%20Learning/Painting%20with%20RL)


---

[Vinayak Gupta](https://github.com/Vinayak-VG)
4th August 2021

