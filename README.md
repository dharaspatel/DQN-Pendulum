# Optimization of Mechanical Motion 
  Sander Miller and Dhara Patel 
  
  Franklin W. Olin College of Engineering, Machine Learning 
  
## Background 
In the world of robotics, precision is key. To achieve this order of precision, there are infinite parameters we need to optimize. Using reinforcement learning we can optimize a robot’s performance within minutes. Reinforcement learning uses experiences to learn first-hand what the best way of achieving a particular goal is. It assigns rewards to actions that lead to more optimal states. This model can be applied to any environment. Any robotic motion can use reinforcement learning to optimize. 


#### Our Environment: Inverted Pendulum

We used [OpenAi Gym’s ‘Carpole v1’ environment](https://gym.openai.com/envs/CartPole-v1/). It consists of an inverted pendulum mounted on a cart that can move left or right on a frictionless cart. The system is controlled by applying a force of +/- 1 on the cart. The goal of the cart (agent) is to keep the pendulum upright.

## Model Architechture 
A Deep Q-Network (DQN) is a reinforcement learning model that evaluates the benefit of a particular action given the current state and its future states.

![A System Diagram of DQN](https://pathmind.com/images/wiki/simple_RL_schema.png)

A feedback loop allows it to learn a "pathway" that maximizes its reward. For every iteration, an agent (the cart) takes an action based on a model prediction and executes it on the environment. The environment then returns a reward for that action. This process is repeated until termination 

![Equation for DQN](https://miro.medium.com/max/1434/1*CLBIXdpk8ft0-1MFH8FwUg.png)

The Q-value, which is a cummulation of discounted future rewards, is what allows learning to happen. 

```code
```
### Optimization 

## Results 

## Analysis 

## Next Steps
