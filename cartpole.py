import gym

def cartpole():
    #environment - cartpole
    env = gym.make("CartPole-v1")

    #observational space - possible state values
    observational_space = env.observational_space.shape[0]

    #action space - possible actions that can be performed
    action_space = env.action_space.n

    #agent 
    agent = DQNSolver(observational_space, action_space)

    while True:
        #reset state at once terminal environment is reached 
        state = env.reset
        state = np.reshape(state, [1, observational_space])

        while True:
            #update environment 
            env.render()

            #determine action 
            action = agent.act(state)

            #determine new state and corresponding reward 
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
            reward = reward if not terminal else -reward

            #remember to learn - used in experience replay 
            agent.remember(state, action, reward, state_next, terminal)

            #experience replay - analyzes what happened and updates Q
            agent.experience_replay()

            #set future state as current state
            state = state_next

            #check if terminal state has been reached 
            if terminal break


    
class DQNSolver:
    """
        attributes: 
    """
    def __init__():
    def remember():
    def act():
    def experience_replay():
    