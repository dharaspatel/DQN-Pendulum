import random
import gym
import numpy as np

#LOOK UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from scores.score_logger import ScoreLogger


#defining DQN equation variables: 
ENV_NAME = "CartPole-v1"
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000 #sets the number of iterations in
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0 #exploration rate limits (how much you can explore random actions)
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

def cartpole():
    #debugging
    score_logger = ScoreLogger(ENV_NAME)

    #environment - cartpole
    env = gym.make(ENV_NAME)

    #observational space - possible state values
    observational_space = env.observational_space.shape[0]

    #action space - possible actions that can be performed
    action_space = env.action_space.n

    #agent - object of DQN Solver class, see below
    agent = DQNSolver(observational_space, action_space)

    run = 0
    while True:
        run += 1
        #reset state at once terminal environment is reached 
        state = env.reset
        state = np.reshape(state, [1, observational_space])

        step = 0
        while True:
            step += 1 #iterations 

            #update environment 
            env.render()

            #determine action 
            action = agent.act(state)

            #determine new state and corresponding reward 
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observational_space])
            reward = reward if not terminal else -reward

            #remember to learn - used in experience replay 
            agent.remember(state, action, reward, state_next, terminal)

            #set future state as current state
            state = state_next

            #check if terminal state has been reached 
            if state == terminal:
                print("Run: " + str(run) + ", exploration: " + str(agent.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break
            #experience replay - analyzes what happened and updates Q
                agent.experience_replay()
    
class DQNSolver: #Class that solves the Q-Network 
    def __init__(self, observation_space, action_space):
        #initialize attributes of the class 
        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = EXPLORATION_MAX

        #memory object for previous state variables 
        self.memory = deque(maxlen=MEMORY_SIZE)

        #network strucutre
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))


    def remember(self, state, action, reward, next_state, done):
        #add to the memory 
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #if the exploration limit has not been reached, do random action 
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        #else pick optimized known action depending on reward and probability 
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
        
    def experience_replay(self): #updates probability depending on rewards 
        #make sure there are 20 actions in the memory 
        if len(self.memory) < BATCH_SIZE:
            return
        
        #randomize the batch to reduce the correlation between subsequent actions 
        #ASK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch = random.sample(self.memory, BATCH_SIZE)

        #for every tuple, update...
        for state, action, reward, state_next, terminal in batch:
            
            #initialize update as reward (1/ts) so that the probability of a next_state that hasn't reached terminal is greater
            q_update = reward

            #if it hasn't fallen down,  
            if not terminal:
                #update the reward for 
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            
            #probability of doing a particular action given a particular state
            q_values = self.model.predict(state) 
            
            #set the action in the first tuple to q_update
            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)

        #as the network moves forward, the model is more and more likely to pick an optimized action vs. a random action
        self.exploration_rate *= EXPLORATION_DECAY
        #limit the exploration rate to the exploration minimum
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    