import cartpole
import numpy as np
import pandas as pd 



defaultVals = [.95, .001, .995]
GAMMA = np.arange(.05,.95,.05) #.95
LEARNING_RATE = np.arange(.0001, .0023, .0002)
EXPLORATION_DECAY = np.arange(.85, .999, .01)
Variables = [GAMMA, LEARNING_RATE, EXPLORATION_DECAY]
GammaVals = {}
LearningRateVals = {}
ExplorationDecayVals = {}


data = []
for i in GAMMA:
    print(i)
    runs2solve = cartpole.cartpole(GAMMA = i, LEARNING_RATE = defaultVals[1], EXPLORATION_DECAY = defaultVals[2])
    entry = [i,runs2solve]
    data.append(entry)
    
print(data)
pd.DataFrame(data).to_csv("/home/sander/Documents/PendulumTest/cartpole/GAMMAData.csv")
