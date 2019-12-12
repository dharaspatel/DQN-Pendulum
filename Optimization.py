from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
import cartpole

AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100

GAMMA = np.arange(.05,1,.05)
LEARNING_RATE = np.arange(.0001, .0023, .0002)
EXPLORATION_DECAY = np.arange(.85, .999, .01)
GammaVals = {}
LearningRateVals = {}
ExplorationDecayVals = {}


for i in GAMMA:
    runs2solve = cartpole.simcartpole(GAMMA = i, LEARNING_RATE = .9, EXPLORATION_DECAY = .995)
    GammaVals[i] = runs2solve
print(GammaVals)




    
