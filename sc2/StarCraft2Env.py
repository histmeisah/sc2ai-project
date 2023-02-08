#import subprocess

import gym
import numpy as np
#import pickle
#import time


class StarCraft2Env(gym.Env):
    def __init__(self):
        super(StarCraft2Env,self).__init__()
        self.observation_space = gym.spaces.Box(low = 0,high = 255,shape = (224,224,3),dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(51)

'''
    def step(self,action):
        
        while True:
            try:
                with open('transaction.pkl','rb') as f:
                    transaction = pickle.load(f)
                    break
                time.sleep(0.1)
            except Exception as e:
                time.sleep(0.1)
                pass
        
        transaction['action'] = action
        transaction['isTTO'] = False
        while True:
            try:
                with open('transaction.pkl', 'wb') as f:
                    pickle.dump(transaction,f)
                    break
            except Exception as e:
                time.sleep(0.1)




    def reset(self):
        print('Reset the envirnment!')
        map = np.zeros((224,224,3),dtype=np.uint8)
        transaction={'observation':map,'reward':0,'action':None,'done':False,'isTTO':False,'res':None,'iter':0}  #
        with open('transaction.pkl','wb') as f:
            pickle.dump(transaction,f)

        #subprocess.Popen(['python','WokerRushBot.py'])
        return transaction
'''