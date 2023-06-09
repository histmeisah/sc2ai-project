import multiprocessing
import os
import time

import numpy as np

from D3QN import D3QN
from StarCraft2Env import StarCraft2Env
from WokerRushBot import worker

envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


def main():
    transaction = multiprocessing.Manager().dict()
    lock = multiprocessing.Lock()
    transaction.update({'observation': np.zeros((224,224,3),dtype=np.uint8), 'reward': 0, 'action': None, 'done': False, 'isTTO': False, 'res': None,'iter': 0})
    env = StarCraft2Env()
    #env.reset()



    agent = D3QN(alpha=0.0003, state_dim=50176*3, action_dim=env.action_space.n,
                 ckpt_dir='checkpoints/D3QN/', gamma=0.99, tau=0.005, epsilon=1.0,
                 eps_end=0.05, eps_dec=5e-4, max_size=1000, batch_size=1)
    agent.load_models(10)
    total_rewards, avg_rewards, epsilon_history = [], [], []



    p = multiprocessing.Process(target=worker,args=(transaction,lock))
    p.start()
    total_reward = 0
    res=None
    done = False

    while not done:

        while not np.any(transaction['observation']):
            time.sleep(0.001)
        observation = transaction['observation']


        action = agent.choose_action(observation, isTrain=True)
        #env.step(action)
        lock.acquire()
        transaction['action'] = action
        lock.release()


        #等待到当前动作结束
        while not transaction['isTTO']:
            time.sleep(0.001)

        #获取动作结束的状态、回报、结果、迭代次数
        observation_ = transaction['observation']
        reward = transaction['reward']
        done = transaction['done']
        res = transaction['res']
        iter = transaction['iter']
        print('iter%d'%iter)


    p.join()







if __name__ == '__main__':
    model_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    model_dir = f'models/{model_name}/'
    logs_dir = f'logs/'
    logpath = logs_dir + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + f'.txt'

    main()
