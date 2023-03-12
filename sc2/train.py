import argparse
import multiprocessing
import os
import time

import numpy as np
import torch

from D3QN import D3QN
from StarCraft2Env import StarCraft2Env
from WokerRushBot import worker
from utils import create_directory, plot_learning_curve

envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


def main():
    transaction = multiprocessing.Manager().dict()
    lock = multiprocessing.Manager().Lock()
    transaction.update(
        {'observation': np.zeros((224, 224, 3), dtype=np.uint8), 'information': [], 'reward': 0, 'action': None,
         'done': False,
         'isTTO': False, 'res': None, 'iter': 0})
    env = StarCraft2Env()

    # env.reset()

    agent = D3QN(alpha=0.0003, state_dim=10, information_dim=121, action_dim=env.action_space.n,
                 ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                 eps_end=0.05, eps_dec=5e-4, max_size=1000, batch_size=1)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []

    for episode in range(args.max_episodes):
        p = multiprocessing.Process(target=worker, args=(transaction, lock))
        p.start()
        total_reward = 0
        res = None
        done = False

        concat_arr = []
        while not done:
            '''
            while True:
                try:
                    with open('transaction.pkl', 'rb') as f:
                        transaction = pickle.load(f)
                    if np.any(transaction['observation']):
                        observation = transaction['observation']
                        break
                    time.sleep(0.1)
                except Exception as e:
                    time.sleep(0.1)
                    time.sleep(0.1)
                    pass        
             '''
            while not np.any(transaction['observation']):
                time.sleep(0.001)
            observation = transaction['observation']
            information = transaction['information']
            # print(information)
            concat=agent.get_concat_data()
            concat_arr.append(concat)
            now_concat_arr= []
            for i in range(len(concat_arr)):
                sample = concat_arr[i].unsqueeze(0)  # 将第i个时间步的输入转换为1x128的张量
                now_concat_arr.append(sample)
            concat_data = torch.cat(now_concat_arr, dim=0)
            # print(f'concat_data.shape={concat_data.shape}')
            action = agent.choose_action(observation, information, concat=concat_data, isTrain=True)
            # env.step(action)
            lock.acquire()
            transaction['action'] = action
            lock.release()

            '''
            while True:
                try:
                    with open('transaction.pkl', 'rb') as f:
                        transaction = pickle.load(f)
                    if transaction['isTTO']:
                        observation_ = transaction['observation']
                        reward = transaction['reward']
                        done = transaction['done']
                        res = transaction['res']
                        iter = transaction['iter']
                        break
                    time.sleep(0.1)
                except Exception as e:
                    time.sleep(0.1)
                    pass
            '''

            # 等待到当前动作结束
            while not transaction['isTTO']:
                time.sleep(0.001)

            # 获取动作结束的状态、回报、结果、迭代次数
            observation_ = transaction['observation']
            information_ = transaction['information']

            reward = transaction['reward']
            done = transaction['done']
            res = transaction['res']
            iter = transaction['iter']
            concat_ = agent.get_concat_data()
            print('iter%d' % iter)

            if done:
                if res.name == 'Victory':
                    reward += 50
            agent.remember(observation, information, action, reward, observation_, information_, concat,concat_,done)
            agent.learn()
            total_reward += reward
            observation = observation_
            print('reward:%f' % reward)

        p.join()
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        print('EP:{} Reward:{} Avg_reward:{} Epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        global logpath
        if (os.path.getsize(logpath) > 20 * 1024 * 1024):
            logpath = logs_dir + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + f'.txt'
        with open(logpath, mode='a', encoding='utf-8') as f:
            f.write('Time:{} EP:{} Reward:{} Avg_reward:{} Epsilon:{} Iter:{} Res:{}\n'.
                    format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), episode + 1, total_reward,
                           avg_reward, agent.epsilon, iter, res))

        if (episode + 1) % 10 == 0:
            agent.save_models(episode + 1)

        env = StarCraft2Env()
        # env.reset()
        transaction.update(
            {'observation': np.zeros((224, 224, 3), dtype=np.uint8), 'information': [], 'reward': 0, 'action': None,
             'done': False,
             'isTTO': False, 'res': None, 'iter': 0})

    episodes = [i + 1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, title='Reward', ylabel='reward',
                        figure_file=args.reward_path)
    plot_learning_curve(episodes, epsilon_history, title='Epsilon', ylabel='epsilon',
                        figure_file=args.epsilon_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/D3QN/')
    parser.add_argument('--reward_path', type=str, default='output_images/reward.png')
    parser.add_argument('--epsilon_path', type=str, default='output_images/epsilon.png')

    args = parser.parse_args()

    model_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    model_dir = f'models/{model_name}/'
    logs_dir = f'logs/'
    logpath = logs_dir + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + f'.txt'

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    with open(logpath, mode='a', encoding='utf-8') as f:
        pass

    main()
