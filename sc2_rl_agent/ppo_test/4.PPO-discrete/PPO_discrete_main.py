import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from StarCraft2Env import StarCraft2Env
from utils import create_directory, plot_learning_curve
import multiprocessing
from WokerRushBot import worker
import time
from collections import deque
from replaybuffer import TrajectoryReplayBuffer
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def load_latest_weights(agent, checkpoint_dir):
    latest_episode, latest_actor_file, latest_critic_file = find_latest_checkpoints(checkpoint_dir)

    if latest_actor_file and latest_critic_file:
        print(f"Loading latest model weights from episode {latest_episode}")
        agent.load(checkpoint_dir, latest_episode)
    else:
        print("No saved model weights found. Using initialized networks.")


def find_latest_checkpoints(checkpoint_dir):
    actor_files = sorted(glob.glob(os.path.join(checkpoint_dir, "Actor_net", "*.pt")), key=os.path.getctime)
    critic_files = sorted(glob.glob(os.path.join(checkpoint_dir, "Critic_net", "*.pt")), key=os.path.getctime)

    if actor_files and critic_files:
        latest_actor_file = actor_files[-1]
        latest_critic_file = critic_files[-1]

        # Extract the episode number from the filenames
        latest_episode = int(os.path.splitext(os.path.basename(latest_actor_file))[0].split("_")[-1])

        return latest_episode, latest_actor_file, latest_critic_file
    else:
        return None, None, None


# Load the latest model weights
def load_latest_weights(agent, checkpoint_dir):
    latest_episode, latest_actor_file, latest_critic_file = find_latest_checkpoints(checkpoint_dir)

    if latest_actor_file and latest_critic_file:
        print(f"Loading latest model weights from episode {latest_episode}")
        agent.load(checkpoint_dir, latest_episode)
    else:
        print("No saved model weights found.")


def evaluate_policy(args, agent, transaction):
    lock = multiprocessing.Manager().Lock()

    times = 1
    evaluate_reward = 0
    def launch_worker(transaction, lock):
        p = multiprocessing.Process(target=worker, args=(transaction, lock))
        p.start()
        return p

    for _ in range(times):
        worker_process = launch_worker(transaction, lock)

        transaction.update(
            {'observation': np.zeros((224, 224, 3), dtype=np.uint8), 'information': np.zeros((1, 147)), 'reward': 0,
             'action': None,
             'done': False,
             'isTTO': False, 'res': None, 'iter': 0})

        # observation = transaction['observation']
        # information = transaction['information']
        concat = agent.get_concat_data()
        concat_arr = []

        # 0 zeros
        concat_arr.append(concat)
        now_concat_arr = []
        for i in range(len(concat_arr)):
            sample = concat_arr[i].unsqueeze(0)  # 将第i个时间步的输入转换为1x128的张量
            now_concat_arr.append(sample)
        # concat_data = torch.cat(now_concat_arr, dim=0)
        # done = False
        episode_reward = 0
        done = False
        while not done:

            concat_arr = []

            while not np.any(transaction['observation']):
                time.sleep(0.0001)
            observation = transaction['observation']
            information = transaction['information']
            concat = agent.get_concat_data()
            concat_arr.append(concat)
            now_concat_arr = []
            for i in range(len(concat_arr)):
                sample = concat_arr[i].unsqueeze(0)  # 将第i个时间步的输入转换为1x128的张量
                now_concat_arr.append(sample)
            concat_data = torch.cat(now_concat_arr, dim=0)
            a = agent.evaluate(observation, information,
                               concat_data)  # We use the deterministic policy during the evaluating

            # Store the action in the transaction dictionary
            transaction['action'] = a

            # Wait for the worker process to execute the action and update the transaction dictionary
            while not transaction['isTTO']:
                time.sleep(0.0001)

            # Get the updated state, reward, and done flag from the transaction dictionary
            # observation = transaction['observation']
            reward = transaction['reward']
            done = transaction['done']
            res = transaction['res']
            # iter = transaction['iter']

            if done:
                if res.name == 'Victory':
                    reward += 50
            episode_reward += reward
        worker_process.terminate()
        worker_process.join()
        print("evaluate_end")
    return evaluate_reward / times



def main(args, number):

    def launch_worker(transaction, lock):
        p = multiprocessing.Process(target=worker, args=(transaction, lock))
        p.start()
        return p
    model_queue = deque(maxlen=3)

    transaction = multiprocessing.Manager().dict()
    lock = multiprocessing.Manager().Lock()
    transaction.update(
        {'observation': np.zeros((224, 224, 3), dtype=np.uint8), 'information': np.zeros((1, 147), dtype=float),
         'reward': 0, 'action': None,
         'done': False,
         'isTTO': False, 'res': None, 'iter': 0})
    env = StarCraft2Env()

    args.state_dim = (224, 224, 3)
    args.action_dim = env.action_space.n

    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    trajectory_replay_buffer =TrajectoryReplayBuffer(args)


    agent =PPO_discrete(args)
    find_latest_checkpoints(args.ckpt_dir)
    load_latest_weights(agent, args.ckpt_dir)
    create_directory(args.ckpt_dir, sub_dirs=['Critic_net', 'Actor_net'])

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}'.format('episode', number))

    state_norm = Normalization(shape=args.state_dim,device=device)  # Trick 2:state normalization
    information_norm = Normalization(shape=args.information_dim,device=device)
    concat_norm = Normalization(shape=args.concat_dim,device=device)
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1,device=device)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma,device=device)

    for episode in range(args.max_episodes):
        print("game start")
        transaction.update(
            {'observation': np.zeros((224, 224, 3), dtype=np.uint8), 'information': np.zeros((1, 147), dtype=float),
             'reward': 0, 'action': None,
             'done': False,
             'isTTO': False, 'res': None, 'iter': 0})
        p = multiprocessing.Process(target=worker, args=(transaction, lock))
        p.start()
        concat_arr = []

        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0


        done = False
        while not done:
            while not np.any(transaction['observation']):
                time.sleep(0.01)
            observation = transaction['observation']
            information = transaction['information']
            concat = agent.get_concat_data()
            concat_arr.append(concat)
            now_concat_arr = []
            for i in range(len(concat_arr)):
                sample = concat_arr[i].unsqueeze(0)  # 将第i个时间步的输入转换为1x128的张量
                now_concat_arr.append(sample)
            concat_data = torch.cat(now_concat_arr, dim=0)
            episode_steps += 1
            a, a_logprob = agent.choose_action(observation, information,
                                               concat_data)  # Action and the corresponding log probability
            lock.acquire()
            transaction['action'] = a
            transaction['action_log_prob'] = a_logprob
            lock.release()
            while not transaction['isTTO']:
                time.sleep(0.01)

            # 获取动作结束的状态、回报、结果、迭代次数

            observation_ = transaction['observation']
            information_ = transaction['information']
            concat_ = agent.get_concat_data()
            reward = transaction['reward']
            done = transaction['done']
            res = transaction['res']
            iter = transaction['iter']
            print(f"observation.shape={observation.shape}")
            replay_buffer.store(args,observation, information, concat, a, a_logprob, reward, observation_, information_, concat_,
                                done)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done:
                trajectory_replay_buffer.store_trajectory(replay_buffer)
                replay_buffer.reset()
                if res.name == 'Victory':
                    reward += 50
            print('done = %s' % done)

            # def store(self, s, i, c, a, a_logprob, r, s_, i_, c_, done):


            observation = observation_
            print('reward:%f' % reward)

            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if trajectory_replay_buffer.count == args.batch_size:
                agent.update(trajectory_replay_buffer, total_steps)
        print('test_call')
        time.sleep(1)
        p.join()
        env.close()
        #
            # Evaluate the policy every 'evaluate_freq' steps
        if episode % 50 == 0 and episode > 0:
            evaluate_and_save(agent, args, episode, evaluate_rewards, model_queue, total_steps, writer)



def evaluate_and_save(agent, args, episode, evaluate_rewards, model_queue, total_steps, writer):
    print("start evaluate")

    transaction = multiprocessing.Manager().dict()
    model_queue.append((episode, agent.actor.state_dict(), agent.critic.state_dict()))
    for idx, (ep, actor_state, critic_state) in enumerate(model_queue):
        actor_checkpoint_file = os.path.join(args.ckpt_dir, "Actor_net", f"actor_{ep}.pt")
        critic_checkpoint_file = os.path.join(args.ckpt_dir, "Critic_net", f"critic_{ep}.pt")

        # Save actor and critic states using their respective save_checkpoint methods
        agent.actor.save_checkpoint(actor_checkpoint_file)
        agent.critic.save_checkpoint(critic_checkpoint_file)

    agent.save(args.ckpt_dir, episode)
    evaluate_num = episode // 10
    saved_agent = PPO_discrete(args)
    saved_agent.load(args.ckpt_dir, episode)
    evaluate_reward = evaluate_policy(args, agent, transaction)
    evaluate_rewards.append(evaluate_reward)
    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
    writer.add_scalar('step_rewards_{}', evaluate_rewards[-1], global_step=total_steps)
    if evaluate_num % args.save_freq == 0:
        np.save('./data_train/PPO_discrete_number_{}.npy', evaluate_num), np.array(evaluate_rewards)

if __name__ == '__main__':
    env = StarCraft2Env()

    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--information_dim", type=int, default=147, help=" game inner information")
    parser.add_argument("--concat_dim", type=int, default=256, help=" game concat state")
    parser.add_argument("--action_dim", type=int, default=int(env.action_space.n), help=" designed actions")
    parser.add_argument("--state_dim", type=int, default=(224,224,3), help=" game state")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")


    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--max_episodes", type=int, default=10000, help="Maximum number of episodes")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/ppo/')
    parser.add_argument('--weights_directory', type=str, default='checkpoints/ppo/')
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_index = 1
    main(args, number=1)
