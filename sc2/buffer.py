import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.information_memory = [None for _ in range(max_size)]
        self.action_memory = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.next_information_memory = [None for _ in range(max_size)]
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool)
        self.concat_memory = [None for _ in range(max_size)]
        self.next_concat_memory = [None for _ in range(max_size)]

    def store_transition(self, state, information, action, reward, state_, information_, concat,next_concat, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state.reshape(1, -1)
        self.information_memory[mem_idx] = information
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_.reshape(1, -1)
        self.next_information_memory[mem_idx] = information_
        self.terminal_memory[mem_idx] = done
        self.concat_memory[mem_idx] = concat
        self.next_concat_memory[mem_idx]=next_concat

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]

        information = []
        for i in range(len(batch)):
            information.append(self.information_memory[batch[i]])

        actions = self.action_memory[batch]

        rewards = self.reward_memory[batch]

        states_ = self.next_state_memory[batch]
        concat = []
        for i in range(len(batch)):
            concat.append(self.concat_memory[batch[i]])
        next_concat = []
        for i in range(len(batch)):
            next_concat.append(self.next_concat_memory[batch[i]])
        information_ = []
        for i in range(len(batch)):
            information_.append(self.next_information_memory[batch[i]])

        terminals = self.terminal_memory[batch]

        return states, information, actions, rewards, states_, information_, terminals, concat,next_concat, batch

    def ready(self):
        return self.mem_cnt > self.batch_size
