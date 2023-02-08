import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, information_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()
        self.alex_net = nn.Sequential(
            # 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
            # 同时，步幅为4，以减少输出的⾼度和宽度。
            # 另外，输出通道的数⽬远⼤于LeNet
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
            nn.Conv2d(48, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
            # 除了最后的卷积层，输出通道的数量进⼀步增加。
            # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
            nn.Conv2d(128, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 128)).to(device)

        self.fc1 = nn.Linear(128, fc1_dim).to(device)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim).to(device)

        self.information_layer1 = nn.Linear(information_dim, 128).to(device)
        self.information_layer2 = nn.Linear(128, 128).to(device)
        self.information_layer3 = nn.Linear(128, 256).to(device)

        self.concat_layer1 = nn.Linear(fc2_dim + 256, 1024).to(device)
        self.concat_layer2 = nn.Linear(1024, 512).to(device)
        self.concat_layer3 = nn.Linear(512, fc2_dim).to(device)

        self.V = nn.Linear(fc2_dim, 1).to(device)
        self.A = nn.Linear(fc2_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, information):
        # print(state.shape)
        x = torch.relu(self.alex_net(state))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # print('information.shape',information.shape)
        i = torch.relu(self.information_layer1(information))
        i = torch.relu(self.information_layer2(i))
        i = torch.relu(self.information_layer3(i))
        # print('i.shape',i.shape)
        # print(f'shape_i={i.shape}')
        # print(f'shape_x = {x.shape}')
        x = torch.concat((x, i), dim=1)
        x = torch.relu(self.concat_layer1(x))
        x = torch.relu(self.concat_layer2(x))
        x = torch.relu(self.concat_layer3(x))
        V = self.V(x)
        A = self.A(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class D3QN:
    def __init__(self, alpha, state_dim, information_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=.5, eps_end=0.01, eps_dec=5e-7,
                 max_size=100000, batch_size=1):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, information_dim=information_dim,
                                          action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, information_dim=information_dim,
                                            action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=150528, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, information, action, reward, state_, information_, done):
        self.memory.store_transition(state, information, action, reward, state_, information_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, information, isTrain=True):
        observation = np.reshape(observation, (3, 224, 224))
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)  # 把224*224*3的展开
        work_supply_list = []
        mineral_list = []
        gas_supply_list = []
        army_supply_list = []
        enemy_count_list = []
        game_time_list = []
        by_count_list = []
        bf_count_supply_list = []
        vs_count_list = []
        vr_count_list = []
        base_pending_list = []
        gateway_count_list = []
        warp_gate_count_list = []
        work_supply = information['work_supply']
        mineral = information['mineral']
        gas = information['gas']
        army_supply = information['army_supply']
        enemy_count = information['enemy_count']
        game_time = information['game_time']
        by_count = information['by_count']
        bf_count = information['bf_count']
        vs_count = information['vs_count']
        vr_count = information['vr_count']
        base_pending = information['base_pending']
        gateway_count = information['gateway_count']
        warp_gate_count = information['warp_gate_count']
        work_supply_list.append(work_supply)
        mineral_list.append(mineral)
        gas_supply_list.append(gas)
        army_supply_list.append(army_supply)
        enemy_count_list.append(enemy_count)
        game_time_list.append(game_time)
        by_count_list.append(by_count)
        bf_count_supply_list.append(bf_count)
        vs_count_list.append(vs_count)
        vr_count_list.append(vr_count)
        base_pending_list.append(base_pending)
        gateway_count_list.append(gateway_count)
        warp_gate_count_list.append(warp_gate_count)
        work_supply_tensor = torch.tensor(work_supply_list)
        mineral_tensor = torch.tensor(mineral_list)
        gas_supply_tensor = torch.tensor(gas_supply_list)
        army_supply_tensor = torch.tensor(army_supply_list)
        enemy_count_tensor = torch.tensor(enemy_count_list)
        game_time_tensor = torch.tensor(game_time_list)
        by_count_tensor = torch.tensor(by_count_list)
        bf_count_supply_tensor = torch.tensor(bf_count_supply_list)
        vs_count_tensor = torch.tensor(vs_count_list)
        vr_count_tensor = torch.tensor(vr_count_list)
        base_pending_tensor = torch.tensor(base_pending_list)
        gateway_count_tensor = torch.tensor(gateway_count_list)
        warp_gate_count_tensor = torch.tensor(warp_gate_count_list)
        # print(work_supply_tensor)
        # print(work_supply_tensor.shape)
        information_tensor = torch.concat((work_supply_tensor, mineral_tensor, gas_supply_tensor, army_supply_tensor,
                                           enemy_count_tensor, game_time_tensor, by_count_tensor,
                                           bf_count_supply_tensor, vs_count_tensor, vr_count_tensor,
                                           base_pending_tensor, gateway_count_tensor, warp_gate_count_tensor), dim=0).to(device)
        # print(np.shape(state))
        # print(state.shape)
        information_tensor = information_tensor.unsqueeze(0)
        q_vals = self.q_eval.forward(state, information_tensor)
        action = torch.argmax(q_vals).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.memory.ready():
            return

        states, information, actions, rewards, next_states, next_information, terminals, batch = self.memory.sample_buffer()
        batch_idx = torch.arange(self.batch_size, dtype=torch.long).to(device)

        states = np.array(np.reshape(states, (3, 224, 224)))
        states_tensor = torch.tensor(states, dtype=torch.float)
        states_tensor = torch.unsqueeze(states_tensor, dim=0).to(device)
        # print(np.shape(states_tensor))

        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)

        next_states = np.array(np.reshape(next_states, (3, 224, 224)))
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(device)
        next_states_tensor = torch.unsqueeze(next_states_tensor, dim=0).to(device)
        # print(np.shape(next_states_tensor))

        terminals_tensor = torch.tensor(terminals).to(device)
        work_supply_list = []
        mineral_list = []
        gas_supply_list = []
        army_supply_list = []
        enemy_count_list = []
        game_time_list = []
        by_count_list = []
        bf_count_supply_list = []
        vs_count_list = []
        vr_count_list = []
        base_pending_list = []
        gateway_count_list = []
        warp_gate_count_list = []

        for i in range(len(batch)):
            work_supply = information[i]['work_supply']
            mineral = information[i]['mineral']
            gas = information[i]['gas']
            army_supply = information[i]['army_supply']
            enemy_count = information[i]['enemy_count']
            game_time = information[i]['game_time']
            by_count = information[i]['by_count']
            bf_count = information[i]['bf_count']
            vs_count = information[i]['vs_count']
            vr_count = information[i]['vr_count']
            base_pending = information[i]['base_pending']
            gateway_count = information[i]['gateway_count']
            warp_gate_count = information[i]['warp_gate_count']
            work_supply_list.append(work_supply)
            mineral_list.append(mineral)
            gas_supply_list.append(gas)
            army_supply_list.append(army_supply)
            enemy_count_list.append(enemy_count)
            game_time_list.append(game_time)
            by_count_list.append(by_count)
            bf_count_supply_list.append(bf_count)
            vs_count_list.append(vs_count)
            vr_count_list.append(vr_count)
            base_pending_list.append(base_pending)
            gateway_count_list.append(gateway_count)
            warp_gate_count_list.append(warp_gate_count)
        work_supply_tensor = torch.tensor(work_supply_list)
        mineral_tensor = torch.tensor(mineral_list)
        gas_supply_tensor = torch.tensor(gas_supply_list)
        army_supply_tensor = torch.tensor(army_supply_list)
        enemy_count_tensor = torch.tensor(enemy_count_list)
        game_time_tensor = torch.tensor(game_time_list)
        by_count_tensor = torch.tensor(by_count_list)
        bf_count_supply_tensor = torch.tensor(bf_count_supply_list)
        vs_count_tensor = torch.tensor(vs_count_list)
        vr_count_tensor = torch.tensor(vr_count_list)
        base_pending_tensor = torch.tensor(base_pending_list)
        gateway_count_tensor = torch.tensor(gateway_count_list)
        warp_gate_count_tensor = torch.tensor(warp_gate_count_list)
        # print(work_supply_tensor)
        # print(work_supply_tensor.shape)
        information_tensor = torch.concat((work_supply_tensor, mineral_tensor, gas_supply_tensor, army_supply_tensor,
                                           enemy_count_tensor, game_time_tensor, by_count_tensor,
                                           bf_count_supply_tensor, vs_count_tensor, vr_count_tensor,
                                           base_pending_tensor, gateway_count_tensor, warp_gate_count_tensor), dim=0).to(device)
        information_tensor = information_tensor.unsqueeze(0)



        next_work_supply_list = []
        next_mineral_list = []
        next_gas_supply_list = []
        next_army_supply_list = []
        next_enemy_count_list = []
        next_game_time_list = []
        next_by_count_list = []
        next_bf_count_supply_list = []
        next_vs_count_list = []
        next_vr_count_list = []
        next_base_pending_list = []
        next_gateway_count_list = []
        next_warp_gate_count_list = []

        for i in range(len(batch)):
            next_work_supply = next_information[i]['work_supply']
            next_mineral = next_information[i]['mineral']
            next_gas = next_information[i]['gas']
            next_army_supply = next_information[i]['army_supply']
            next_enemy_count = next_information[i]['enemy_count']
            next_game_time = next_information[i]['game_time']
            next_by_count = next_information[i]['by_count']
            next_bf_count = next_information[i]['bf_count']
            next_vs_count = next_information[i]['vs_count']
            next_vr_count = next_information[i]['vr_count']
            next_base_pending = next_information[i]['base_pending']
            next_gateway_count = next_information[i]['gateway_count']
            next_warp_gate_count = next_information[i]['warp_gate_count']
            next_work_supply_list.append(next_work_supply)
            next_mineral_list.append(next_mineral)
            next_gas_supply_list.append(next_gas)
            next_army_supply_list.append(next_army_supply)
            next_enemy_count_list.append(next_enemy_count)
            next_game_time_list.append(next_game_time)
            next_by_count_list.append(next_by_count)
            next_bf_count_supply_list.append(next_bf_count)
            next_vs_count_list.append(next_vs_count)
            next_vr_count_list.append(next_vr_count)
            next_base_pending_list.append(next_base_pending)
            next_gateway_count_list.append(next_gateway_count)
            next_warp_gate_count_list.append(next_warp_gate_count)
        next_work_supply_tensor = torch.tensor(next_work_supply_list)
        next_mineral_tensor = torch.tensor(next_mineral_list)
        next_gas_supply_tensor = torch.tensor(next_gas_supply_list)
        next_army_supply_tensor = torch.tensor(next_army_supply_list)
        next_enemy_count_tensor = torch.tensor(next_enemy_count_list)
        next_game_time_tensor = torch.tensor(next_game_time_list)
        next_by_count_tensor = torch.tensor(next_by_count_list)
        next_bf_count_supply_tensor = torch.tensor(next_bf_count_supply_list)
        next_vs_count_tensor = torch.tensor(next_vs_count_list)
        next_vr_count_tensor = torch.tensor(next_vr_count_list)
        next_base_pending_tensor = torch.tensor(next_base_pending_list)
        next_gateway_count_tensor = torch.tensor(next_gateway_count_list)
        next_warp_gate_count_tensor = torch.tensor(next_warp_gate_count_list)
        # print(work_supply_tensor)
        # print(work_supply_tensor.shape)
        next_information_tensor = torch.concat(
            (next_work_supply_tensor, next_mineral_tensor, next_gas_supply_tensor, next_army_supply_tensor,
             next_enemy_count_tensor, next_game_time_tensor, next_by_count_tensor,
             next_bf_count_supply_tensor, next_vs_count_tensor, next_vr_count_tensor,
             next_base_pending_tensor, next_gateway_count_tensor, next_warp_gate_count_tensor), dim=0).to(device)
        next_information_tensor = next_information_tensor.unsqueeze(0)

        # print(information_tensor)
        # print(information_tensor.shape)

        # print(f'worker_supply ={work_supply_list}')

        with torch.no_grad():
            q_ = self.q_target.forward(next_states_tensor, next_information_tensor)
            max_actions = torch.argmax(self.q_eval.forward(next_states_tensor, next_information_tensor), dim=-1)
            # print(f'terminals_shape = {terminals_tensor.shape}')
            # print(f'q_table={q_}')
            # print(f'q.shape = {q_.shape}')
            # print(f'terminals_tensor= {terminals_tensor}')
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
        q = self.q_eval.forward(states_tensor, information_tensor)[batch_idx, actions_tensor]

        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
