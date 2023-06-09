import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer

from efficientnet_pytorch import EfficientNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MyEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0').to(device)
        self.classifier = nn.Linear(1000, num_classes).to(device)

    def forward(self, x):
        x = self.efficientnet(x).to(device)
        print(f'x.shape={x.shape}')
        x = self.classifier(x).to(device)
        return x

class AlphaStarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AlphaStarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x, hidden=None):
        x = x.to(device)
        if hidden is None:
            batch_size = x.size(1)
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if hidden is not None:
            hidden = (hidden[0].to(device), hidden[1].to(device))
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out[-1][-1].unsqueeze(0))
        return output, hidden


def information_process(information):
    n = len(information)
    information_list = [[] for _ in range(n)]
    for i, value in enumerate(information.values()):
        information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor, (-1,))
    information_tensor = information_tensor.unsqueeze(0).to(device)
    return information_tensor
# def concat_process(concat):
#     n= len(concat)
#     concat_list = [[] for _ in range(n)]
#     for i in range()

def learn_information_process(information, batch):
    n = len(information[0])
    information_list = [[] for _ in range(n)]
    for j in range(len(batch)):
        for i, value in enumerate(information[j].values()):
            information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor, (-1,))
    information_tensor = information_tensor.unsqueeze(0).to(device)
    return information_tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(planes).to(device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.bn2 = nn.BatchNorm2d(planes).to(device)

        self.shortcut = nn.Sequential().to(device)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes).to(device)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1).to(device)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2).to(device)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2).to(device)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2).to(device)
        self.linear = nn.Linear(25088, num_classes).to(device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        # print(f'out_shape = {out.shape}')
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout).to(device)
        self.fc = nn.Linear(input_dim, d_model).to(device)
        self.output = nn.Linear(d_model, output_dim).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x):
        x = self.fc(x)
        # x = x.transpose(0, 1)
        # print(f'x_transformer_shape={x.shape}')
        output = self.transformer(x, x)
        # output = output.transpose(0, 1)
        output = self.output(output)
        output = self.softmax(output)
        return output


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, information_dim, action_dim ):
        super(DuelingDeepQNetwork, self).__init__()
        self.resnet = ResNet18().to(device)
        self.efficientnet = MyEfficientNet().to(device)
        self.transformer = Transformer(input_dim=information_dim, output_dim=128).to(device)
        self.lstm = AlphaStarLSTM(input_size=256,hidden_size=12,num_layers=10,output_size=128).to(device)
        self.concat_layer1 = nn.Linear(10 + 128, 256).to(device)
        self.concat_layer2 = nn.Linear(256, 512).to(device)
        self.concat_layer3 = nn.Linear(512, 256).to(device)
        self.concat_data = torch.zeros((1,256)).to(device)
        self.V = nn.Linear(128, 1).to(device)
        self.A = nn.Linear(128, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, information,concat):
        # print(f'state_shape={state.shape}')
        x = torch.relu(self.efficientnet(state))
        i = torch.relu(self.transformer(information))

        # print('i.shape',i.shape)
        # print(f'shape_i={i.shape}')
        # print(f'shape_x = {x.shape}')
        x = torch.concat((x, i), dim=1)
        # print(f'x.shape={x.shape}')
        x = torch.relu(self.concat_layer1(x))
        x = torch.relu(self.concat_layer2(x))
        x = torch.relu(self.concat_layer3(x)).clone().detach().requires_grad_(True)
        self.concat_data = x
        # print(f'x.dim={x.shape}')
        x = x.unsqueeze(0)
        concat=concat.to(device)
        concat = torch.cat((concat,x), dim=0)
        result,_ = self.lstm(concat)
        V = self.V(torch.relu(result))
        A = self.A(torch.relu(result))
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q
    def get_concat_data(self):
        return self.concat_data
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class D3QN:
    def __init__(self, alpha, state_dim, information_dim, action_dim,  ckpt_dir,
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
                                          action_dim=action_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, information_dim=information_dim,
                                            action_dim=action_dim)

        self.memory = ReplayBuffer(state_dim=150528, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, information, action, reward, state_, information_,concat_data, next_concat_data,done):
        self.memory.store_transition(state, information, action, reward, state_, information_,concat_data ,next_concat_data,done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, information, concat,isTrain=True):
        observation = np.reshape(observation, (3, 224, 224))
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)  # 把224*224*3的展开
        # print(len(information))

        information_tensor = information_process(information)
        # print(f'information_tensor = {information_tensor.shape}')

        q_vals = self.q_eval.forward(state, information_tensor,concat)
        action = torch.argmax(q_vals).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)

        return action
    def get_concat_data(self):

        return self.q_eval.get_concat_data()

    def learn(self):
        if not self.memory.ready():
            return

        states, information, actions, rewards, next_states, next_information, terminals, concat,next_concat,batch = self.memory.sample_buffer()
        batch_idx = torch.arange(self.batch_size, dtype=torch.long).to(device)

        states = np.array(np.reshape(states, (3, 224, 224)))
        states_tensor = torch.tensor(states, dtype=torch.float)
        states_tensor = torch.unsqueeze(states_tensor, dim=0).to(device)
        # print(f'concat={concat}')
        # print(f'concat.shape = {concat.shape}')
        concat_tensor = torch.stack(concat).to(device)
        # print(np.shape(states_tensor))

        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)

        next_states = np.array(np.reshape(next_states, (3, 224, 224)))
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(device)
        next_states_tensor = torch.unsqueeze(next_states_tensor, dim=0).to(device)
        next_concat_tensor = torch.stack(next_concat).to(device)
        # print(np.shape(next_states_tensor))

        terminals_tensor = torch.tensor(terminals).to(device)

        information_tensor = learn_information_process(information, batch)

        next_information_tensor = learn_information_process(next_information, batch)

        # print(information_tensor)
        # print(information_tensor.shape)

        # print(f'worker_supply ={work_supply_list}')

        with torch.no_grad():
            q_ = self.q_target.forward(next_states_tensor, next_information_tensor,next_concat_tensor)
            max_actions = torch.argmax(self.q_eval.forward(next_states_tensor, next_information_tensor,next_concat_tensor), dim=-1)
            # print(f'terminals_shape = {terminals_tensor.shape}')
            # print(f'q_table={q_}')
            # print(f'q.shape = {q_.shape}')
            # print(f'terminals_tensor= {terminals_tensor}')
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
        q = self.q_eval.forward(states_tensor, information_tensor,concat_tensor)[batch_idx, actions_tensor]

        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
