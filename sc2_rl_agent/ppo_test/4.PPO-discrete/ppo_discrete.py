import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from buffer import ReplayBuffer
from efficientnet_pytorch import EfficientNet
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_name('efficientnet-b0').to(device)
        self.classifier = nn.Linear(1000, num_classes).to(device)

    def forward(self, x):
        x = self.efficientnet(x).to(device)
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
        x = torch.as_tensor(x, dtype=torch.float32, device=device).clone().detach()
        x = self.fc(x)
        # x = x.transpose(0, 1)
        # print(f'x_transformer_shape={x.shape}')
        output = self.transformer(x, x)
        # output = output.transpose(0, 1)
        output = self.output(output)
        output = self.softmax(output)
        return output


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.resnet = ResNet18().to(device)
        self.efficientnet = MyEfficientNet().to(device)
        self.transformer = Transformer(input_dim=args.information_dim, output_dim=128).to(device)
        self.concat_transformer = Transformer(input_dim=256, output_dim=256)
        self.lstm = AlphaStarLSTM(input_size=256, hidden_size=12, num_layers=10, output_size=128).to(device)
        self.concat_layer1 = nn.Linear(10 + 128, 256).to(device)
        self.concat_layer2 = nn.Linear(256, 512).to(device)
        self.concat_layer3 = nn.Linear(512, 256).to(device)
        self.concat_data = torch.zeros((1, 256)).to(device)
        self.fc1 = nn.Linear(128, 256).to(device)
        self.fc2 = nn.Linear(256, 64).to(device)
        self.fc3 = nn.Linear(64, args.action_dim).to(device)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh].to(device)  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.concat_layer1)
            orthogonal_init(self.concat_layer2)
            orthogonal_init(self.concat_layer3)

            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, state, information, concat):
        x = torch.relu(self.efficientnet(state))
        information = torch.as_tensor(information, device=device).clone().detach()
        i = torch.relu(self.transformer(information))
        x = torch.cat((x, i), dim=1)
        x = torch.relu(self.concat_layer1(x))
        x = torch.relu(self.concat_transformer(x))
        x = torch.relu(self.concat_layer2(x))
        x = torch.relu(self.concat_layer3(x)).clone().detach().requires_grad_(True)
        self.concat_data = x
        x = x.unsqueeze(0)
        concat = concat.to(device)
        concat = torch.cat((concat, x), dim=0)
        result, _ = self.lstm(concat)
        result = self.activate_func(self.fc1(result)).to(device)
        result = self.activate_func(self.fc2(result)).to(device)
        a_prob = torch.softmax(self.fc3(result), dim=1).to(device)
        return a_prob

    def get_concat_data(self):
        return self.concat_data

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.resnet = ResNet18().to(device)
        self.efficientnet = MyEfficientNet().to(device)
        self.transformer = Transformer(input_dim=args.information_dim, output_dim=128).to(device)
        self.lstm = AlphaStarLSTM(input_size=256, hidden_size=12, num_layers=10, output_size=128).to(device)
        self.concat_layer1 = nn.Linear(10 + 128, 256).to(device)
        self.concat_transformer = Transformer(input_dim=256, output_dim=256)

        self.concat_layer2 = nn.Linear(256, 512).to(device)
        self.concat_layer3 = nn.Linear(512, 256).to(device)
        self.concat_data = torch.zeros((1, 256)).to(device)
        self.fc1 = nn.Linear(128, 256).to(device)
        self.fc2 = nn.Linear(256, 64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh].to(device)  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.concat_layer1)
            orthogonal_init(self.concat_layer2)
            orthogonal_init(self.concat_layer3)

            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state, information, concat):
        x = torch.relu(self.efficientnet(state))
        i = torch.relu(self.transformer(information))
        x = torch.cat((x, i), dim=1)
        x = torch.relu(self.concat_layer1(x))
        x = torch.relu(self.concat_transformer(x))
        x = torch.relu(self.concat_layer2(x))
        x = torch.relu(self.concat_layer3(x)).clone().detach().requires_grad_(True)
        self.concat_data = x
        x = x.unsqueeze(0)
        concat = concat.to(device)
        concat = torch.cat((concat, x), dim=0)
        result, _ = self.lstm(concat)
        result = self.activate_func(self.fc1(result)).to(device)
        result = self.activate_func(self.fc2(result)).to(device)
        v_s = self.fc3(result)
        return v_s

    def get_concat_data(self):
        return self.concat_data

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class PPO_discrete:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args)
        self.critic = Critic(args)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, observation, information,
                 concat):  # When evaluating the policy, we select the action with the highest probability
        observation = np.reshape(observation, (3, 224, 224))
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)  # 把224*224*3的展开

        a_prob = self.actor(state, information, concat).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a
    def save(self, checkpoint_dir, episode):
        actor_checkpoint_file = os.path.join(checkpoint_dir, "Actor_net", f"actor_{episode}.pt")
        critic_checkpoint_file = os.path.join(checkpoint_dir, "Critic_net", f"critic_{episode}.pt")

        self.actor.save_checkpoint(actor_checkpoint_file)
        self.critic.save_checkpoint(critic_checkpoint_file)

    def load(self, checkpoint_dir, episode):
        actor_checkpoint_file = os.path.join(checkpoint_dir, "Actor_net", f"actor_{episode}.pt")
        critic_checkpoint_file = os.path.join(checkpoint_dir, "Critic_net", f"critic_{episode}.pt")

        if os.path.exists(actor_checkpoint_file) and os.path.exists(critic_checkpoint_file):
            self.actor.load_checkpoint(actor_checkpoint_file)
            self.critic.load_checkpoint(critic_checkpoint_file)
        else:
            print(f"Error: Checkpoint files for episode {episode} not found.")
    def choose_action(self, observation, information, concat):
        observation = np.reshape(observation, (3, 224, 224))
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)  # 把224*224*3的展开
        # print(len(information))

        information_tensor =information
        with torch.no_grad():
            dist = Categorical(probs=self.actor(state, information_tensor, concat))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def get_concat_data(self):
        return self.actor.get_concat_data()

    def update(self, trajectory_buffer, total_steps):
        s, i, c, a, a_logprob, r, s_, i_, c_, done = trajectory_buffer.get_all_trajectories_tensors()
        s = s.permute(0, 3, 1, 2)
        s = s.to(device)
        s_ = s_.permute(0, 3, 1, 2)
        s_ = s_.to(device)

        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        print(f's.shape={s.shape}')
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s, i, c)
            vs_ = self.critic(s_, i_, c_)
            deltas = r + self.gamma * (1.0 - done) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(s.size(0))), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index], i[index], c[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
