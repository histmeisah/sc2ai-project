import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        self.s = []
        self.i = []
        self.c = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.i_ = []
        self.c_ = []
        self.done = []
        self.count = 0

    def store(self, args, s, i, c, a, a_logprob, r, s_, i_, c_, done):
        self.s.append(torch.tensor(s, dtype=torch.float).unsqueeze(0))

        self.i.append(i.unsqueeze(0))
        self.c.append(c.detach().unsqueeze(0))
        self.a.append(torch.tensor(a, dtype=torch.long).unsqueeze(0))
        self.a_logprob.append(torch.tensor(a_logprob, dtype=torch.float).unsqueeze(0))
        self.r.append(torch.tensor(r, dtype=torch.float).unsqueeze(0))
        self.s_.append(torch.tensor(s_, dtype=torch.float).unsqueeze(0))
        self.i_.append(i_.unsqueeze(0))
        self.c_.append(c_.detach().unsqueeze(0))
        self.done.append(torch.tensor(done, dtype=torch.float).unsqueeze(0))
        self.count += 1
        print(f's.shape={s.shape}')
    def reset(self):
        self.s = []
        self.i = []
        self.c = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.i_ = []
        self.c_ = []
        self.done = []
        self.count = 0

class TrajectoryReplayBuffer:
    def __init__(self, args):
        self.trajectories = []
        self.count = 0

    def store_trajectory(self, trajectory):
        s = torch.cat(trajectory.s, dim=0)
        i = torch.cat(trajectory.i, dim=0)
        c = torch.cat(trajectory.c, dim=0)
        a = torch.cat(trajectory.a, dim=0)
        a_logprob = torch.cat(trajectory.a_logprob, dim=0)
        r = torch.cat(trajectory.r, dim=0)
        s_ = torch.cat(trajectory.s_, dim=0)
        i_ = torch.cat(trajectory.i_, dim=0)
        c_ = torch.cat(trajectory.c_, dim=0)
        done = torch.cat(trajectory.done, dim=0)

        self.trajectories.append((s, i, c, a, a_logprob, r, s_, i_, c_, done))
        self.count += 1
    def get_training_data(self):
        if self.count > 0:
            s, i, c, a, a_logprob, r, s_, i_, c_, done = self.trajectories.pop(0)
            self.count -= 1
            return s, i, c, a, a_logprob, r, s_, i_, c_, done
        else:
            raise ValueError("No trajectory data available.")

    def get_all_trajectories(self):
        if self.count>0:
            all_trajectories = []
            for trajectory in self.trajectories:
                s, i, c, a, a_logprob, r, s_, i_, c_, done = trajectory
                all_trajectories.append((s, i, c, a, a_logprob, r, s_, i_, c_, done))

            return all_trajectories
        else:
            raise ValueError("No trajectory data available.")

    def get_all_trajectories_tensors(self):
        s_all, i_all, c_all, a_all, a_logprob_all, r_all, s__all, i__all, c__all, done_all = [], [], [], [], [], [], [], [], [], []

        for trajectory in self.trajectories:
            s, i, c, a, a_logprob, r, s_, i_, c_, done = trajectory
            s_all.append(s)
            i_all.append(i)
            c_all.append(c)
            a_all.append(a)
            a_logprob_all.append(a_logprob)
            r_all.append(r)
            s__all.append(s_)
            i__all.append(i_)
            c__all.append(c_)
            done_all.append(done)

        return (torch.cat(s_all), torch.cat(i_all), torch.cat(c_all), torch.cat(a_all), torch.cat(a_logprob_all), torch.cat(r_all),
                torch.cat(s__all), torch.cat(i__all), torch.cat(c__all), torch.cat(done_all))
