import os
import matplotlib.pyplot as plt
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + 'is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + 'create successfully!')


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)

def information_process(information):
    n = len(information)
    information_list = [[] for _ in range(n)]
    for i, value in enumerate(information.values()):
        information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor,(-1,))
    information_tensor =information_tensor.unsqueeze(0).to(device)
    return information_tensor
def learn_information_process(information,batch):
    n = len(information[0])
    information_list = [[] for _ in range(n)]
    for j in range(len(batch)):
        for i, value in enumerate(information[j].values()):
            information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor,(-1,))
    information_tensor =information_tensor.unsqueeze(0).to(device)
    return information_tensor
