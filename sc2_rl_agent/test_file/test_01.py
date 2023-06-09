import numpy as np
import torch
n=3
my_list = [[] for _ in range(n)]
print(my_list)  # 输出：[[], [], [], []]
information = {"name": 1, "age": 30, "city": 3}

# for i, value in enumerate(information.values()):
#     my_list[i].append(value)
# print(my_list)
# my_list = torch.tensor(my_list)
# print(my_list.shape)
def information_process(information):
    n = len(information)
    information_list = [[] for _ in range(n)]
    for i, value in enumerate(information.values()):
        information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor,(-1,))
    information_tensor =information_tensor.unsqueeze(0)
    return information_tensor
result = information_process(information)

def learn_information_process(information,batch):
    n = len(information[0])
    information_list = [[] for _ in range(n)]
    for j in range(len(batch)):
        for i, value in enumerate(information[j].values()):
            information_list[i].append(value)
    information_tensor = torch.tensor(information_list)
    information_tensor = torch.reshape(information_tensor,(-1,))
    information_tensor =information_tensor.unsqueeze(0)
    return information_tensor