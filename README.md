# sc2ai-project
this a simple project for  sc2 ai 

this code based on burnysc2 ,use this bot : https://github.com/Dotagoodgogo/sc2AI/blob/master/ and my classmate(jiaxiang lee)`s work about D3QN sc2 

this idea comes from https://www.youtube.com/@sentdex



how to run this project? 

First , we should use StarCraft II Editor.exe to download the newest ladder map 

![image](https://user-images.githubusercontent.com/49554454/217539085-d14f0177-33a4-42f1-ac7d-ac9f61ad29f2.png)

when we open this , please log in your blz account and search the map name which you want

![image](https://user-images.githubusercontent.com/49554454/217540537-db80aca9-aec7-4d30-b4f9-f4dc818a1697.png)


then put maps to your sc2 file (if your sc2 file didnt exist , please create a new one) 

![image](https://user-images.githubusercontent.com/49554454/217539085-d14f0177-33a4-42f1-ac7d-ac9f61ad29f2.png)

or you can download maps 



RUN: you can run train.py to train your sc2 ai

INPUT : 3x224x224 mini map and 13 internal parameters about worker_supply,gametime...

OUTPUT :  51 actions 
       such as train_workers,build_supply,...,attack

REINFORCE LEARNING ALGORITHM:
D3QN(Dueling Double DQN)

# 2023/3/12
add new network structure by ChatGPT,now it has efficientnet+transformer+lstm+d3qn

add new zergagent,but not fit new network

add new informations with protoss agent 

# 2023/6/9
Due to a busy academic schedule, I haven't had enough time to update this project as frequently as I'd like. In fact, the codes I'm uploading now were completed three months ago.

In this update, I've incorporated a PPO agent. However, there are some bugs identified during the training phase, which I will adjust promptly. This PPO agent is powered by ppo-max, an open-source PPO code. I would like to extend my gratitude for their significant contribution, which you can find here: https://github.com/Lizhi-sjtu/DRL-code-pytorch.

Additionally, we've obtained more information from burnysc2, the details of which I will introduce in the upcoming update.

Lastly, I've added some rule-based agents for the three races: Protoss, Zerg, and Terran. Please note that there are still some bugs with the Terran agent which have not been resolved yet."

I hope this revised update log suits your needs better. Feel free to reach out if you need further assistance.


