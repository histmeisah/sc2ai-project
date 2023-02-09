# sc2ai-project
this a simple project for  sc2 ai 

this code based on burnysc2 ,use this bot : https://github.com/Dotagoodgogo/sc2AI/blob/master/ and my classmate`s work about D3QN sc2 

this idea comes from https://www.youtube.com/@sentdex



how to run this project? 

First , we should use StarCraft II Editor.exe to download the newest ladder map 

![image](https://user-images.githubusercontent.com/49554454/217539085-d14f0177-33a4-42f1-ac7d-ac9f61ad29f2.png)

when we open this , please log in your blz account and search the map name which you want

![image](https://user-images.githubusercontent.com/49554454/217540537-db80aca9-aec7-4d30-b4f9-f4dc818a1697.png)


than put maps to your sc2 file (if your sc2 file didnt exist , please create a new one) 

![image](https://user-images.githubusercontent.com/49554454/217539085-d14f0177-33a4-42f1-ac7d-ac9f61ad29f2.png)

or you can download maps 



RUN: you can run train.py to train your sc2 ai

INPUT : 3x224x224 mini map and 13 internal parameters about worker_supply,gametime...

OUTPUT :  51 actions 
       such as train_workers,build_supply,...,attack

REINFORCE LEARNING ALGORITHM:
D3QN(Dueling Double DQN)
