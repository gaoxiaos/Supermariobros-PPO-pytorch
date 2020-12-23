# Supermariobros-PPO-pytorch
基于超级玛丽游戏的pytorch版本强化学习实践教程

rl(ppo) course with super-mario-bros

你可以直接在jupyter notebook中开始学习（course.ipynb、course2.ipynb）

<img src="https://img.alicdn.com/tfs/TB1lGFGlIieb18jSZFvXXaI3FXa-254-236.gif" />



## run the code with docker (推荐)
play with docker (ON your local computer with display),just run:

推荐使用docker直接运行，可以无需关注软件环境

```Shell
docker run --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay 
```

if you want debuge the code and exec into container ,command like this:
```Shell
docker run --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay  /bin/bash
```

train the model:

```Python
python ppo_lstm.py
```

test on super-mario-bros(see the video of agent):

```Python
python test_lstm.py
```

## run the code witch conda
```Shell
conda create -n ppo python=3.7
conda activate ppo
```
python request:
```
torch torchvision
gym_super_mario_bros
spinup(要用源码安装：https://spinningup.openai.com/en/latest/user/installation.html)
opencv-python

```
train:
```Python
python ppo_lstm.py
```

test:
```Python
python test_lstm.py
```

## learn the course in jupyter notebook:
the notebook can be find at course.ipynb、course2.ipynb

## jion the rl Communication group,contact us:
remarks（添加请备注）：github rl

<img src="/doc/20201201160554.jpg" width = "200" height = "200" alt="" align=center />

## learn more in our DRL Training camp （aliyun tianchi）
you can find some ppo info on https://tianchi.aliyun.com/specials/promotion/aicamprl
