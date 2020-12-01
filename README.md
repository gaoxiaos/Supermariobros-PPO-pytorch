# Supermariobros-PPO-pytorch
rl(ppo) course on super-mario-bros

<img src="/doc/timg.jpeg" width = "300" height = "200" alt="" align=center />

# play with docker (ON your local computer with display),just run:
docker run —-gpu all -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay 

if you want debuge the code and exec into container ,command like this:
docker run —-gpu all -it -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay  /bin/bash

train the model:
'''python
python ppo_lstm.py
'''

test on super-mario-bros(see the video of agent)
'''
python test_lstm.py
'''

# play with code


# learn the course in jupyter notebook:
the notebook can be find at course.ipynb

# jion the rl Communication group,contact us:
webcode:

# learn more in our DRL Training camp （aliyun tianchi）
you can find some ppo info on www.tianchi.aliyun.com
