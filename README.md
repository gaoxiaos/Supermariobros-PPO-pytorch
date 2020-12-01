# Supermariobros-PPO-pytorch
rl(ppo) course on super-mario-bros

<img src="/doc/timg.jpeg" width = "250" height = "150" alt="" align=center />

## play with docker (ON your local computer with display),just run:
```
docker run —-gpu all -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay 
```

if you want debuge the code and exec into container ,command like this:
```
docker run —-gpu all -it -v /tmp/.X11-unix:/tmp/.X11-unix registry.cn-shanghai.aliyuncs.com/tcc-public/super-mario-ppo:localdisplay  /bin/bash
```

train the model:
```Python
python ppo_lstm.py
```

test on super-mario-bros(see the video of agent)
```
python test_lstm.py
```


## learn the course in jupyter notebook:
the notebook can be find at course.ipynb、course2.ipynb

## jion the rl Communication group,contact us:
remarks（添加请备注）：github rl
![avatar](/doc/20201201160554.jpg)

## learn more in our DRL Training camp （aliyun tianchi）
you can find some ppo info on https://tianchi.aliyun.com/specials/promotion/aicamprl
