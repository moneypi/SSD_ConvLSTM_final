下载MOT16.zip
https://pan.baidu.com/s/1XxfqbYcEoWHgi3D2ysqOkQ?pwd=miao

执行`python convert_mot16.py`生成数据集
在没有cuda的时候，执行如下命令进行训练
`python train.py --dataset RDVOC --cuda False --visdom False`
在有cuda的时候，执行如下命令进行训练
`python train.py --dataset RDVOC --cuda True --visdom False`
