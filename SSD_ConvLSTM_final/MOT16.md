下载MOT16.zip
执行python convert_mot16.py
在没有cuda的时候，执行如下命令进行训练
`python train.py --dataset RDVOC --cuda False --visdom False`
在有cuda的时候，执行如下命令进行训练
`python train.py --dataset RDVOC --cuda True --visdom False`