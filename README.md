## 基于生成对抗网络的车牌重构算法


## 1.为什么需要进行重构？
>近年来，由于深度学习具有泛化性强、自学习目标特征等特点，因此在车牌图像识别问题中获得了较好的效果。但是在实际场景中由于原始车牌图像存在模糊、污损、扭曲、倾斜的问题造成车牌字符关键信息缺失，由此造成深度特征提取网络难以准确提取车牌识别的关键特征，这个问题导致常规的基于深度学习的车牌识别方法效果较差，进而需要首先对车牌进行重构，再进行识别。
><div align=center><img width="188" height="48" src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/test/%E7%9A%96AEA718.jpg"/></div>
><div align=center><img width="188" height="48"src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/output/%E7%9A%96AEA718.jpg"/></div>
>
## 2.如何使用？
>```
>python detect.py
>```

## 3.实验平台及实验数据集
>实验平台搭载Inter Xeon E5 2650处理器，376GB内存，4张NVIDIA 2080Ti 12G显卡；深度学习框架采用pytorch-1.8,以及NVIDIA公司CUDA11.2的GPU运行平台以及cuDNN8.0深度学习GPU加速库。实验数据集采用**中国城市停车数据集(Chinese City Parking Dataset, CCPD)**和**西安建筑科技大学停车场数据集(XAUAT-Parking)**。

## 4.评价指标
>实验采用峰值信噪比(PSNR)与结构相似性(SSIM)进行定量分析
<div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/PSNR.png"/></div> 
<div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/SSIM.png"/></div> 

## 5.效果展示
**①CCPD Weather子数据集效果**  
><div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/4.png"/></div>
**①XAUAT-Parking 数据集效果**  
><div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/1.png"/></div>
><div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/3.png"/></div>
><div align=center><img src="https://github.com/dbwaax/Licence-plate-reconstrution/blob/main/image/2.png"/></div>

## License

[CC0 1.0 (Public Domain)](LICENSE.md)
