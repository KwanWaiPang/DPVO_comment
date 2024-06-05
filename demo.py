import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad() #装饰器，表示该函数不会计算梯度
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None #初始化slam为空
    queue = Queue(maxsize=8) #创建一个队列，队列的最大长度为8，应该是用于多线程操作？

    # 读取数据集，根据数据集的类型，选择不同的读取方式（图片或视频）
    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start() #启动进程，启动了一个进程并从队列中获取数据进行处理

    while 1:
        # 从队列中获取数据
        (t, image, intrinsics) = queue.get()
        if t < 0: break #如果t小于0，跳出循环

        # 将图像数据从 NumPy 数组转换为 PyTorch 张量，并改变维度顺序（从 HWC 到 CHW）。
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        # 将内参数据从 NumPy 数组转换为 PyTorch 张量。
        intrinsics = torch.from_numpy(intrinsics).cuda()

        # 初始化SLAM对象（DPVO），并传入配置对象 cfg、网络模型 network、图像的高度和宽度、是否可视化。等参数
        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        # 数据再次转移到GPU上（重复了？）
        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics) #调用DPVO对象的__call__方法，传入时间戳、图像数据和内参数据
    
    #更新12次？？？然后等待子进程结束 
    for _ in range(12):
        slam.update()

    reader.join() #等待子进程结束
    print('finished!!!')

    return slam.terminate()

# 主函数
if __name__ == '__main__':
    import argparse #导入argparse模块，该模块用于解析命令行参数。
    # 创建一个ArgumentParser对象，parser。这个对象包含将要添加的所有命令行参数及其处理方式。
    parser = argparse.ArgumentParser()

    # 添加命令行参数以及其默认值
    parser.add_argument('--network', type=str, default='dpvo.pth') # 网络模型
    parser.add_argument('--imagedir', type=str) #数据集路径
    parser.add_argument('--calib', type=str) #相机内参，校准参数
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml") #VO配置文件
    parser.add_argument('--timeit', action='store_true') #如果命令行中包含这个参数，为True，否则为False。
    parser.add_argument('--viz', action="store_true") #如果命令行中包含这个参数，为True，否则为False。
    
    # 解析命令行参数并将结果赋值给args。
    args = parser.parse_args()

    # 在运行时将指定的配置文件内容合并到程序的配置对象cfg中
    # cfg 是一个配置对象，用来存储程序运行时所需的各种配置参数
    cfg.merge_from_file(args.config)#args.config就是VO的配置文件，通过merge_from_file函数将配置文件中的内容合并到程序的配置对象cfg中。

    print("Running with config...")
    print(cfg)

    # 运行主函数
    run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)


        

