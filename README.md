[comment]: <> (# DPVO)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> DPVO (中文注释版~仅供个人学习记录用)
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/pdf/2208.04726">Paper</a> 
  | <a href="https://github.com/princeton-vl/DPVO">Original Github Page</a>
  | <a href="https://blog.csdn.net/gwplovekimi/article/details/139436796?spm=1001.2014.3001.5501">CSDN 配置教程</a>
  </h3>
  <div align="center"></div>


<br>

* 单纯运行demo，参数：矫正文档、数据路径、是否可视化，以及stride。同时需要下载权重模型
~~~
python demo.py \
    --imagedir=<path to image directory or video> \
    --calib=<path to calibration file> \
    --viz # enable visualization
    --stride=5 #？？？
~~~

* 训练模型，参数 --name=<your name>（用于存放log file），训练的时候模型每10K代就会验证一次
~~~
python train.py --steps=240000 --lr=0.00008 --name=<your name>
~~~
