### conda

```
conda create --name openmmlab python=3.10 -y
conda activate openmmlab
```

### pytorch

Using pytorch 2.4.1 for cuda 11.8

```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mkl==2024.0
```
(mkl==2024.0是为了消除import torch时出现的libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent错误，详见：https://blog.csdn.net/G_C_H/article/details/143474066)


### mmengine

```
pip install -U openmim
mim install mmengine
```

### mmcv

For cuda 11.8 & pytorch 2.4.1

```
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
```

### mmdet

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install 'setuptools>=64,<69'
pip install -v --no-build-isolation -e .
```

- conda 自带的 setuptools 60.x 太旧不支持 PEP 660，82.x 又移除了 mim 依赖的pkg_resources。安装兼容版本：)
- pip 的默认行为是创建临时隔离环境来执行 setup.py，那个环境里只有 setuptools，没有 torch。所以 setup.py
  一运行就报 ModuleNotFoundError: No module named 'torch'，pip 连这个包有哪些依赖都解析不了。--no-build-isolation 告诉 pip：不要创建隔离环境，直接在当前 conda 环境里执行
  setup.py，这样就能找到你已经装好的 PyTorch。

### mmdet3d

```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v --no-build-isolation -e .
```

### spconv

```
pip install spconv-cu118
```

spconv for cuda 11.8, ref to: https://github.com/traveller59/spconv