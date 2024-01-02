# GraphCast-torch

本次比赛是基于AI的区域天气预测大赛，提供10年的再分析数据，通过输入历史70个大气变量数据，预测华东区域未来1-5天的5个地面变量。

## GraphCast模型

论文天气需求预测代码实现, 模型采用了三个 V100-32GB 卡数据并行训练，使用 cudnn12.0 和 torch2.0 。

模型训练数据采用比赛初赛和复赛的训练全部数据，并使用了地势数据和海路掩码两个常量数据。

### Requirements
* python 3.8.10
* Ubuntu 20.04.1
* torch 2.0.1+cu118
* Pandas 2.0.1
* torch-scatter 2.1.1
* torch-geometric 2.3.1
* timm 0.9.2
* xarray[complete] 2023.1.0

To install all dependencies:

`pip install -r requirements.txt`

### Usage
* 首先要用 `pretrain_graphcast.py` 和 `utils/params.py` 预训练预测 4 个时间步，训练 70 个 epoch，warmup epoch 设置为 5，学习率使用 cosine 函数，最大学习率设置为5e-4。
```
python pretrain_graphcast.py
``` 

* 其次要用 `finetuning_graphcast.py`  和 `utils/params.py` 微调预测 8 个时间步，训练10个epoch左右，学习率设置为 1e-5 或者 1e-6，得到最优模型。
``` 
python finetuning_graphcast.py
``` 
* 最后，模型继续微调预测 20 个时间步，训练不到 2 到 3 个 epoch，学习率设置为 2e-7，得最终模型。

### 其他：

* `utils/` 文件夹下存放模型使用到的工具、评价指标和默认参数。
* `data_factory/` 文件夹下存放模型需要构建的图结构，以及模型的 DataLoader。
* `model/` 文件夹下存放 graphcast 模型。

