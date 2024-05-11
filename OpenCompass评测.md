# OpenCompass评测 [OpenCompass github](https://github.com/open-compass/opencompass)

## 作业:使用 OpenCompass评测

**遇到错误请运行：**

```bash
pip install -r requirements.txt
pip install protobuf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

```

### 命令行快速评测

OpenCompass 预定义了许多模型和数据集的配置，你可以通过 工具 列出所有可用的模型和数据集配置。

```bash
# 列出所有配置
# python tools/list_configs.py
# 列出所有跟 llama (模型)及 ceval（数据集） 相关的配置
python tools/list_configs.py llama ceval
```













- 运行   在数据集上测试 感觉下面的命令没用GPU资源.....

```bash

cd ~/opencompass
conda activate opencompass
export MKL_SERVICE_FORCE_INTEL=1

python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b   --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs device_map='auto' trust_remote_code=True  --max-seq-len 2048   --max-out-len 16   --batch-size 8    --num-gpus 1 

python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug


＃python run.py configs/eval_demo.py

```

- 命令 解释 
```bash
python run.py --datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace 模型路径
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 1024 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug \ 
--reuse  latest  # 断点续跑 ...
```


####  结果  

> 上面的配置 我自己跑 老是没有结果 .....报错 05/04 18:46:48 可能是 nvidia 驱动的问题    - OpenCompass - ERROR - /root/opencompass/opencompass/tasks/openicl_eval.py - _score - 241 - Task [opencompass.models.huggingface.HuggingFace_Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct/ceval-high_school_physics]: No predictions found.

 ![2024-04-22-17-08](https://github.com/jingkeke/internLM2/assets/16113137/7c079449-ac43-4a9f-a66d-dcd0ee254147)

 > 默认输出 outputs/defaults/时间戳下


```bash

ceval-physician                                 -          -         -       -
04/22 17:05:28 - OpenCompass - INFO - write summary to /root/opencompass/outputs/default/20240422_170005/summary/summary_20240422_170005.txt
04/22 17:05:28 - OpenCompass - INFO - write csv to /root/opencompass/outputs/default/20240422_170005/summary/summary_20240422_170005.csv

```




### 快速评测

#### config 快速评测

在 `config` 下添加模型配置文件 `eval_llama3_8b_demo.py`

```python
from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets

datasets = [*mmlu_datasets]

from opencompass.models import HuggingFaceCausalLM

models = [
dict(
type=HuggingFaceCausalLM,
abbr='Llama3_8b', # 运行完结果展示的名称
path='/root/model/Meta-Llama-3-8B-Instruct', # 模型路径
tokenizer_path='/root/model/Meta-Llama-3-8B-Instruct', # 分词器路径
model_kwargs=dict(
device_map='auto',
trust_remote_code=True
),
tokenizer_kwargs=dict(
padding_side='left',
truncation_side='left',
trust_remote_code=True,
use_fast=False
),
generation_kwargs={"eos_token_id": [128001, 128009]},
batch_padding=True,
max_out_len=100,
max_seq_len=2048,
batch_size=16,
run_cfg=dict(num_gpus=1),
)
]
```



运行`python run.py configs/eval_llama3_8b_demo.py`




### 查看支持的数据集和模型
- 列出所有跟 internlm 及 ceval 相关的配置
```bash
python tools/list_configs.py internlm ceval

```

### 自定义数据集评测配置[ref](https://blog.csdn.net/nlpx2000/article/details/138042822)

介绍自定义ceval数据集评测配置方法，详细数据集配置请参见opencompass官方文档。

第一步：opencompass/config/database/ceval目录下新建或者修改ceval文件，按照现有文件格式修改！以ceval_gen_5f30c7.py文件为例修改或新建：注意红框处的修改。




第二步：opencompass/ opencompass/database/目录下新建或者修改ceval文件，按照现有文件格式修改加载数据集！以ceval.py文件为例修改或新建，文件中的新建类，在第一步文件中必须引入该文件中实现的模块。



第三步：opencompass/ opencompass/database/_init_.py文件中添加第二步中的py文件，注册实现的模块。



第四步：opencompass/config/database/ceval/ceval_gen.py文件中修改为第一步中的数据集文件名称：



第五步：执行命令：

```bash

python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug

```


## 介绍 
[OpenCompass github](https://github.com/open-compass/opencompass)

由 OpenCompass 贡献者曹茂松、刘卓鑫带来【OpenCompass 大模型评测实战】课程内容的讲解
视频地址：https://www.bilibili.com/video/BV1Pm41127jU/
课程文档：https://github.com/InternLM/Tutorial/blob/camp2/opencompass/readme.md


#### 功能
- 支持 vllm 提高推理速度 
- 任务切分 (并行)
![2024-04-22-16-35](https://github.com/jingkeke/internLM2/assets/16113137/812bbe89-2a68-4f9a-8421-d6a6f9335831)


#### 支持模型:
- HuggingFace
- API
- 自定义



#### CompassHub：高质量评测基准社区

![2024-04-22-16-27](https://github.com/jingkeke/internLM2/assets/16113137/73dac1e1-30f4-4bcd-8d96-c0881152701b)


####  T-Eval：工具agent 能力评测基准

![2024-04-22-16-29](https://github.com/jingkeke/internLM2/assets/16113137/93468382-bfd1-4162-94f8-c58bd55b1458)

