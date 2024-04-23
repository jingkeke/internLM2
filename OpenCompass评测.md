# OpenCompass评测 [OpenCompass github](https://github.com/open-compass/opencompass)

## 作业:使用 OpenCompass评测



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

 ![2024-04-22-17-08](https://github.com/jingkeke/internLM2/assets/16113137/7c079449-ac43-4a9f-a66d-dcd0ee254147)

 > 默认输出 outputs/defaults/时间戳下


```bash

ceval-physician                                 -          -         -       -
04/22 17:05:28 - OpenCompass - INFO - write summary to /root/opencompass/outputs/default/20240422_170005/summary/summary_20240422_170005.txt
04/22 17:05:28 - OpenCompass - INFO - write csv to /root/opencompass/outputs/default/20240422_170005/summary/summary_20240422_170005.csv

```




##### 查看支持的数据集和模型
- 列出所有跟 internlm 及 ceval 相关的配置
```bash
python tools/list_configs.py internlm ceval

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


