#【LMDeploy 量化部署 LLM&VLM实战】课程讲解

[课程视频]([url](https://www.bilibili.com/video/BV1tr421x75B/)
[课程文档]([url](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/README.md))
 

## 命令行& 接口方式  



```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)
```

![2024-04-12-07-36](https://github.com/jingkeke/internLM2/assets/16113137/e813d1f6-2451-457c-b132-c6466500e8fb)


### 使用LMDeploy与模型对话


![2024-04-12-07-40](https://github.com/jingkeke/internLM2/assets/16113137/ac4cc47c-f0af-4f90-80bc-a90cab4ce8e2)


### 2 命令行客户端连接API服务器
![2024-04-12-08-44](https://github.com/jingkeke/internLM2/assets/16113137/e4dfd6c0-edc6-4739-9d90-6105b5267489)

- 道德约束 有点强......
![2024-04-12-08-53](https://github.com/jingkeke/internLM2/assets/16113137/b7a89511-56e0-4a49-b38b-656ac16aaf95)


### 4.3 网页客户端连接API服务器
> 可以长上下文, 可以多轮会话 
![2024-04-12-08-51](https://github.com/jingkeke/internLM2/assets/16113137/616e96cc-0b7d-4850-ae0b-fdc390da3a14)


## Python代码集成

```python
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```
![2024-04-12-09-02](https://github.com/jingkeke/internLM2/assets/16113137/a22dcb55-636f-4c61-ae80-8d97ce907923)



### 5.2 向TurboMind后端传递参数
- 设置KV Cache占用比例 参数

![2024-04-12-09-04](https://github.com/jingkeke/internLM2/assets/16113137/fe29aa25-1549-491e-b2db-9e09ea1f7903)




## 视觉多模态大模型llava
> 用到了项目  https://github.com/haotian-liu/LLaVA

有点没搞懂 没理解是什么约定,用的哪个配置文件,为什么会用项目的llava目录 
```bash
# 会安装llava 到  /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/llava ,但是没理解是什么约定,用的哪个配置文件,为什么会用项目的llava目录

 pip install git+https://github.com/haotian-liu/LLaVA.git@4e2277a060da264c4f21b364c867cc622c945874

```
- 提示词用中文可能有问题(没有用中文训练)  ,需要训练 ,

![2024-04-12-09-18](https://github.com/jingkeke/internLM2/assets/16113137/cd25dd5c-3b22-4f60-9df8-803b709daa70)


## 定量比较LMDeploy与Transformer库的推理速度差异
```python
# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response, history = model.chat(tokenizer, inp, history=history)
    total_words += len(response)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```


# LMDeploy 介绍 
LMDeploy是涵盖了LLM 任务全套轻量化、部署和服务解决方案的集成功能包，TurboMind是LMDeploy的一个推理引擎，是一个子模块。LMDeploy也可以使用pytorch作为推理引擎。


#### LMDeploy不仅支持运行InternLM系列大模型，还支持其他第三方大模型。支持的模型列表如下：

Model	Size
Llama	7B - 65B
Llama2	7B - 70B
InternLM	7B - 20B
InternLM2	7B - 20B
InternLM-XComposer	7B
QWen	7B - 72B
QWen-VL	7B
QWen1.5	0.5B - 72B
QWen1.5-MoE	A2.7B
Baichuan	7B - 13B
Baichuan2	7B - 13B
Code Llama	7B - 34B
ChatGLM2	6B
Falcon	7B - 180B
YI	6B - 34B
Mistral	7B
DeepSeek-MoE	16B
DeepSeek-VL	7B
Mixtral	8x7B
Gemma	2B-7B
Dbrx	132B

### TurboMind
TurboMind是LMDeploy团队开发的一款关于LLM推理的高效推理引擎，它的主要功能包括：LLaMa 结构模型的支持，continuous batch 推理模式和可扩展的 KV 缓存管理器。

TurboMind推理引擎仅支持推理TurboMind格式的模型。因此，TurboMind在推理HF格式的模型时，会首先自动将HF格式模型转换为TurboMind格式的模型。该过程在新版本的LMDeploy中是自动进行的，无需用户操作。


