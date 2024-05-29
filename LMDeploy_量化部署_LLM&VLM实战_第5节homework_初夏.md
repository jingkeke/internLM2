# 第五节课作业（请交到第5节课）



## 基础作业（结营必做）



完成以下任务，并将实现过程记录截图：

- 配置 LMDeploy 运行环境
- 以命令行方式与 InternLM2-Chat-1.8B 模型对话



### 命令行& 接口方式  



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
> ![2024-04-12-08-51](https://github.com/jingkeke/internLM2/assets/16113137/616e96cc-0b7d-4850-ae0b-fdc390da3a14)


### Python代码集成

```python
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

![2024-04-12-09-02](https://github.com/jingkeke/internLM2/assets/16113137/a22dcb55-636f-4c61-ae80-8d97ce907923)



#### 5.2 向TurboMind后端传递参数

- 设置KV Cache占用比例 参数

![2024-04-12-09-04](https://github.com/jingkeke/internLM2/assets/16113137/fe29aa25-1549-491e-b2db-9e09ea1f7903)

```python
from lmdeploy import pipeline, TurbomindEngineConfig
 
# 调低 k/v cache内存占比调整为总显存的 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)
 
pipe = pipeline('/root/models/internlm2-chat-1_8b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```




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


# 







## 进阶作业



完成以下任务，并将实现过程记录截图：

- 设置KV Cache最大占用比例为0.4，开启W4A16量化，以命令行方式与模型对话。（优秀学员必做）
- 以API Server方式启动 lmdeploy，开启 W4A16量化，调整KV Cache的占用比例为0.4，分别使用命令行客户端与Gradio网页客户端与模型对话。（优秀学员必做）
- 使用W4A16量化，调整KV Cache的占用比例为0.4，使用Python代码集成的方式运行internlm2-chat-1.8b模型。（优秀学员必做）
- 使用 LMDeploy 运行视觉多模态大模型 llava gradio demo。（优秀学员必做）
- 将 LMDeploy Web Demo 部署到 [OpenXLab](https://github.com/InternLM/Tutorial/blob/camp2/tools/openxlab-deploy) 。















#【LMDeploy 量化部署 LLM&VLM实战】课程讲解

### 设置KV Cache最大占用比例为0.4，开启W4A16量化，以命令行方式与模型对话。（优秀学员必做）

```bash
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4

```
- 提问截图 
![2024-04-12-09-54](https://github.com/jingkeke/internLM2/assets/16113137/a78f2eb4-d6f5-48de-99e3-bf218c634262)



### 以API Server方式启动 lmdeploy，开启 W4A16量化，调整KV Cache的占用比例为0.4，分别使用命令行客户端与Gradio网页客户端与模型对话。

```bash
# lmdeploy serve api_server -h  #查看参数

lmdeploy serve api_server \
    /root/internlm2-chat-1_8b-4bit \
    --model-format awq \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1 \
    --cache-max-entry-count 0.4


```
- 命令行客户端

```bash
lmdeploy serve api_client http://localhost:23333
 
```
- 截图
  ![2024-04-12-10-04](https://github.com/jingkeke/internLM2/assets/16113137/51a45096-09e0-4b85-8b59-b1912c84cba4)

- Gradio网页客户端
~~ 报错 DeprecationWarning: concurrency_count has been deprecated. Set the concurrency_limit directly on event listeners e.g. btn.click(fn, ..., concurrency_limit=10) or gr.Interface(concurrency_limit=10). If necessary, the total number of workers can be configured via `max_threads` in launch(). ~~

```bash
# lmdeploy serve gradio --help 
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```



### 使用W4A16量化，调整KV Cache的占用比例为0.4，使用Python代码集成的方式运行internlm2-chat-1.8b模型。（优秀学员必做）
```python

from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占比调整为总显存的 40%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.4,model_format='awq')

pipe = pipeline('/root/internlm2-chat-1_8b-4bit',
                backend_config=backend_config)
response = pipe(['青岛是'])

print(response)

 
```

![2024-04-12-10-39](https://github.com/jingkeke/internLM2/assets/16113137/7204ca38-6837-42be-b837-fbe4fb4932f6)



### 使用 LMDeploy 运行视觉多模态大模型 llava gradio demo （优秀学员必做）

 ![2024-04-12-09-18](https://github.com/jingkeke/internLM2/assets/16113137/cd25dd5c-3b22-4f60-9df8-803b709daa70)

### 将 LMDeploy Web Demo 部署到 OpenXLab （OpenXLab cuda 12.2 的镜像还没有 ready，可先跳过，一周之后再来做）

```bash
 
```

 
