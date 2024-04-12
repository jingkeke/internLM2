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
 
 
