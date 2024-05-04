
# XTuner 微调 LLM：1.8B、多模态、Agent 
[文档](https://github.com/InternLM/Tutorial/tree/camp2/xtuner)


###  [XTuner 微调个人小助手部分](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)
> 微调 会自己测试结果 

- 使用 deepspeed 来加速训练 
```bash
# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2

```
![2024-04-12-11-49](https://github.com/jingkeke/internLM2/assets/16113137/2f176858-0191-4413-82a2-e821136fa2b9)

最终输出

![2024-04-12-12-12](https://github.com/jingkeke/internLM2/assets/16113137/88ef8434-13f9-41a4-a601-eb3bb80acf73)

```bash
04/12 11:57:11 - mmengine - INFO - Iter(train) [720/768]  lr: 2.1272e-06  eta: 0:00:27  time: 0.7224  data_time: 0.0059  memory: 5661  loss: 0.0103
04/12 11:57:18 - mmengine - INFO - Iter(train) [730/768]  lr: 1.3493e-06  eta: 0:00:21  time: 0.7234  data_time: 0.0056  memory: 5661  loss: 0.0132
04/12 11:57:25 - mmengine - INFO - Iter(train) [740/768]  lr: 7.4681e-07  eta: 0:00:16  time: 0.7251  data_time: 0.0058  memory: 5661  loss: 0.0127
04/12 11:57:32 - mmengine - INFO - Iter(train) [750/768]  lr: 3.2080e-07  eta: 0:00:10  time: 0.7275  data_time: 0.0070  memory: 5661  loss: 0.0091
04/12 11:57:40 - mmengine - INFO - Iter(train) [760/768]  lr: 7.2010e-08  eta: 0:00:04  time: 0.7342  data_time: 0.0071  memory: 5661  loss: 0.0178
04/12 11:57:46 - mmengine - INFO - after_train_iter in EvaluateChatHook.
04/12 11:57:47 - mmengine - INFO - Sample output:
<s><|User|>:请你介绍一下你自己
<|Bot|>:我是jzy的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

04/12 11:57:48 - mmengine - INFO - Sample output:
<s><|User|>:你是谁
<|Bot|>:我是jzy的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

04/12 11:57:49 - mmengine - INFO - Sample output:
<s><|User|>:你是我的小助手吗
<|Bot|>:是的</s>
 
```



### 参考 llama3 的微调 [ref](https://github.com/SmartFlowAI/Llama3-Tutorial/blob/main/docs/agent.md)

- 权重转换为 HuggingFace 格式，并合并到原权重中。

```bash
# 转换权重
xtuner convert pth_to_hf ~/Llama3-Tutorial/configs/llama3-agentflan/llama3_8b_instruct_qlora_agentflan_3e.py \
    ~/llama3_agent_pth/iter_18516.pth \
    ~/llama3_agent_pth/iter_18516_hf
```

- 指令合并权重：

```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/model/Meta-Llama-3-8B-Instruct \
    /share/new_models/agent-flan/iter_2316_hf \
    ~/llama3_agent_pth/merged
```

###  运行微调后的模型 Lagent Web Demo



因为我们在微调前后都需要启动 Web Demo 以观察效果，因此我们将 Web Demo 部分单独拆分出来。

首先我们先来安装 lagent。

```
pip install lagent
```



然后我们使用如下指令启动 Web Demo：

```
streamlit run ~/Llama3-Tutorial/tools/agent_web_demo.py 微调前/后 LLaMA3 模型路径
```

