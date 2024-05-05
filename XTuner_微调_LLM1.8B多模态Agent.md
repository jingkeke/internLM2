
# XTuner 微调 LLM：1.8B、多模态、Agent 
[文档](https://github.com/InternLM/Tutorial/tree/camp2/xtuner)


###  [XTuner 微调个人小助手部分](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)
> 微调 会自己测试结果 

> [!NOTE] 微调认知的步骤
> 1. 我们首先是在 GitHub 上克隆了 XTuner 的源码，并把相关的配套库也通过 pip 的方式进行了安装。
> 2. 然后我们根据自己想要做的事情，利用脚本准备好了一份关于调教模型认识自己身份弟位的数据集。
> 3. 再然后我们根据自己的显存及任务情况确定了使用 InternLM2-chat-1.8B 这个模型，并且将其复制到我们的文件夹里。
> 4. 最后我们在 XTuner 已有的配置文件中，根据微调方法、数据集和模型挑选出最合适的配置文件并复制到我们新建的文件夹中。



- XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

开箱即用意味着假如能够连接上 Huggingface 以及有足够的显存，其实就可以直接运行这些配置文件，XTuner就能够直接下载好这些模型和数据集然后开始进行微调
```
# 列出所有内置配置文件
# xtuner list-cfg

# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b
```

- 使用 deepspeed 来加速训练 
```bash
# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2

```
![2024-04-12-11-49](https://github.com/jingkeke/internLM2/assets/16113137/2f176858-0191-4413-82a2-e821136fa2b9)

#### 使用模型验证 测试微调结果

- 最终输出



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
- xtuner 内置 模型进行对话
```bash
# 与模型进行对话
xtuner chat /root/ft/final_model --prompt-template internlm2_chat

```

### 图像微调

-  查询xtuner内置配置文件
```
xtuner list-cfg -p llava_internlm2_chat_1_8b
```

- 训练数据格式 
```
[
  {
    "id": "000000033471",
    "image": "coco/train2017/000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  },
```






## 文档 
<details>
<summary><b>常用超参等参数介绍</b></summary>

**常用超参**

| 参数名                  | 解释                                                     |
| ----------------------- | -------------------------------------------------------- |
| **data_path**           | 数据路径或 HuggingFace 仓库名                             |
| **max_length**          | 单条数据最大 Token 数，超过则截断                         |
| **pack_to_max_length**  | 是否将多条短数据拼接到 max_length，提高 GPU 利用率        |
| **accumulative_counts** | 梯度累积，每多少次 backward 更新一次参数                  |
| **sequence_parallel_size** | 并行序列处理的大小，用于模型训练时的序列并行              |
| **batch_size**          | 每个设备上的批量大小                                      |
| **dataloader_num_workers** | 数据加载器中工作进程的数量                                |
| **max_epochs**          | 训练的最大轮数                                             |
| **optim_type**          | 优化器类型，例如 AdamW                                    |
| **lr**                  | 学习率                                                    |
| **betas**               | 优化器中的 beta 参数，控制动量和平方梯度的移动平均         |
| **weight_decay**        | 权重衰减系数，用于正则化和避免过拟合                      |
| **max_norm**            | 梯度裁剪的最大范数，用于防止梯度爆炸                      |
| **warmup_ratio**        | 预热的比例，学习率在这个比例的训练过程中线性增加到初始学习率 |
| **save_steps**          | 保存模型的步数间隔                                         |
| **save_total_limit**    | 保存的模型总数限制，超过限制时删除旧的模型文件             |
| **prompt_template**     | 模板提示，用于定义生成文本的格式或结构                    |
| ...... | ...... |

> 如果想把显卡的现存吃满，充分利用显卡资源，可以将 `max_length` 和 `batch_size` 这两个参数调大。
</details>

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

