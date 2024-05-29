 第 4 节课作业



记录复现过程并截图

## 基础作业（结营必做）



- 训练自己的小助手认知（记录复现过程并截图）





###  [XTuner 微调个人小助手部分](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)

> 微调 会自己测试结果 

> [!NOTE] 微调认知的步骤
>
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

### 







## 进阶作业



- 将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab（优秀学员必做）
- ### 复现多模态微调（优秀学员必做）



<details>
           <summary style="font-size:20">运行代码</summary>
- 下载项目  `/root/tutorial`

```
cd ~ && git clone https://github.com/InternLM/tutorial -b camp2 && conda activate xtuner0.1.17 && cd tutorial

```

- `/root/tutorial/xtuner/llava/`目录下的文件结构应该是这样：
  - [llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py](./llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py)


```bash
|-- llava_data
|   |-- repeat.py
|   |-- repeated_data.json
|   |-- test_img
|   |   `-- oph.jpg
|   `-- unique_data.json
`-- llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py
```


-  查询xtuner内置配置文件

```
xtuner list-cfg -p llava_internlm2_chat_1_8b
```

- 训练数据

```bash
cd /root/tutorial/xtuner/llava/
xtuner train /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2
```


![2024-05-05-15-08](https://github.com/jingkeke/internLM2/assets/16113137/87a0bb79-0989-4305-997d-c4f5c8bc6a48)


##### 


​        

###### Finetune后 结果对比

> 即：**加载 1.8B 和 Fintune阶段产物 到显存。**

```bash
# 解决小bug
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# pth转huggingface
#xtuner convert pth_to_hf 参数:
#  config                config file name or path.
#  pth_model             pth model file
#  save_dir              the directory to save HuggingFace model


xtuner convert pth_to_hf \
  /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py \
  /root/tutorial/xtuner/llava/work_dirs/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy/iter_1200.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_1200_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_1200_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

</details>



- 输出结果:
  ![2024-05-05-15-16](https://github.com/jingkeke/internLM2/assets/16113137/e841f2db-d344-44e8-92b8-aa0e6b8c9d5f)

![2024-05-05-15-26](https://github.com/jingkeke/internLM2/assets/16113137/5af4f3b4-f819-457c-8f24-b5f8b719dc98)





```bash
```















OpenXLab 部署教程：https://github.com/InternLM/Tutorial/tree/camp2/tools/openxlab-deploy
