
## 作业 

### 用 Lagent 自定义工具

#### 启动并使用 Lagent Web Demo
![2024-04-18-19-46](https://github.com/jingkeke/internLM2/assets/16113137/d83b298f-aa13-452c-b16d-5601c5edf3e7)

### 用 Lagent 自定义工具

工具调用格式参考: [InternLM/chat/chat\_format\_zh-CN.md at main · InternLM/InternLM · GitHub](https://github.com/InternLM/InternLM/blob/main/chat/chat_format_zh-CN.md)
就是 会包装上 `<|interpreter|>`
```python
 def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return Internlm2Agent(
            llm=model,
            protocol=Internlm2Protocol(
                tool=dict(
                    begin='{start_token}{name}\n',
                    start_token='<|action_start|>',
                    name_map=dict(
                        plugin='<|plugin|>', interpreter='<|interpreter|>'),
                    belong='assistant',
                    end='<|action_end|>\n',
                ), ),
            max_turn=7)
```
- 查询天气 agent
![2024-04-18-20-07](https://github.com/jingkeke/internLM2/assets/16113137/48050591-d4e4-472f-98b7-68276655a511)

### AgentLego：组装智能体“乐高”
- 图像识别 并标注 
![2024-04-18-20-15](https://github.com/jingkeke/internLM2/assets/16113137/9af1628d-c7b4-400b-99c8-d0075209071a)

- 配置调用 internLM2 agent
  -    配置, 感觉目前之前很多的tool 
  ![2024-04-18-20-42](https://github.com/jingkeke/internLM2/assets/16113137/8a10885c-84ac-4598-83ca-8cca7222f237)
  ![2024-04-18-20-46](https://github.com/jingkeke/internLM2/assets/16113137/320ea023-a2d5-430a-9b3b-22c437a8e3a9)
- 图片识别 感觉肝脏之类的没有做训练(想看看医学上的通用性), 其他还不错.
![2024-04-18-20-55](https://github.com/jingkeke/internLM2/assets/16113137/004c502e-bcad-4071-864b-69f3a56a83cf)

#### 用 AgentLego 自定义工具
> 测试 agent 功能没有问题, 虽然图片生成有些没实际用途....
![2024-04-18-21-08](https://github.com/jingkeke/internLM2/assets/16113137/628b55d5-381b-4288-bccf-3fdcf8873eea)





## 多轮调用 测试

- 多轮调用 参考 [InterLM 驱动的 ReAct 智能体](https://lagent.readthedocs.io/zh-cn/latest/get_started/quickstart.html#interlm-react)





## 笔记

### 1. Lagent 简介
>  Lagent 轻量级智能体框架

- [ Lagent文档 ](https://lagent.readthedocs.io/zh-cn/latest/get_started/install.html) [agentlego 文档](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html)
  - [将 agentlego 工具集成到 LangChain 中：](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html#id4) #langchain 
  - [将 agentlego 工具集成到 Transformers agent 中：](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html#transformers-agent)  #transformers_agent  #llm调用function 

  

- demo 详细文档可以访问：
	- [Lagent：轻量级智能体框架](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md)。
	- [AgentLego：组装智能体“乐高”](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md)。

### [Agent 工具能力微调](https://github.com/InternLM/Tutorial/blob/camp2/agent/finetune.md)



















