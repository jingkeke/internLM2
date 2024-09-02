# [**Lagent 自定义你的 Agent 智能体** ](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Lagent/task.md#lagent-%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BD%A0%E7%9A%84-agent-%E6%99%BA%E8%83%BD%E4%BD%93)

> [任务](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Lagent/task.md)、[文档](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Lagent/readme.md)、[视频](https://www.bilibili.com/video/BV18M4m1y7sm/)
- conda 环境可以 这样?
```bash
conda activate /share/pre_envs/icamp3_rag 
```

## 基础任务（完成此任务即完成闯关）


- 使用 Lagent 自定义一个智能体，并使用 Lagent Web Demo 成功部署与调用，记录复现过程并截图。


![](assets/2024-08-20-11-13.png)







- 直接用lagent 


```bash
lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --model-name internlm2_5-7b-chat

# lagent 前端 
streamlit run examples/internlm2_agent_web_demo.py

```

 ![](assets/2024-08-20-10-53.png)




## 资料 学习
##### [Agent 工具能力微调](https://github.com/InternLM/Tutorial/blob/camp2/agent/finetune.md) 可以在自己的代码中根据模型的输出来实现调用函数的逻辑  
> 还没出来   2024-08-20 10:28


###  lagent

- Lagent 中关于工具部分的介绍文档位于 [https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html](https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html) 。


> **第三期没有单独提 agentlego ,是准备合并到 agent 里面? 


###  [agentlego 文档](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html)

- [将 agentlego 工具集成到 LangChain 中：](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html#id4) #langchain 
- [将 agentlego 工具集成到 Transformers agent 中：](https://agentlego.readthedocs.io/zh-cn/latest/get_started.html#transformers-agent)  #transformers_agent  #llm调用function 



### 3.基于 Lagent 自定义智能体 ,使用 Lagent 自定义工具主要分为以下几步：
>参考 [ref](https://zhuanlan.zhihu.com/p/715243451)

1. 继承 `BaseAction` 类
2. 实现简单工具的 `run` 方法；或者实现[工具包](https://zhida.zhihu.com/search?q=%E5%B7%A5%E5%85%B7%E5%8C%85&zhida_source=entity&is_preview=1)内每个子工具的功能
3. 简单工具的 `run` 方法可选被 `tool_api` 装饰；工具包内每个子工具的功能都需要被 `tool_api` 装饰

实现一个调用 MagicMaker API 以完成文生图的功能。

#### 1)创建工具文件

```text
cd /root/agent_camp3/lagent
touch lagent/actions/magicmaker.py
```

#### 2)创建`/root/agent_camp3/lagent/lagent/actions/magicmaker.py`

```text
import json
import requests

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class MagicMaker(BaseAction):
    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
    
    @tool_api
    def generate_image(self, keywords: str) -> dict:
        """Run magicmaker and get the generated image according to the keywords.

        Args:
            keywords (:class:`str`): the keywords to generate image

        Returns:
            :class:`dict`: the generated image
                * image (str): path to the generated image
        """
        try:
            response = requests.post(
                url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
                data=json.dumps({
                    "official": True,
                    "prompt": keywords,
                    "style": self.style,
                    "poseT": False,
                    "aspectRatio": self.aspect_ratio
                }),
                headers={'content-type': 'application/json'}
            )
        except Exception as exc:
            return ActionReturn(
                errmsg=f'MagicMaker exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        image_url = response.json()['data']['imgUrl']
        return {'image': image_url}
```

#### 3）修改`/root/agent_camp3/lagent/examples/internlm2_agent_web_demo.py`来适配我们的自定义工具

```text
from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
+ from lagent.actions.magicmaker import MagicMaker
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol

...
        action_list = [
            ArxivSearch(),
+             MagicMaker(),
        ]
```