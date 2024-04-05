
##  测试1 请创作一个 300 字的小故事



![2024-04-02-07-07](https://github.com/jingkeke/internLM2/assets/16113137/6a34f0ed-82d7-4e08-8059-12bd95c69279)


##  下载模型 

![2024-04-05-07-22](https://github.com/jingkeke/internLM2/assets/16113137/a48d77f4-e223-425f-8e95-a0c8ad9eb2be)

- huggingface
```python

import os 
from huggingface_hub import hf_hub_download  # Load model directly 

hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json")

```

- modelscope

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')


```


##  测试2    运行微调虚拟人模型  八戒-Chat-1.8B 模型

- 八戒-Chat-1.8B：https://www.modelscope.cn/models/JimmyMa99/BaJie-Chat-mini/summary
- Chat-嬛嬛-1.8B：https://openxlab.org.cn/models/detail/BYCJS/huanhuan-chat-internlm2-1_8b
- Mini-Horo-巧耳：https://openxlab.org.cn/models/detail/SaaRaaS/Horowag_Mini


![2024-04-02-08-33](https://github.com/jingkeke/internLM2/assets/16113137/8f1125a6-e40d-4048-aae2-3a0a2952bafe)

## 测试3 灵笔 

#### 5.3 图文写作实战（开启 50% A100 权限后才可开启此章节）
> 个人感觉很不错 实用性很高
![2024-04-06-07-19](https://github.com/jingkeke/internLM2/assets/16113137/809a26b0-51e8-429a-80df-f3554427ecac)

#### 5.4 图片理解实战
- 测试了几次 作为OCR使用 , 开头40字左右 准确率很高,后面就有点自由发挥... 吐过 调大默认的参数 ,会产生大量的重复. 
![2024-04-06-07-36](https://github.com/jingkeke/internLM2/assets/16113137/27dd7acb-b8b0-4c39-9090-118a021364c6)



