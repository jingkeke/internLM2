#  笔记

茴香豆 做智能问答 

- 配置 good_questions.json
- 配置 bad_questions.json
- 每次对问题评分 ,拒绝评分低的, 回复评分高的 

#### RAG

RAG （Retrieval Augmented Generation ）是一种结合了检索（Retrieval）和生成（Generation）的技术，旨
在通过利用外部知识库来增强大型语言模型（LLMs）的性能。它通过检索与用户输入相关的信息片段，并结合这些
信息来生成更准确、更丰富的回答。



##### RAG 优化方案

![2024-05-25-22-37](assets/2024-05-25-22-37.png)







#####  向量数据库 

![2024-05-25-22-38](assets/2024-05-25-22-38.png)



##### RAG vs  微调（Fine-tuning） 如何选择 

![2024-05-25-22-40](assets/2024-05-25-22-40.png)

![2024-05-25-22-41](assets/2024-05-25-22-41.png)



##### 评估框架和基准测试

经典评估指标：

- 准确率 （Accuracy）
-  召回率 （Recall）
- F1分数 （F1 Score）
-  BLEU分数（用于机器翻译和文本生成）
- ROUGE分数（用于文本生成
  的评估）

![2024-05-25-22-42](assets/2024-05-25-22-42.png)



## [茴香豆](https://github.com/InternLM/HuixiangDou)

是一个基于LLMs的领或知识功手，由书生浦语团队开发的开源大模型

- 专为即时通讯工具中的群聊场景优化的工作流，提供及时准确的技术支持和自动化问答服务。

- 通过RAG）技木，茴香豆能修理解和高效准确的回应与特定知识

![2024-05-25-22-45](assets/2024-05-25-22-45.png)



###### 茴香豆完整工作流

![2024-05-25-22-47](assets/2024-05-25-22-47.png)



##### rag 综述



- [0 RAG 概述](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#0-rag-概述)

- [RAG 效果比对](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#rag-效果比对)

- [1 环境配置](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#1-环境配置)

- [1.1 配置基础环境](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#11-配置基础环境)
- [1.2 下载基础文件](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#12-下载基础文件)
- [1.3 下载安装茴香豆](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#13-下载安装茴香豆)

- [2 使用茴香豆搭建 RAG 助手](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#2-使用茴香豆搭建-rag-助手)

- [2.1 修改配置文件](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#21-修改配置文件)
- [2.2 创建知识库](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#22-创建知识库)
- [2.3 运行茴香豆知识助手](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#23-运行茴香豆知识助手)

- [3 茴香豆进阶（选做）](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#3-茴香豆进阶选做)

- [3.1 加入网络搜索](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#31-加入网络搜索)
- [3.2 使用远程模型](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#32-使用远程模型)
- [3.3 利用 Gradio 搭建网页 Demo](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#33-利用-gradio-搭建网页-demo)
- [3.4 配置文件解析](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#34-配置文件解析)
- [3.5 文件结构](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#35-文件结构)



## 茴香豆零编程接入微信

参考 [ref ](https://zhuanlan.zhihu.com/p/686579577)    **原理** 是 android 应用会扫描屏幕, 然后 自己判断 是否自动输入内容?  感觉挺牛逼的











