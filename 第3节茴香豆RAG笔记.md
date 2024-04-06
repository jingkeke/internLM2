茴香豆 做智能问答, 
- 配置 good_questions.json
- 配置 bad_questions.json
- 每次对问题评分 ,拒绝评分低的, 回复评分高的 


### 测试1  命令行用 茴香豆 回复问题 

- 命令行评估测试 
```bash
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json

```

![2024-04-07-07-21](https://github.com/jingkeke/internLM2/assets/16113137/6ed775e8-8245-4b79-9954-65a76fb6ef0a)


- 命令行问答
```bash
# 填入问题
sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py

# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone

```
![2024-04-07-07-12](https://github.com/jingkeke/internLM2/assets/16113137/d01d93ed-ff74-44c6-b2dc-3210f2555455)








### 利用 Gradio 搭建网页 Demo

- 利用 Gradio 搭建网页 Demo
```bash
cd /root/huixiangdou
python3 -m tests.test_query_gradio 

```

![2024-04-07-07-45](https://github.com/jingkeke/internLM2/assets/16113137/13f48ac9-3443-4ea9-86a5-36e61c62b913)







