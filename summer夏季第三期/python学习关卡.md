## 任务



####  请用Python实现一个wordcount函数，统计英文字符串中每个单词出现的次数。返回一个字典，key为单词，value为对应单词出现的次数。

```python
def wordcount(text):
    """
    Count the occurrences of each word in a given text.

    Args:
    text (str): The input text to count words from.

    Returns:
    dict: A dictionary where keys are words and values are the counts of those words.
    """
    from collections import Counter
    import re


    # Use regex to find words, considering words as sequences of alphanumeric characters
    # 在上面的代码中，`\b\w+\b` 是一个正则表达式模式：

# - `\b` 代表单词边界（word boundary），也就是说匹配整个单词，而不是部分单词。
# - `\w+` 代表匹配一个或多个字母、数字或下划线字符（word characters）。
# - `re.findall()` 函数会在字符串 `text.lower()` 中找到所有匹配的模式，然后将结果返回为一个列表。

    words = re.findall(r'\b\w+\b', text.lower())

    return Counter(words)



if __name__ == "__main__":
    text = """
                Got this panda plush toy for my daughter's birthday, 
                who loves it and takes it everywhere. It's soft and 
                super cute, and its face has a friendly look. It's 
                a bit small for what I paid though. I think there 
                might be other options that are bigger for the 
                same price. It arrived a day earlier than expected, 
                so I got to play with it myself before I gave it 
                to her.
                """


    print(wordcount(text))

```

- 截图 

![2024-07-15-10-14](assets/2024-07-15-10-14.png)





###  学习 

https://aicarrier.feishu.cn/wiki/VRgCwVAVziIMI9kvvolcA8xUndg

# Python教程内容

https://github.com/charleson98/internlm_python_tutorial/blob/main/internlm_summer_camp_python_tutorial.ipynb













## 在vscode使用命令行进行debug

很多时候我们要debug的不止是一个简单的python文件，而是很多参数，参数中不止会有简单的值还可能有错综复杂的文件关系,甚至debug一整个项目。这种情况下，直接使用命令行来发起debug会是一个更好的选择。

#### vscode设置

vscode也支持通过remote的方法连接我们在命令行中发起的debug server。首先我们要配置一下debug的config。

还是点击VSCode侧边栏的“Run and Debug”（运行和调试)，单击"create a lauch.json file"

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=NmRjMzBhOTQ1YzZiZGQxMzVhZDlkNTk4MjMxMWY0NWFfU3hvOHdmT1NNUG9SNlNyd3VFdG0zZVc5cXdHb3BXTE5fVG9rZW46V2N1SWJqQjBab3Bmc0Z4VW1XWWNQUUdrbkVmXzE3MjEwMDk4NzI6MTcyMTAxMzQ3Ml9WNA)

选择debugger时选择python debuger。选择debug config时选择remote attach就行，随后会让我们选择debug server的地址，因为我们是在本地debug，所以全都保持默认直接回车就可以了，也就是我们的server地址为localhost:5678。

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=NDY0NzJiODE0YTU4MzQ4MTVjM2RlZDRkZGNjZTFkM2ZfTDhPcDZ6WTJvbVR6SEtrYm01TW1qanpIbjVJZXJzakxfVG9rZW46THpaaGI5Y2djb2JQRmV4TzNUOGNkRjlIbk5XXzE3MjEwMDk4NzI6MTcyMTAxMzQ3Ml9WNA)

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=NTU1MTIyNTllMzI0ZWJmMzIyM2FlYTJkN2ZjMTU3M2VfcWFFRjBRaUxCM2kyM2hxRnh3d3JsdzVxNUlhbElCZTdfVG9rZW46R091RmJ4ODY1b1FMeGt4d2pUemMxZTJVbnVoXzE3MjEwMDk4NzI6MTcyMTAxMzQ3Ml9WNA)

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGM0MzM0MzE1ODk1Y2IxZWYyOGVjMGNmOTIwYWQ0YWFfZldqQTBVaVZDV29jYmRPakNoRWhnNGZMb0ppNFpoVmpfVG9rZW46TVJmdGJBb2Njbzh6cWN4NlMwMGN2NXV3bkVnXzE3MjEwMDk4NzI6MTcyMTAxMzQ3Ml9WNA)

配置完以后会打开配置的json文件，但这不是重点，可以关掉。这时我们会看到run and debug界面有变化，出现了debug选项。

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=M2ExMjQ3MjhjMmM4OTcyZjM2NjM4ODlmOWY0MDQ0ZmFfVlNXMUw4dno2c0dzZHVuMWJBTFNEUTQ1MHpoM1RvQlNfVG9rZW46RmVvcmJiTUtBb1RpNFl4TldubWNTSXV6bnBnXzE3MjEwMDk4NzI6MTcyMTAxMzQ3Ml9WNA)

#### debug命令行

现在vscode已经准备就绪，让我们来看看如何在命令行中发起debug。如果没有安装debugpy的话可以先通过pip install debugpy安装一下。

```Shell
python -m debugpy --listen 5678 --wait-for-client ./myscript.py
```

./myscript.py可以替换为我们想要debug的python文件，后面可以和直接在命令行中启动python一样跟上输入的参数。记得要先在想要debug的python文件打好断点并保存。

--wait-for-client 参数会让我们的debug server在等客户端连入后才开始运行debug。在这就是要等到我们在run and debug界面启动debug。

先在终端中发起debug server，然后再去vscode debug页面单击一下绿色箭头开启debug。