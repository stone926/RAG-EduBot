# 📚 Z.ai 高校专业课智能导学助教系统 (RAG)
欢迎加入本项目！这是一个基于 RAG（检索增强生成）技术开发的高校核心课程（如《操作系统》、《计算机组成原理》）智能答疑系统。本项目完全由纯 Python 编写，未使用笨重的第三方框架，核心检索算法采用了工业级的 FAISS 向量库，前端采用 Streamlit 实现了极简的现代化 Web 交互与流式输出。

## 🚀 快速开始 (Quick Start)
为了确保你能顺利在自己的电脑上跑通本项目，请严格按照以下步骤操作：

### 1.需要安装的python库
- streamlit
- langchain
- faiss-cpu
- numpy
- pypdf
- unstructured
- python-docx
- zai
- spacy

### 2.配置API-key
1. 到[质谱清言官网](https://bigmodel.cn/console/overview)上申请一个API-key
2. 在当前目录新建一个key.txt文件，将自己的API-key写入。

**注意：请不要将自己的API-key推送到仓库中**

### 3.准备知识库
为了避免 Git 仓库臃肿，所有的课件和 PDF 资料不进行代码托管

新建一个名为knowledge的目录，在里面存放对应课程的资料

### 4.生成向量库
目前支持操作系统和计算机组成原理两门课程，二者的向量库要分开生成。

运行embedding.py即可读取knowledge中的文件并生成向量库，注意知识库内容与生成的文件名要匹配，操作系统的向量库叫做vector_db_os.json,计算机组成的向量库叫做vector_db_co.json。

### 5.启动web界面
在终端输入`streamlit run UI.py`即可运行程序

## 代码结构指南
### 1.load_file.py
用于读取knowledge目录中的文件，然后将文本切分成若干个文本块，目前支持读取doxc、txt、pdf三种类型的文件。

### 2.embedding.py
通过调用embedding-3模型，将文本块转化为向量，最终存储在向量库中。

### 3.RAG.py
输入问题，将问题转化为向量并与向量库中的向量计算余弦相似度，返回大于相似度阈值的前五个向量。

### 4.API.py
调用glm-4-flash模型，输入问题后调用RAG进行检索，将检索结果与问题一同发给AI，得到回答

### 5.UI.py
用streamlit搭建的前端界面设计

## 后续开发计划
to be continue...