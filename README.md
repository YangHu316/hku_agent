# 📌 HKU-Agent —— 真实可靠的 HKU 智能信息代理（可迁移、低幻觉）

HKU-Agent 是一个基于 LLM 的智能信息系统，通过“页面抓取 + 本地知识库 + Agent 控制”的方式，提供 **真实、可靠、低幻觉** 的香港大学（HKU）相关信息。

系统具备高度可迁移性，只需替换知识库内容即可将本 Agent 用于其他领域（如大学、公司、产品文档等）。
现已推出在线版本！！：http://103.115.64.135:5000
---

## 🧠 核心特点

### ✅ 1. 真实可靠（Real-Source Grounded）

HKU-Agent 不直接让大模型生成内容，而是通过：

* **crawler.py** 抓取真实网页
* **hybrid\_collector.py** 整合信息
* **LLM 仅做总结 + 提炼，不会凭空生成**

确保每一个回答都有真实来源，不依赖幻觉。

---

### ✅ 2. 幻觉极低（Low Hallucination）

在 **hku\_agent.py** 中，通过：

* 明确的提示词模板
* 明确要求“必须基于真实知识库内容”
* 不允许虚构信息
* 无信息则直接响应“未找到”

从架构上减少幻觉，而不是后处理修正。

---

### ✅ 3. 可迁移性极高（Domain-Transferable）

本项目使用的知识库来源于：

```
/knowledgebase/...
```

只要替换这里的文件：

* 可将 HKU Agent → 迁移到：

  * 其他大学
  * 公司知识库
  * 产品文档问答
  * 行业信息系统
  * FAQ / 政策文档助手
* **无需修改核心代码**

---

### ✅ 4. 模块化架构（Easy to Extend）

```
app.py               # 主入口
crawler.py           # 自动爬取真实网页
hku_agent.py         # Agent + LLM 控制逻辑
hybrid_collector.py  # 多路数据整合
index.html           # 前端展示
start.bat            # 一键启动
```

每个模块功能清晰，可以直接扩展或替换。

---

## 🚀 快速开始（Quick Start）

### 1️⃣ 设置 DeepSeek API Key（必须）

Windows（CMD）：

```bat
setx DEEPSEEK_API_KEY "你的 DeepSeek API Key"
```

设置后重新打开 CMD，测试：

```bat
echo %DEEPSEEK_API_KEY%
```

---

### 2️⃣ 安装依赖

如果你有 `requirements.txt`，可直接安装：

```bash
pip install -r requirements.txt
```

如果没有，可按报错提示补装依赖包。

---

### 3️⃣ 启动系统

方式一：双击运行（推荐）

```
start.bat
```

方式二：命令行运行

```bash
python app.py
```

系统将在本地启动服务，并自动加载知识库。

---

## 📚 知识库（Knowledge Base）

HKU-Agent 的所有回答均来自：

```
/knowledgebase/
```

知识库可以包括：

* 官方网页抓取内容
* FAQ
* 结构化资料
* 手册/政策文本
* 你自己添加的 JSON / TXT / HTML

### ⭐ 要迁移到其他领域，只需替换这里的文件，不需要改 Agent 代码本身。

---

## 🧩 Agent 工作机制（hku\_agent.py）

HKU-Agent 使用以下流程确保可靠性：

1. **从知识库检索**
2. **抓取相关网页内容（可选）**
3. **将所有真实信息交给 LLM**
4. **LLM 执行总结、推理，不得虚构**
5. **最终输出结构化、可靠的回答**

提示词中包含：

* 「必须基于真实信息」
* 「不可编造」
* 「未找到则说未找到」
* 「引用来源」

> 你可以在 hku\_agent.py 中自定义行为，如回答格式、内容约束、引用方式等。

---

## 🔧 修改为其他用途（最核心功能）

若你想将 HKU-Agent 用于 “其他大学 / 企业文档 / 产品资料”，只需两步：

### ➤ 第一步：替换知识库内容

把 `/knowledgebase/` 里文件换成你自己的数据。

例如：

```
knowledgebase/
│── faq.json
│── news.html
│── rules.txt
│── handbook.pdf (可解析)
```

### ➤ 第二步（可选）：修改 Agent 提示词

在 `hku_agent.py` 中改：

```python
system_prompt = """你是一个提供真实信息的智能代理...
```

换成：

* 公司知识库助手
* 法规助手
* 产品说明助手
* 医信系统助手（注意医疗合规）

不用动任何底层逻辑。

---

## 🛡️ 安全说明

本项目通过环境变量读取 API Key：

```python
os.getenv("DEEPSEEK_API_KEY")
```

意味着：

* ❌ 仓库中不会包含 API Key
* ❌ start.bat 不会包含 API Key
* ❌ 不会被同步到 GitHub
* ✔ 你本地运行时拥有完整权限
* ✔ 外部用户必须自己配置 key 才能运行

完全符合公开仓库的安全要求。

---

## 🔮 未来扩展

* 向量数据库（FAISS / Chroma）增强检索
* 多模型支持（DeepSeek / OpenAI / Groq）
* Web 前端 UI 美化
* 多用户会话
* 自动更新知识库
* 添加 RAG（检索增强生成）结构

---

## 📄 License

All rights reserved

---


