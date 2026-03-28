# 📅 年月日知识智能专家 (Time Expert AI)

这是一个基于 RAG（检索增强生成）与确定性日期算法构建的智能体，旨在帮助小学生和家长轻松学习年月日相关知识。

## 🌟 核心特性

- **双引擎驱动**：结合了 100% 准确的 Python 日期算法与强大的 LLM（DeepSeek/OpenAI）讲解能力。
- **现代化 UI**：基于 FastAPI + Vue 3 + Tailwind CSS 构建的毛玻璃风格界面，支持响应式与扫码分享。
- **双角色模式**：提供“小学生”和“家长”两种模式，AI 会根据角色调整语气和深度。
- **离线支持**：即使在没有 API Key 的情况下，也能通过本地算法和知识库提供核心服务。

## 📂 项目结构

```text
.
├── core/               # 核心逻辑 (Agent, 日期算法)
├── data/               # 知识库数据 (Markdown 格式)
├── static/             # 静态资源
├── templates/          # UI 模板 (Vue 3 + Tailwind)
├── .env                # 配置文件 (API Keys, 端口等)
├── Dockerfile          # 云端部署镜像定义
├── requirements.txt    # 项目依赖
└── server.py           # 服务入口
```

## 🚀 本地运行

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **配置 API Key**（可选）：
   在 `.env` 文件中填入你的 `DEEPSEEK_API_KEY`。

3. **启动服务**：
   ```bash
   python server.py
   ```
   访问 `http://localhost:8000` 即可开始体验。

## ☁️ 云端部署 (Hugging Face Spaces)

本应用已完美适配 Hugging Face Spaces (Docker SDK)：

1. 在 Hugging Face 创建一个新的 Space，SDK 选择 **Docker**。
2. 上传除 `.env` 以外的所有文件。
3. 在 Space 设置中添加 Secret: `DEEPSEEK_API_KEY`。
4. 构建完成后，即可通过公网域名访问并生成分享二维码。

## 📖 知识库

系统默认加载 `data/knowledge_base.md`。你可以通过修改此文件来扩展智能体的知识储备。

## 📄 许可

MIT License
