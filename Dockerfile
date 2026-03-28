# 使用轻量级 Python 镜像
FROM python:3.11-slim-bookworm

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# 创建必要的目录
RUN mkdir -p static templates core data

# Hugging Face Spaces 默认暴露 7860 端口
EXPOSE 7860

# 运行 FastAPI 应用
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
