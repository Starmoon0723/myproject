FROM harbor.qihoo.net/finloan-infra-library/framework-python3.12-bw:250630111254819-a721a60 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini tzdata locales \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && sed -i 's/^# *\(zh_CN.UTF-8\)/\1/' /etc/locale.gen \
    && sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置locale和less相关环境变量
ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LESSCHARSET=utf-8

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY config.yaml ./
COPY app.py ./
COPY video_due_diligence ./video_due_diligence

# 设置环境变量（生产环境不启用reload）
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 使用 tini 作为入口点，确保信号处理正确
ENTRYPOINT ["/usr/bin/tini", "--"]

# 启动命令：直接使用 uvicorn 运行应用
CMD ["uvicorn", "video_due_diligence.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
