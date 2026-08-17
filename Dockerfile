# 企业方案多智能体协作系统 · P2 工作流服务
FROM python:3.11-slim

WORKDIR /app

# 先装依赖元数据，利用层缓存
COPY pyproject.toml ./
COPY src ./src
COPY frontend ./frontend

# 可编辑安装：__file__ 仍指向 /app/src，frontend 路径解析正常
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["sh", "-c", "uvicorn p2_agent.main:app --host 0.0.0.0 --port 8000"]
