# 논문 Ⅵ장 1절 환경 완벽 재현
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 기본 환경
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    openjdk-11-jre \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리
WORKDIR /workspace

# Python 패키지
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter 노트북 포트
EXPOSE 8888 6006

CMD ["/bin/bash"]