FROM python:3.10-slim

# 기본 작업 디렉토리 설정
WORKDIR /workspace

# 사전 설치를 위해 requirements 복사
COPY requirements.txt .

# 기본 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch GPU 버전은 별도로 설치 (cu121 기준)
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 나머지 파일 복사
COPY . .

CMD ["bash"]
