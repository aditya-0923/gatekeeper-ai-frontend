# Works for Hugging Face Docker Spaces (port 7860) and any other Docker host.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=7860 DATA_DIR=/tmp/gatekeeper
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "server.py"]
