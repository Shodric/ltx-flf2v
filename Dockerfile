# Source only. This image has not been built, and it does not contain weights.
# ComfyUI pin matches the schema read for workflow.py.
FROM runpod/pytorch:1.2.0-cu1281-torch280-ubuntu2404

ARG COMFY_REF=d9238b25d23e0354b0169c8197b63eca7b46443f

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFY_DIR=/opt/ComfyUI \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth 1 origin "${COMFY_REF}" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt \
    && mkdir -p /opt/ComfyUI/input /opt/ComfyUI/output \
        /opt/ComfyUI/models/diffusion_models \
        /opt/ComfyUI/models/text_encoders \
        /opt/ComfyUI/models/vae

WORKDIR /opt/worker
COPY requirements.txt handler.py workflow.py models.py ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8188
ENTRYPOINT []
CMD ["python3", "-u", "/opt/worker/handler.py"]
