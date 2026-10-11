# syntax=docker/dockerfile:1.7
# CPU image. Build:  docker build -t lowlight-enhancer:cpu .
#            tests:  docker build --target test -t lowlight-enhancer:test .

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
# No apt: all dependencies install from wheels that bundle their native libraries
# (torch ships libgomp, opencv-python-headless needs no GUI/glib libs).
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
RUN pip install --upgrade pip \
 && pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

COPY requirements.txt requirements-iqa.txt ./
RUN pip install -r requirements.txt \
 && pip install -r requirements-iqa.txt "opencv-python-headless>=4.10,<5" \
 && pip install --no-deps facexlib pyiqa==0.1.16 \
 && (pip uninstall -y opencv-python opencv-contrib-python || true) \
 && pip install --force-reinstall --no-deps "opencv-python-headless>=4.10,<5" \
 && python -c "import torch, cv2, fastapi, pyiqa; print('torch', torch.__version__, 'cv2', cv2.__version__)"

# ---------------------------------------------------------------- runtime base
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/home/app \
    LLE_WEIGHTS_DIR=/weights \
    TORCH_HOME=/weights/torch \
    HF_HOME=/weights/hf \
    OMP_NUM_THREADS=4
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && mkdir -p /weights && chown app:app /weights
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY --chown=app:app weights.yaml pytest.ini ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
RUN chmod +x scripts/entrypoint.sh

# Optional: bake weights into the image (fully offline image, no volume needed).
#   docker build --build-arg BAKE_WEIGHTS=1 -t lowlight-enhancer:cpu .
ARG BAKE_WEIGHTS=0
ARG ALLOW_UNPINNED_WEIGHTS=false
USER app
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
      LLE_ALLOW_UNPINNED_WEIGHTS=${ALLOW_UNPINNED_WEIGHTS} LLE_LOG_JSON=false \
      sh scripts/entrypoint.sh download-weights --prefetch-iqa; \
    fi

# ---------------------------------------------------------------- test
FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY --chown=app:app tests ./tests
USER app
ENTRYPOINT ["/srv/scripts/entrypoint.sh"]
CMD ["pytest", "-q"]

# ---------------------------------------------------------------- final (default target)
FROM runtime AS final
EXPOSE 9090
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9090/health', timeout=5).status == 200 else 1)"
ENTRYPOINT ["/srv/scripts/entrypoint.sh"]
CMD ["serve"]
