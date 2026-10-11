#!/bin/sh
set -eu

: "${LLE_WEIGHTS_DIR:=/weights}"
mkdir -p "$LLE_WEIGHTS_DIR/pyiqa" "$LLE_WEIGHTS_DIR/torch" "$HOME/.cache"
# pyiqa caches its metric files under ~/.cache/pyiqa; keep them on the weights volume so
# the container runs offline after the first start.
if [ ! -L "$HOME/.cache/pyiqa" ]; then
  rm -rf "$HOME/.cache/pyiqa"
  ln -s "$LLE_WEIGHTS_DIR/pyiqa" "$HOME/.cache/pyiqa"
fi

cmd="${1:-serve}"
case "$cmd" in
  serve)
    if [ "${LLE_AUTO_DOWNLOAD_WEIGHTS:-false}" = "true" ]; then
      python -m scripts.download_weights --prefetch-iqa
    fi
    exec uvicorn app.main:app \
      --host 0.0.0.0 --port "${PORT:-9090}" \
      --workers 1 --no-access-log --timeout-keep-alive 5
    ;;
  download-weights)
    shift
    exec python -m scripts.download_weights "$@"
    ;;
  benchmark)
    shift
    exec python -m scripts.benchmark "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
