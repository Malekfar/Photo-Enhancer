# Low-light product photo enhancer

FastAPI service that brightens low-light e-commerce photos while preserving structure
(text, logos, edges, textures), color, and highlights.

```
decode (EXIF orientation, ICC -> sRGB, float32 [0,1])
  -> denoise        SCUNet real-image weights | OpenCV NLM fallback   (auto when dark & noisy)
  -> enhance        retinexformer (default, FiveK ckpt) | zerodce | classical
  -> preserve_color luminance-gain transfer (ratio Y_enh/Y_orig, capped)
  -> roll-off       soft-knee shoulder on max(R,G,B), white point -> 1.0 (no hard clip)
  -> highlights     m = guided_filter(smoothstep(t_low, t_high, Y_orig)); out = m*orig + (1-m)*enh
  -> auto-exposure  median luminance -> [0.40, 0.50], gain weighted by (1-m), never darkens by default
  -> sharpen        optional luminance-only unsharp mask (off by default)
  -> strength       out = orig + s*(out - orig)
  -> guard          soft-knee white point chosen so clipped% <= input clipped% + 0.2%
encode (JPEG 4:4:4 q95 | PNG | WebP, sRGB ICC embedded, EXIF stripped by default)
```

## Layout

```
app/
  main.py            FastAPI app, endpoints, error handling
  service.py         model registry (loaded once), concurrency/timeout runner, request job
  config.py          all settings (env vars, prefix LLE_)
  weights.py         weights.yaml parsing, resumable download, SHA256 verification
  evaluation.py      PSNR/SSIM/LPIPS, synthetic darkening, reference scenes, LOL-v2 loader
  models/            architectures vendored from the official repos (unchanged parameter names)
  pipeline/          io, denoise, enhance, highlight, adjust, metrics, tiling, pipeline
  static/index.html  upload page with before/after slider
scripts/             download_weights.py, benchmark.py, entrypoint.sh
tests/               unit, quality, API tests
weights.yaml         pinned weight URLs + SHA256
```

## Run

```bash
docker compose up --build            # CPU, http://localhost:9090
docker compose --profile gpu up --build enhancer-gpu   # CUDA, http://localhost:9091
```

On first start the entrypoint downloads the pinned weights into the `weights` volume and
verifies their SHA256. A checksum mismatch stops the container with an explicit
`CHECKSUM MISMATCH` error. Once the volume is populated the service needs no network
(`LLE_AUTO_DOWNLOAD_WEIGHTS=false` forbids downloads entirely).

Fully offline image (weights baked in):

```bash
docker build --build-arg BAKE_WEIGHTS=1 -t lowlight-enhancer:cpu .
docker run --rm -p 9090:9090 -e LLE_AUTO_DOWNLOAD_WEIGHTS=false lowlight-enhancer:cpu
```

### Weights and the SCUNet pin

All URLs in `weights.yaml` come from the official repositories:

| entry | source | pinned |
|---|---|---|
| `retinexformer_fivek` | caiyuanhao1998/Retinexformer, official Google Drive folder | yes |
| `retinexformer_lolv2_real` | same | yes |
| `zerodce_pp` | Li-Chongyi/Zero-DCE_extension, `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth` | yes |
| `scunet_color_real_psnr` | cszn/SCUNet download script -> cszn/KAIR release v1.0 | **no** |

The SCUNet checksum is not pinned yet (its download could not be completed when the
manifest was written). Unpinned weights are refused by default; until then denoising uses
OpenCV NLM. To pin it (trust-on-first-use):

```bash
docker compose run --rm -e LLE_ALLOW_UNPINNED_WEIGHTS=true enhancer \
  download-weights scunet_color_real_psnr
```

The SHA256 is printed and stored in `/weights/pins.json` and enforced on every later start.
Copy it into `weights.yaml` (`sha256:`) to pin it permanently, then restart.

Check each upstream repository's license before commercial use.

## API

| endpoint | description |
|---|---|
| `POST /v1/enhance` | multipart `file` + params -> image; metrics in `X-Enhance-Metrics` (JSON) |
| `POST /v1/enhance/json` | same -> `{image_base64, metrics, pipeline, timings_ms, params}` |
| `POST /v1/enhance/batch` | multipart `files` (repeat) + params -> ZIP with images and `metrics.json` |
| `GET /health` | 200 when the default method is loaded, else 503 with load failures |
| `GET /v1/models` | loaded backends, checkpoints, denoisers, failures |
| `GET /` | review page with before/after slider |

```bash
# image out
curl -sS -F file=@dark.jpg -F method=retinexformer -F strength=0.9 \
  -D headers.txt -o enhanced.jpg http://localhost:9090/v1/enhance
grep -i x-enhance-metrics headers.txt

# JSON with base64 image and metrics (incl. NIQE/BRISQUE)
curl -sS -F file=@dark.png -F output_format=png -F iqa=true \
  http://localhost:9090/v1/enhance/json | jq '.metrics, .pipeline'

# select the LOL-v2 checkpoint, keep the model's own color rendering
curl -sS -F file=@dark.jpg -F checkpoint=retinexformer_lolv2_real -F preserve_color=false \
  -o out.jpg http://localhost:9090/v1/enhance

# batch -> zip
curl -sS -F files=@a.jpg -F files=@b.jpg -F method=zerodce -o enhanced.zip \
  http://localhost:9090/v1/enhance/batch
```

Errors are JSON: `{"error": {"code", "message", "details", "request_id"}}` with status
400 (invalid image), 413 (file/pixels/batch too large), 415 (unsupported type), 422
(invalid parameter), 503 (backend not loaded), 504 (timeout or no free slot).

### Request parameters

| param | values | default | effect |
|---|---|---|---|
| `method` | `retinexformer`, `zerodce`, `classical` | `retinexformer` | enhancement backend |
| `checkpoint` | `retinexformer_fivek`, `retinexformer_lolv2_real` | `retinexformer_fivek` | retinexformer weights |
| `strength` | 0-1 | 1 | linear blend original -> result |
| `denoise` | `auto`, `on`, `off` | `auto` | auto = median luma < 0.30 and projected noise > 0.02 |
| `denoise_strength` | 0-1 | 1 | residual blend of the denoiser |
| `highlight_low` / `highlight_high` | 0-1, low < high | 0.6 / 0.9 | highlight mask smoothstep thresholds on original Y |
| `preserve_color` | bool | true | apply enhancement as luminance gain only |
| `sharpen` | bool | false | luminance unsharp mask (amount 0.3, sigma 1) |
| `output_format` | `jpeg`, `png`, `webp` | `jpeg` | |
| `quality` | 1-100 | 95 | JPEG/WebP quality (WebP 100 = lossless) |
| `strip_exif` | bool | true | drop EXIF (orientation is always applied to pixels) |
| `iqa` | bool | true | NIQE/BRISQUE before/after (computed at <= 1024 px long side) |

### Tuning guide

- **Too dark after processing**: raise `LLE_AE_TARGET_LOW/HIGH` (e.g. 0.45/0.55) or
  `LLE_AE_MAX_GAIN`. Raising `strength` only helps if it is below 1.
- **Bright areas look flat or washed out**: lower `highlight_low` (e.g. 0.5) so more of the
  original tone is kept; raise `LLE_KNEE` (0.85 -> 0.9) for a later, harder shoulder.
- **Bright areas not lifted enough**: raise `highlight_low`/`highlight_high` (e.g. 0.7/0.95).
- **Color cast from the model**: keep `preserve_color=true` (default). Turn it off only if
  the model's color rendering (e.g. FiveK white balance) is preferred.
- **Plastic / smeared textures**: lower `denoise_strength` (0.5) or set `denoise=off` for
  clean sources; check `pipeline.denoise.projected_noise` in the JSON response.
- **Visible noise**: `denoise=on`; pin SCUNet (it is far better than OpenCV NLM on real noise).
- **Halo-free, structure-exact output on big images**: keep `LLE_LARGE_IMAGE_MODE=guided`
  (model runs at <= `LLE_ENHANCE_MAX_MODEL_PIXELS`, its tone mapping is transferred to the
  full-resolution pixels with an edge-aware local affine model, so all detail comes from the
  original). `tile` runs the model at full resolution (slower, overlap-feathered tiles).
- **CLAHE strength (classical)**: `LLE_CLAHE_CLIP_LIMIT` (2.0; lower = gentler).
- **Throughput**: CPU `LLE_MAX_CONCURRENCY=1` with `LLE_TORCH_THREADS` = cores; GPU 2-4.
  Scale out with more containers instead of uvicorn workers (each worker loads all models).

### Main environment variables

All settings live in `app/config.py`; every field is `LLE_<NAME>`. Lists are JSON.

| var | default | |
|---|---|---|
| `LLE_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:1` |
| `LLE_ENABLED_BACKENDS` | `["retinexformer","zerodce","classical"]` | |
| `LLE_REQUIRED_BACKENDS` | `[]` | names whose load failure aborts startup, e.g. `["retinexformer_fivek"]` |
| `LLE_DEFAULT_METHOD` | `auto` | `auto` picks backend, checkpoint and color mode per image |
| `LLE_RETINEXFORMER_CHECKPOINTS` | `["retinexformer_fivek","retinexformer_lolv2_real"]` | |
| `LLE_MAX_INPUT_MEGAPIXELS` | 24 | checked from the header before decoding |
| `LLE_MAX_UPLOAD_MB` | 60 | |
| `LLE_MAX_BATCH_FILES` | 20 | |
| `LLE_ENHANCE_MAX_MODEL_PIXELS` | 250000 | retinexformer inference resolution cap; CPU needs ~1 GB RAM per 0.2 MP |
| `LLE_LARGE_IMAGE_MODE` | `guided` | or `tile` (`LLE_TILE_SIZE` 512, `LLE_TILE_OVERLAP` 64) |
| `LLE_DENOISE_BACKEND` | `auto` | `scunet`, `opencv` |
| `LLE_KNEE` | 0.85 | soft-knee start |
| `LLE_CLIP_TOLERANCE` | 0.002 | allowed increase of clipped-pixel share |
| `LLE_MAX_GAIN` | 8 | luminance-gain cap for `preserve_color` |
| `LLE_AE_TARGET_LOW` / `HIGH` | 0.40 / 0.50 | |
| `LLE_AE_ALLOW_DARKEN` | false | |
| `LLE_MAX_CONCURRENCY` | 1 | inference slots (semaphore) |
| `LLE_REQUEST_TIMEOUT_S` | 180 | per request, including queueing |
| `LLE_IQA_ENABLED` | true | default for `iqa` and IQA warm-up at startup |
| `LLE_LOG_JSON` / `LLE_LOG_LEVEL` | true / INFO | structured logs with request id |

## Tests

```bash
docker compose --profile test run --rm tests
```

This downloads the pinned weights into the shared volume (if missing), then runs:

- unit tests per stage: decoding (formats, EXIF orientation, ICC, alpha, header-based
  decompression-bomb rejection, truncated files), padding/cropping exactness, tiling identity
  and seam test, guided upsampling, noise estimation, denoisers, every enhancer (shape,
  dtype, range, odd sizes, guided/tile modes), highlight mask/shoulder/guard, auto-exposure,
  unsharp mask, strength;
- quality tests: references (synthetic product scenes + scikit-image photos) darkened with
  gamma 2.2-3.5, exposure scale and Poisson-Gaussian noise, run through the full pipeline per
  backend; asserts PSNR/SSIM/LPIPS improve over the dark input, that the clipped-pixel share
  never exceeds input + 0.2% on any image, and that a well-exposed white background stays
  white. `pytest -s` prints the per-backend numbers;
- API tests (FastAPI TestClient / httpx): all endpoints, formats, batch partial failure,
  validation, size limits, unavailable backend, timeout, request id.

Model-dependent tests are skipped (with the load error as reason) when weights are absent.
LOL-v2: mount the dataset and set `LOL_V2_DIR` to the `Real_captured` directory
(`Test/Low`, `Test/Normal`).

## Benchmark

```bash
docker compose run --rm enhancer benchmark --iqa --json /weights/bench.json
docker compose --profile gpu run --rm enhancer-gpu benchmark --device cuda --latency-mp 1 12 24
# with LOL-v2 (mount it first, e.g. -v /data/LOL-v2:/data/lol:ro)
docker compose run --rm -v /data/LOL-v2:/data/lol:ro enhancer benchmark --lol-dir /data/lol/Real_captured
```

Prints Markdown tables per backend/checkpoint, each with `preserve_color` on and off:
PSNR, SSIM, LPIPS (vs reference), optional NIQE/BRISQUE, worst-case clipped-share
increase, and median full-pipeline latency at the requested megapixel sizes.

### Results

Not measured yet: the numbers have to come from running the benchmark on the target
hardware. Paste the output tables here. Then set `LLE_DEFAULT_METHOD`,
`LLE_RETINEXFORMER_DEFAULT_CHECKPOINT` and the default `preserve_color` from the evidence.
FiveK is the default checkpoint because it is trained on retouched photographs rather than
on LOL's very dark indoor captures.

## Notes and limits

- Output is 8 bit per channel. 16-bit grayscale PNGs are read at full precision. 16-bit RGB
  PNGs are reduced to 8 bit by Pillow on decode.
- Animated WebP/PNG: only the first frame is processed.
- Images with alpha keep it in PNG/WebP. JPEG output is flattened onto white.
- A timed-out request returns 504 immediately. The running computation cannot be
  interrupted, so its slot is freed only when the work finishes, which keeps the server
  from overloading.
- NIQE/BRISQUE are computed on a copy downscaled to `LLE_IQA_MAX_SIDE` (1024). pyiqa caches
  its metric files under `/weights/pyiqa`. If pyiqa cannot load, scores are `null` with
  `iqa_error`, and enhancement is unaffected.
- Retinexformer attention is global, so `tile` mode output differs slightly from a
  full-frame run. `guided` mode has no tiles and therefore no seams.
