# LTX-2.5 one-pass FLF2V worker

Queue worker for a Ramifica scene clip: still S, still E, and an English
`videoPrompt`. One distilled LTX-2.5 pass. Scene audio is muxed into the mp4.
The narrator track is not this model's job.

OpenRouter stays the live video provider. Nothing here is deployed.

## Status

| Check | State |
| --- | --- |
| Source | authored |
| Unit test | `python3 workers/ltx-flf2v/test_workflow.py` |
| Docker image | not built |
| GPU / Comfy `/prompt` | not run |
| Runpod endpoint, volume, secret, job | not created |
| Hub listing | not published |

`ResizeImageMaskNode` uses the dotted prompt keys from ComfyUI
`d9238b25d23e0354b0169c8197b63eca7b46443f` (`resize_type.width`,
`resize_type.height`, `resize_type.crop`). That POST has not been executed.

## Contract

Default clip: **480×864**, **24 fps**, **10 s**, **241 frames** (`8n+1`).
Edges must be multiples of 32, from 64 up to that canvas. `fps` is 1–24.
`duration_seconds` is 1–10. A frame count that is not `8n+1` is rejected.

`prompt_enhance`, an upscaler, a second pass, a raw `workflow`, and base64
stills are refused. Send `start_image_url` and `end_image_url` (http or https).

`action: "validate"` returns this contract and does not start Comfy. It still
cold-starts a GPU if Hub or an endpoint runs it. Do not publish until that
charge is accepted.

Generate output: base64 mp4 (cap 12 MB), width, height, frames, fps, seed,
and `audio: true`. `audio: true` means the graph links audio into
`CreateVideo`. The file has not been probed. A still is not a success.

## Weights

Four files from gated `Lightricks/LTX-2.5`, about 39 GB. Not baked, and not
downloaded inside the job.

- `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` → `models/diffusion_models/`
- `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` → `models/text_encoders/`
- `ltx-2.5-video-vae-bf16.safetensors` → `models/vae/`
- `ltx-2.5-audio-vae-bf16.safetensors` → `models/vae/`

Do not load `gemma4_e2b_it_int8_convrot.safetensors`.

Lookup order: `COMFY_MODEL_DIR`, then the Runpod HF snapshot
`/runpod-volume/huggingface-cache/hub/models--Lightricks--LTX-2.5/snapshots/{hash}/`,
then `/runpod-volume/models` or `/workspace/models`. Missing files fail the job.
The token env is `HUGGINGFACE_ACCESS_TOKEN` (secret name to create later:
`hf_read`). The handler never logs it.

## Suggested endpoint

Not created. Do not paste this image onto `wan22-ti2v` (`er0eouw8ks0w1m`).

- New endpoint, pool `AMPERE_48` (NVIDIA A40), `minCudaVersion` 12.8, allow 12.8 through 13.2
- Workers min 0, max 1, idle 10 s, FlashBoot off
- Container disk 60 GB for the image unpack, not the weights (unverified)
- `EU-SE-1` only if a STANDARD network volume is added
- Volume 100 GB only if the HF cache cannot hold this one repo
- First generate waits for an explicit paid go

Build context, when a build is actually requested:

```bash
docker build -f workers/ltx-flf2v/Dockerfile workers/ltx-flf2v
```

## Hub

`.runpod/hub.json` and `.runpod/tests.json` describe this directory. Hub reads
a repository root. Point Hub at a repo whose root is this worker, not at the
Ramifica app. A GitHub release is what indexes Hub. A commit does not. The
only test sends `action: validate`. No deploy badge.

## License

Worker code license is undecided. No `LICENSE` file is added. ComfyUI is
GPL-3.0. Do not publish this worker as an open API until a license is chosen.
