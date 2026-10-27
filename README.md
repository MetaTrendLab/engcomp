# engcomp v1f: uid 14

REAX (Bittensor subnet 92) engine-competition entry, track s1-fast.

- Image: `ghcr.io/elninolab/v1f@sha256:8aef3a88da9fe6a6c8464bb217d9a64051ff0706a5018057e7eacf5a9b889982`
- Committed by uid 14, hotkey `5Gn8txQhUg1NKqu89dmTT4Lxebd6jFBNrov3nj3HZsG7sq83`
- Base image (unchanged): `ghcr.io/johnkell9801-hub/s1r5@sha256:c9edd8addd36896f0c8783d6eb5b92345188b84e5734dca70cc269375ee99397`

## What this entry changes

This entry ships **no compiled code of its own**. Every native file in the image (all `.cubin` / `.so` under
`/opt/s1eng/` and `/opt/reax/`) is the base image's file, byte for byte; their sources and licences are those of the
parents below (label `ai.reax.engine.parents`). On top of the base image the Dockerfile only sets:

```
ENV S1ENG_Q8_RKF=0 S1ENG_Q8_BAYES_P=0.22
```

## Parents (`ai.reax.engine.parents`)

| uid | image digest | files |
|---|---|---|
| 56 | `sha256:96d42d82d646ac19bb4db4f5fbcb4fbe5b2a079e722ed1c1558dc09a46f36953` | /opt/s1eng/, /opt/reax/ |
| 62 | `sha256:8467e642812347a6c34f5c3f26eb7dd42678e4a4a532aad09f3d22d9fa0138a7` | /opt/s1eng/, /opt/reax/ |
| 133 | `sha256:3119bb372381049b82d094f6a855d4e5fe3f1d1e2f257139ae2eb07106f6a9fe` | /opt/s1eng/, /opt/reax/ |
| 161 | `sha256:ba2ae16cfefb327734bc2bc945730e3b2fa016591b06b0f133310baf4b01eeae` | /opt/s1eng/, /opt/reax/ |
| 164 | `sha256:c9edd8addd36896f0c8783d6eb5b92345188b84e5734dca70cc269375ee99397` | /opt/s1eng/, /opt/reax/ |
| 176 | `sha256:e904735e0342ed596a42581d4187631f8434e1f9a3990acd59d533ad86c19d66` | /opt/s1eng/, /opt/reax/ |
| 240 | `sha256:61e29b762e99bb41cebd5a6cd64d8c8e69f9656131637d8eaf05776bbc8a0251` | /opt/s1eng/, /opt/reax/ |
| 246 | `sha256:148b8ecd3656755bea125880c6921fccf8613544e088e066d396862f53936dbc` | /opt/s1eng/, /opt/reax/ |

## Build

```bash
ENGINE_HOTKEY=5Gn8txQhUg1NKqu89dmTT4Lxebd6jFBNrov3nj3HZsG7sq83 ./build.sh <registry>/<name>:v1f
```

`Dockerfile` (comments removed), `build.sh` and `parents.json` here are the files of the build. The image also carries
`/opt/engcomp/SOURCES.md` with a link to this branch when it was built by our submit tool.

Licence: the files of this branch are MIT; the base image and its kernels keep their own licence
(`org.opencontainers.image.licenses` of the image).
