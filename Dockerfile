FROM ghcr.io/johnkell9801-hub/s1r5@sha256:c9edd8addd36896f0c8783d6eb5b92345188b84e5734dca70cc269375ee99397
ENV S1ENG_Q8_RKF=0
ARG ENGINE_HOTKEY=""
ARG ENGINE_PARENTS="[]"
LABEL ai.reax.engine.hotkey="${ENGINE_HOTKEY}" \
      ai.reax.engine.parents="${ENGINE_PARENTS}" \
      ai.reax.engine.ownership="[]" \
      org.opencontainers.image.title="s1-fast engine: uid 164 e51x (RF3 + RFI + q8rf2 + Q8F2, Q8 from layer 0) with RKF off (engcomp v1g)" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-License-1.0"

COPY SOURCES.md /opt/engcomp/SOURCES.md
LABEL org.opencontainers.image.source="https://github.com/MetaTrendLab/engcomp/tree/v1g"
