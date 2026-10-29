FROM engv2-dm:18x-ns
USER 10002:10002
COPY SOURCES.md KERNELS.md /opt/engcomp/
LABEL ai.reax.engine.hotkey="5HEmARQrC2Qrp3KciXngp28MoBCoCF2bnBfwGsvcgxUozfUg" \
      ai.reax.engine.parents='[]' \
      ai.reax.engine.ownership='[]' \
      ai.reax.engine.models="mammoth,blink,jeffan,vision" \
      org.opencontainers.image.source="https://github.com/MetaTrendLab/engcomp/tree/v2-dm18n-v2-dm18n" \
      org.opencontainers.image.licenses="MIT AND LicenseRef-REAX-Engine-1.1 AND Apache-2.0" \
      org.opencontainers.image.title="REAX engine v2.0 dm-all entry v2-dm18n-v2-dm18n (uid 208)"
