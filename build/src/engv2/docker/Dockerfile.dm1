ARG BLINK=engv2-blink:m11
ARG Q35REF=ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588
FROM ${Q35REF} AS q35ref
USER root
RUN find /opt/venv -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.npy' \
      -o -name '*.npz' -o -name '*.gguf' -o -name '*.ckpt' -o -name '*.pkl' -o -name '*.onnx' -o -name '*.h5' \
      -o -name '*.tflite' -o -name '*.msgpack' -o -name '*.pte' \) -print -delete
FROM ${BLINK}
USER root
COPY --from=q35ref /opt/venv /opt/venv
COPY q35-engine /opt/q35/engine
COPY docker/entry_dm.sh /opt/engv2/entry_dm.sh
COPY q35-engine/kern/f8g6.cubin q35-engine/f8g6.py jeffan/f8/engv2_jeffan_f8.py jeffan/f8/jeffan_f8_table_pro6000.json \
     jeffan/f8/install.py q35-engine/kern/f8sk.cubin q35-engine/f8sk.py /opt/engv2/jf8/
RUN python3 /opt/engv2/jf8/install.py
RUN /opt/venv/bin/python -m compileall -q /opt/q35/engine && chmod +x /opt/engv2/entry_dm.sh
COPY docker/ltfz/small-frozen.lt docker/ltfz/vis-frozen.lt /opt/engv2/ltfz/
ENV REAX_LT_SMALL_FROZEN=/opt/engv2/ltfz/small-frozen.lt REAX_LT_VIS_FROZEN=/opt/engv2/ltfz/vis-frozen.lt
ARG MODELS=mammoth,blink,reax1,jeffan,vision
LABEL ai.reax.engine.models="${MODELS}" org.opencontainers.image.title="reax-engv2-dm"
USER 10002:10002
ARG ENGINE_HOTKEY=
LABEL ai.reax.engine.hotkey="${ENGINE_HOTKEY}"
ENTRYPOINT ["/opt/engv2/entry_dm.sh"]
