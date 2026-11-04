# banach in a container: CLI, engine and the OpenAI/Anthropic compatible server.
#
#   docker run --rm -p 8080:8080 -v banach:/data ghcr.io/gautam1858/banach \
#     serve hf:QuantFactory/granite-3.1-1b-a400m-instruct-GGUF/granite-3.1-1b-a400m-instruct.Q8_0.gguf --host 0.0.0.0
#
FROM debian:bookworm AS build
ARG LLAMA_COMMIT=9309bf1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY . .
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp && cd /llama.cpp && git checkout $LLAMA_COMMIT \
 && git apply /src/engine/patches/0001-banach-precision-ladder.patch
RUN cmake -S engine -B /build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/llama.cpp \
      -DBANACH_PORTABLE=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
 && ninja -C /build-engine banach-engine
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/banach /usr/local/bin/banach
COPY --from=build /build-engine/banach-engine /usr/local/bin/banach-engine
ENV BANACH_HOME=/data BANACH_ENGINE=/usr/local/bin/banach-engine
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["banach"]
CMD ["--help"]
