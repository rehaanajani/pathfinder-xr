# Path Finder, Track 3 entry. Build context is this folder (src/):
#
#   docker build -t pathfinder src/
#   docker run --rm -v "$PWD/data:/data" pathfinder /data -o /data/estimated_trajectory.txt
#
# /data holds imu0/data.csv (or a .zip of it, or the CSV itself). The default runs the C++ entry with the learned
# velocity network. The Python reference: docker run --rm --entrypoint python3 -v ... pathfinder /app/main.py /data ...

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ make zlib1g-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY main.cpp Makefile ./
RUN make pathfinder

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends zlib1g python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/pathfinder /app/pathfinder
COPY main.py velnet.py velnet_weights.bin velnet_weights_world.bin /app/
ENTRYPOINT ["/app/pathfinder"]
CMD ["--help"]
