# Path Finder, Track 3 entry: main.py with the network weights embedded. Build context is
# this folder (src/):
#
#   docker build -t pathfinder src/
#   docker run --rm -v "$PWD/data:/data" pathfinder /data -o /data/estimated_trajectory.txt
#
# /data holds imu0/data.csv (or a .zip of it, or the CSV itself). Only main.py is copied in: it needs nothing else.

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY main.py /app/
ENTRYPOINT ["python3", "/app/main.py"]
CMD ["--help"]
