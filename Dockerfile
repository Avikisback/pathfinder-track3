# Path Finder, Track 3. Runs the estimator (src/main.py: Python 3 + NumPy, one thread, offline) on one dataset.
#
#   docker build -t pathfinder .
#   docker run --rm --network none -v /path/to/dataset:/data -v "$PWD/out":/out pathfinder /data /out/estimated_trajectory.txt
#
# /data may be a folder that holds imu0/data.csv, a zip of such a folder, or a CSV file.

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src/main.py /app/main.py
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["python3", "/app/main.py"]
