FROM  --platform=linux/amd64 python:3.10.8

# OS Dependencies
RUN apt-get update && \
      apt-get -y install sudo

RUN sudo apt update
RUN sudo apt install -y libb64-dev libsndfile1-dev

# Python dependencies
RUN pip install "tritonclient[all]==2.28.0" tqdm numpy soundfile jiwer==2.5.1 geventhttpclient==2.0.2 pandas indic-nlp-library locust

# Add users
# WIP

# Copy Data
# RUN mkdir -p /app/src
# COPY . /app
WORKDIR /app/src

# Run the dev container
CMD sleep 1000m


# docker run --net host --ipc host -p 8089:8089 --rm -v /Users/ashwin/ai4b/perf_testing/Dhruva-Evaluation-Suite:/app locust_test /bin/bash -c "cd /app/src && locust -f /app/src/locust_new.py --users 1 --spawn-rate 1 -H http://host.docker.internal:8001 --model=ASRBatchOffConfModel --scorer=MUCSBatchOffConfScorer"
# docker run --net host --ipc host -p 8089:8089 --rm -v /home/locust/app:/app locust_test /bin/bash -c "cd /app/src && locust -f /app/src/locust_new.py --users 1 --spawn-rate 1 -H http://host.docker.internal:8001 --model=ASRBatchOffConfModel --scorer=MUCSBatchOffConfScorer"