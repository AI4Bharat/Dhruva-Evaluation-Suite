FROM  --platform=linux/amd64 python:3.10.8

# OS Dependencies
RUN apt-get update && \
      apt-get -y install sudo

RUN sudo apt update
RUN sudo apt install -y libb64-dev libsndfile1-dev

# Python dependencies
RUN pip install "tritonclient[all]==2.28.0" tqdm numpy soundfile jiwer==2.5.1 geventhttpclient==2.0.8 pandas indic-nlp-library locust

# Add users
# WIP

# Copy Data
COPY . .

# Run the dev container
CMD sleep 1000m
