# Use a slim version of the official Python image as a base image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Torch first because it's required but slowed to install
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY model_server/requirements.txt ./

RUN pip install -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Remove as no need after installation
RUN rm -f requirements.txt

COPY model_server/* ./
COPY src ./src
COPY cfg ./cfg

EXPOSE 3000
