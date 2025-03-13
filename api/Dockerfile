# Use a slim version of the official Python image as a base image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# Copy only the requirements file first to leverage Docker layer caching
COPY api/requirements.txt ./

RUN pip install -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Remove as no need after installation
RUN rm -f requirements.txt

# Copy the rest of the application files
COPY api ./api
COPY src ./src
COPY cfg ./cfg

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
