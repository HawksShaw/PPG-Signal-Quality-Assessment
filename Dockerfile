#Define Python version
FROM python:3.13-slim
#Set working dir inside the container
WORKDIR /app
#Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
#Copy requirements
COPY requirements.txt .
#Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt
#Copy code
COPY . .
#Create folders for storage
RUN mkdir -p object_store
#Add Env
ENV PYTHONPATH=/app:/app/src
#Expose the port
EXPOSE 8000
#Show command for server start
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]