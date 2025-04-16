# Use a Python base image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_deployment.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy Files
COPY codebase/ ./codebase/
COPY data/ ./data/
COPY main.py ./

# Expose the port that the Flask application listens on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=./main.py

# Command to run the Flask application.  Use the standard "flask run" command
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]