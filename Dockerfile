# Dockerfile
ARG TARGETPLATFORM

# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

RUN echo "Building for $TARGETPLATFORM"

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install gcc and other dependencies
RUN apt-get update && apt-get install -y gcc libffi-dev musl-dev

# Install backports.zoneinfo separately
RUN pip install backports.zoneinfo

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN python -m pip install --upgrade streamlit-extras

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run app.py