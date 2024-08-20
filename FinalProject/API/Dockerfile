FROM python:3.11-slim

# Create the working directory of the container
RUN mkdir app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY src .

# Make port 8000 available to the world outside this container
EXPOSE 8000


CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "-p", "8000", "--log-level", "info", "--reload"]