FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir paddlepaddle opencv-python-headless pillow numpy gradio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p output

EXPOSE 7860
CMD ["python", "main.py"]
