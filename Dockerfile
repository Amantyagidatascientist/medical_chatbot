FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.python.org/simple

COPY . .

EXPOSE 8001

CMD ["python", "runner.py"]
