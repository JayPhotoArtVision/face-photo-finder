FROM python:3.12-slim

WORKDIR /app

# OpenCV માટે જરૂરી સિસ્ટમ લાઇબ્રેરીઓ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]