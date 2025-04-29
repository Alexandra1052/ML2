# Imagine de bază oficială cu Python
FROM python:3.11-slim

# Setăm directorul de lucru
WORKDIR /app

# Copiem fișierele necesare
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul fișierelor în container
COPY . .

# Comanda implicită
CMD ["python", "predictor.py"]
