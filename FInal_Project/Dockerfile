# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
