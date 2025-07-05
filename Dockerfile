FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose the port your app uses
EXPOSE 8000

# Command to run your app
CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8000"]