# Use a slim Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes the main.py, vercel.json, and the kr_sbert_onnx_int8 directory
# .gitattributes will ensure that the ONNX model pointer is handled correctly by Git LFS
COPY . .

# Hugging Face Spaces automatically exposes port 7860
# No need for an EXPOSE instruction, but it's good practice for documentation
EXPOSE 7860

# Command to run the Uvicorn server on the port expected by Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]