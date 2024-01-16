FROM python:3.9

# RUN pip install Flask==2.0.1 Pillow==8.4.0 opencv-python==4.5.3.56 opencv-contrib-python==4.8.1.78 numpy==1.24.3 

ENV FLASK_ENV=production

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
# Copy the Flask app code to the working directory
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 8080

# Set the entrypoint command to run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
# CMD ["python", "app.py"]
