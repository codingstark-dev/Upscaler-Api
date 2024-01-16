FROM python:3.10-slim

RUN pip install Flask==2.0.1 Pillow==8.4.0 opencv-python==4.5.3.56 opencv-contrib-python==4.8.1.78 numpy==1.24.3 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
ENV FLASK_ENV=production

COPY . /opt/

EXPOSE 80

WORKDIR /opt

ENTRYPOINT ["python", "app.py", "--port", "80"]
