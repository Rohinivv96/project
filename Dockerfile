FROM python:3.9.0
RUN apt-get -y update  && apt-get -y install curl && apt-get -y --no-install-recommends install \ 
  python3-dev \
  build-essential \
  ffmpeg \
  libsm6 \
  libxext6 \
  tesseract-ocr \
  imagemagick
  
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
#EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["sh", "-c", "streamlit run --server.port $PORT main.py"]
