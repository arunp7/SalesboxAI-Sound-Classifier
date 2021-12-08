FROM python:3.7
COPY . /salesboxai-sound-classifier
WORKDIR /salesboxai-sound-classifier
EXPOSE 5000
RUN apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev 
RUN pip install -r requirements.txt
RUN pip install tensorflow
ENTRYPOINT ["python"]
CMD ["app.py"]