FROM python:3.7
COPY . /salesboxai-sound-classifier
WORKDIR /salesboxai-sound-classifier
EXPOSE 5000
RUN pip install -r requirements.txt
RUN pip install tensorflow
ENTRYPOINT ["python"]
CMD ["app.py"]