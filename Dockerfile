FROM python:3.6

WORKDIR /app

COPY . /app

RUN cd /app && \
    pip install -r requirements.txt

EXPOSE 80

CMD ["python", "/app/app.py"]
