FROM python:3.6

WORKDIR /app

ADD . /app

RUN cd /app && \
    pip install -r requirements.txt

CMD streamlit run --server.port $PORT /app/app.py 