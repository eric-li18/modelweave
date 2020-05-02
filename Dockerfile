FROM python:3.6

WORKDIR /app

COPY . /app

RUN cd /app && \
    pip install -r requirements.txt

ENV PORT 8080

CMD ["streamlit", "run", "--server.port", "8080", "/app/app.py"]
