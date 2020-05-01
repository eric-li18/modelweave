FROM python:3.6

WORKDIR /app

COPY . /app

RUN cd /app && \
    pip install -r requirements.txt

ENV PORT 8501

CMD ["streamlit run", "/app/app.py"]
