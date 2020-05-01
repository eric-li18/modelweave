FROM python:3.6

ADD requirements.txt /app
RUN cd /app && \
    pip install -r requirements.txt

RUN git clone git@github.com:eric-li18/modelweave.git

RUN pip install -r requirements.txt

CMD ["app.py"]
