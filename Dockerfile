FROM python:3.6
WORKDIR /app
RUN git clone git@github.com:eric-li18/modelweave.git
RUN pip install -r requirements.txt