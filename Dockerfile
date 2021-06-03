FROM python:latest

LABEL maintainer="er.akverma8@gmail.com"

ENV PYTHONPATH="/"

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "first_ann_try.py" ]
