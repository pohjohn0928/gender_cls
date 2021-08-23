FROM python:3.8
ENV CUDA_VISIBLE_DEVICES=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD python fastApi.py
