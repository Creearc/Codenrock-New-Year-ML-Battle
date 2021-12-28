FROM python:3.6
COPY ./ /app
COPY /home/alexandr/datasets/santas
COPY . .
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python3","run.py"]
