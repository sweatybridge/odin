FROM quay.io/basisai/python-cuda:3.9.2-10.1

WORKDIR /data

RUN curl -Lo Imagenet.tar.gz https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz \
    && tar -xvzf Imagenet.tar.gz \
    && rm -rf Imagenet.tar.gz

WORKDIR /models

RUN curl -Lo densenet10.pth.tar.gz https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz \
    && tar -xvzf densenet10.pth.tar.gz \
    && rm -rf densenet10.pth.tar.gz

WORKDIR /code

COPY code/requirements.txt .
RUN pip install -r requirements.txt

COPY code .

CMD ["python", "main.py"]
