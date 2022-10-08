FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# setup basic packages
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip -y

# setup CUDA env
RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"

# install Python and pip
RUN apt-get install python3.7 -y
RUN python3.7 --version
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt-get install python3-distutils -y
RUN python3.7 get-pip.py
RUN apt-get install python3.7-dev -y

# pip install dependencies
RUN pip3 install transformers sentencepiece flask redis torch sacremoses gunicorn

# install redis and configure it
RUN apt-get install redis-server -y
RUN sed -i 's/bind 127.0.0.1 .*/bind 127.0.0.1/g' /etc/redis/redis.conf

# copy ws file
WORKDIR /src
COPY . /src/

# run test.py file to download and start model
RUN python3.7 load.py

# expose port and run server
EXPOSE 8081
CMD service redis-server restart && gunicorn --bind 0.0.0.0:8081 server:app
