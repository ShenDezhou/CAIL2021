FROM jittor/jittor-cuda
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Beijing /etc/localtime

RUN sed -i 's/archive.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list
#RUN apt-get update
#RUN apt-get upgrade -y
#RUN apt-get remove python python-dev python-opencv -y
#RUN apt-get install python3 python3-dev python3-pip -y
#RUN apt-get install python3-opencv -y

#RUN mkdir -p /root/.pip
#RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#COPY ./Freeze.txt /workspace/requirements.txt
#RUN  pip3 install -r /workspace/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . /workspace
#RUN pip3 install /workspace/torch-1.5.1+cu101-cp36-cp36m-linux_x86_64.whl /workspace/torchvision-0.6.1+cu101-cp36-cp36m-linux_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple/
#RUN rm -f torch-1.5.1+cu101-cp36-cp36m-linux_x86_64.whl torchvision-0.6.1+cu101-cp36-cp36m-linux_x86_64.whl
#RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
WORKDIR /workspace
CMD ['python3']
#ENTRYPOINT ['/workspace/one_for_all_inference.py']