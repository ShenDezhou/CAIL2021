FROM machinelearning:1.0
USER root

COPY ./requirements.txt /workspace/Freeze.txt
RUN  pip3 install -r /workspace/Freeze.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN rm -rf /workspace/*
COPY . /workspace
RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
