FROM machinelearning:2.0
USER root

COPY ./Freeze.txt /workspace/Freeze.txt
RUN  pip3 install -r /workspace/Freeze.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN rm -rf /workspace/*
COPY . /workspace
RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
