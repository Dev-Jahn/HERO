FROM nvcr.io/nvidia/pytorch:23.01-py3
ARG DEBIAN_FRONTEND=noninteractive

# basic python packages
RUN pip install transformers \
                tensorboardX ipdb lz4 lmdb

####### horovod for multi-GPU (distributed) training #######
# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# captioning

# captioning eval tool (java for PTBtokenizer and METEOR)
RUN apt-get install -y --no-install-recommends openjdk-8-jdk && apt-get clean

# binaries for cococap eval
ARG PYCOCOEVALCAP=https://github.com/tylin/coco-caption/raw/master/pycocoevalcap
RUN mkdir /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/meteor/meteor-1.5.jar -P /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/meteor/data/paraphrase-en.gz -P /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/tokenizer/stanford-corenlp-3.4.1.jar -P /workspace/cococap_bin/

# add new command here

WORKDIR /src
