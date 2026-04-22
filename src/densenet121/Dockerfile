FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

RUN pip install --no-cache-dir \
    monai==1.4.0 \
    nibabel \
    simpleitk \
    itk \
    smdebug \
    tensorboard \
    tqdm

COPY train.py /opt/program/train.py
COPY model_def.py /opt/program/model_def.py
COPY utils.py /opt/program/utils.py
COPY label_dict.json /opt/program/label_dict.json

WORKDIR /opt/program

ENV SAGEMAKER_PROGRAM train.py
