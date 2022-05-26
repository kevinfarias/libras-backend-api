FROM tensorflow/tensorflow:latest-devel-gpu

WORKDIR /home/app

RUN apt-get update

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

EXPOSE 9999

COPY . .

RUN pip install -r ./requirements.txt

CMD python ./main/socket/server.py