FROM tensorflow/tensorflow:2.4.0

RUN apt-get update && apt-get install -y git

RUN pip install pip==21.0.1

WORKDIR /app

ADD requirements-docker.txt .

RUN pip install -r requirements-docker.txt

ADD . .

RUN git config user.email "bot.guitouni@gmail.com"
RUN git config user.name "Bot G."
