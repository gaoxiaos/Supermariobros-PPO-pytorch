From registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch_with_display:1.4-cuda10.1-py3

RUN pip install joblib \
   && apt-get install -y libopenmpi-dev \
   && git clone https://github.com/openai/spinningup.git \
   && cd spinningup \
   && pip install -e .

ADD . .
ENV DISPLAY=:1

CMD [Xvfb :1 -screen 0 1024x768x16 & && x11vnc -forever -passwd 11111111 -display :1 -rfbport 5901 & && python3 test.py]
