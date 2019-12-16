FROM tensorflow/tensorflow:2.0.0-gpu-py3
COPY console.py /home/
CMD [ "python3", "/home/console.py"] 