FROM circleci/python:3.6.8
COPY . /opt/pungi
WORKDIR /opt/pungi
USER root
RUN pipenv install -e . && pipenv install --dev
CMD ["pipenv", "run", "pungi/main.py", "train"]
