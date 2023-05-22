FROM python:3.11-slim-bullseye

WORKDIR /app

COPY . /app/

ENV PIPENV_VENV_IN_PROJECT=1

RUN pip install pipenv
RUN pipenv sync

EXPOSE 8050

#CMD pipenv run gunicorn --bind 0.0.0.0:8000 --timeout 120 --workers 2 cbwg.wsgi
CMD pipenv run python index.py