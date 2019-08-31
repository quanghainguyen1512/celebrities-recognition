FROM python:3.6 as backend
# ADD . /usr/src/app/
WORKDIR /usr/src/app/

COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# COPY ./entrypoint-prod.sh /usr/src/app/entrypoint-prod.sh
# RUN chmod +x /usr/src/app/entrypoint-prod.sh

ARG API_URL
ENV API_URL=${API_URL}

COPY . .

# EXPOSE 5000
# ENTRYPOINT [ "python", "app.py" ]
# CMD ["/usr/src/app/entrypoint-prod.sh"]
ENTRYPOINT [ "gunicorn", "-b 0.0.0.0:5000", "wsgi:app" ]
