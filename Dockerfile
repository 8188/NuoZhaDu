FROM python:3.9
LABEL maintainer="jsy"
WORKDIR /app/NUOZHADU
COPY requirements.txt ./
RUN pip install --no-cache-dir -i http://pypi.douban.com/simple --trusted-host pypi.douban.com -r requirements.txt
ENV REDIS_HOST host.docker.internal
ENV MQTT_BROKER_URL host.docker.internal
COPY . /app/NUOZHADU
CMD python iCtr.py