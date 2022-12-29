FROM python:3.9
LABEL maintainer="jsy"
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -i http://pypi.douban.com/simple --trusted-host pypi.douban.com -r requirements.txt
COPY . /usr/src/app
CMD python iCtr.py