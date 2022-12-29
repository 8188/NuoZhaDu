## Microservices of NuoZhaDu Hydropower Plant Cylindrical Valve Intelligent Project

### Quick start with docker
```
docker build -t test:1 .
docker run -it -p 8990:8990 --name nzd test:1 /bin/bash
cd app/data/test
python multiprocess_schedule.py
```
- Enter into the container again
```
docker exec -it nzd /bin/bash
```
- Compile *.pyx by yourself
```
cd app/data
python setup.py build_ext -i
```
---
- Having data, you can start it by one step
```
docker-compose up -d
```
- Or by two steps
```
docker build -t test:1 .
docker run -e PYTHONUNBUFFERED=1 -d --name nzd -p 8990:8990 test:1 python iCtr.py
```