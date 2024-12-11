FROM python:3.8
WORKDIR /root/rubick

COPY sched sched

RUN cd sched && python3 setup.py bdist_wheel
ARG RUBICK_VERSION=0.0.0
RUN RUBICK_VERSION=${RUBICK_VERSION} pip install sched/dist/*.whl

WORKDIR /root
RUN rm -rf sched/dist

CMD ["python", "-m", "rubick.sched.main"]
