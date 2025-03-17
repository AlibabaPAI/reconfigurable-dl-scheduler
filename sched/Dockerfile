FROM python:3.8
WORKDIR /root/morphling

COPY sched sched

RUN cd sched && python3 setup.py bdist_wheel
ARG MORPHLING_VERSION=0.0.0
RUN MORPHLING_VERSION=${MORPHLING_VERSION} pip install sched/dist/*.whl

WORKDIR /root
RUN rm -rf sched/dist

CMD ["python", "-m", "morphling.sched.main"]