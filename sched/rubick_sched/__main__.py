import asyncio
import kubernetes_asyncio as kubernetes
import logging
import prometheus_client

from rubick_sched.controller import JobsController

logging.basicConfig()
kubernetes.config.load_incluster_config()
prometheus_client.start_http_server(9091)

controller = JobsController()

loop = asyncio.get_event_loop()
loop.run_until_complete(
    controller.run(),
)
loop.close()
