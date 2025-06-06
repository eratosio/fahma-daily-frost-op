from eratos_docker.run import ModelRunner
from docker import APIClient
import os
import pytest
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta


def run_app(g, night_of, t):
    """Runs the model asynchronously for a given WKT file."""
    load_dotenv()
    docker_client = APIClient()
    runner = ModelRunner(
        "../src",
        docker_client,
    )
    wait_time = 5

    start_date = "2025-04-01"
    end_date = "2025-04-10"
    duration_threshold = 1
    frost_threshold = 0

    try:
        result_docs, model_errors = runner.run_model(
            initial_ports={
                "input_geom": g,
                "input_start_date": start_date,
                "input_end_date": end_date,
                "input_duration_threshold": duration_threshold,
                "input_frost_threshold": frost_threshold,
                "config": {"anything": "e"},
                "secrets": {
                    "id": os.getenv("ERATOS_KEY"),
                    "secret": os.getenv("ERATOS_SECRET"),
                },
            },
            bind_mounts={os.path.realpath('../src') : '/opt/model'})
        return result_docs


    except Exception as e:
        pytest.fail(f"runner.run_model() raised an exception: {e}")


    return
