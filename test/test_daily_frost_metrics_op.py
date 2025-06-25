import logging

from eratos_docker.run import ModelRunner
from docker import APIClient
import os
import pytest
from dotenv import load_dotenv
import json
from shapely import wkt
from eratos.dsutil.netcdf import gridded_geotime_netcdf_props
from src.daily_frost_metrics_on_demand import daily_frost_metrics
import time
from memory_profiler import memory_usage
import numpy as np
from shapely.geometry import MultiPoint
from eratos.creds import AccessTokenCreds
from eratos.adapter import Adapter
import threading
import psutil
import matplotlib.pyplot as plt



def test_run_app():
    """Runs the model asynchronously for a given WKT file."""
    load_dotenv()
    docker_client = APIClient()
    runner = ModelRunner(
        "../src",
        docker_client,
    )
    wait_time = 5
    g = "MULTIPOLYGON (((136.173976025327 -33.0823350440668,136.238068475586 -33.0865101897404,136.23454984744 -33.1405244726167,136.170457397181 -33.1363493269432,136.173976025327 -33.0823350440668)))"
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

def generate_random_convex_polygon(center_x, center_y, radius_km, n_points=10):
    # Approximate degrees per km at mid-latitude (not precise for large areas)
    deg = radius_km / 111.0
    angles = np.sort(np.random.rand(n_points) * 2 * np.pi)
    radii = np.random.rand(n_points) * deg

    while True:

        points = [
            (
                center_x + r * np.cos(a),
                center_y + r * np.sin(a)
            )
            for r, a in zip(radii, angles)
        ]

        polygon = MultiPoint(points).convex_hull
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_valid:
            break

    return polygon.wkt

def get_geometry(size):
    if size == "small":
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=3)
    elif size == "medium":
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=6)
    elif size == "large":
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=9)

def benchmark_run(model_func, geom, start_date, end_date, frost_threshold, duration_threshold, secret):

    def wrapped():

        return model_func(
        geom = geom,
        start_date = start_date,
        end_date = end_date,
        frost_threshold = frost_threshold,
        duration_threshold = duration_threshold,
        secret = secret
    )

    start = time.time()
    mem_usage = memory_usage(wrapped, max_iterations=1)
    elapsed = time.time() - start
    peak_mem = max(mem_usage)

    return elapsed, peak_mem


def test_local_frost_metrics_op_geom_size():
    f = open("../src/secret.json")
    data = json.load(f)

    eratos_key = data['eratos_key']
    eratos_secret = data["eratos_secret"]

    secret = {'id': eratos_key,
              'secret': eratos_secret}
    start_date = "2024-04-01"
    end_date = "2025-04-10"
    sizes = ["small", "medium", "large"]
    frost_threshold = 1
    duration_threshold = 1

    # for size in sizes:
    for size in sizes:
        geom = get_geometry(size)
        elapsed, peak_mem = benchmark_run(daily_frost_metrics, geom, start_date,
                                          end_date, frost_threshold,
                                          duration_threshold, secret)

        print(f"{size.capitalize()} geometry:")
        print(f"  Runtime: {elapsed:.2f} s")
        print(f"  Peak Memory: {peak_mem:.1f} MB\n")


def get_start_date(size):
    if size == 'small':
        return "2024-03-20"
    elif size == 'medium':
        return "2024-2-20"
    elif size == 'large':
        return "2023-12-10"

def test_local_frost_metrics_op_date_span():
    f = open("../src/secret.json")
    data = json.load(f)

    eratos_key = data['eratos_key']
    eratos_secret = data["eratos_secret"]

    secret = {'id': eratos_key,
              'secret': eratos_secret}
    end_date = "2025-04-10"
    sizes = ["small", "medium", "large"]
    frost_threshold = 1
    duration_threshold = 1
    geom = get_geometry('small')
    # for size in sizes:
    size = 'large'
    start_date = get_start_date(size)
    elapsed, peak_mem = benchmark_run(daily_frost_metrics, geom, start_date,
                                      end_date, frost_threshold,
                                      duration_threshold, secret)

    print(f"{size.capitalize()} geometry:")
    print(f"  Runtime: {elapsed:.2f} s")
    print(f"  Peak Memory: {peak_mem:.1f} MB\n")


def test_mem_usage_threading():
    memory_log = []
    time_log = []
    monitoring = True

    def monitor_memory(interval=5):
        process = psutil.Process(os.getpid())
        start_time = time.time()
        while monitoring:
            current_time = time.time() - start_time
            mem = process.memory_info().rss / 1024 ** 2  # Convert to MB
            print(f"[MEMORY] RSS: {mem:.2f} MB")
            time_log.append(current_time)
            memory_log.append(mem)
            time.sleep(interval)

    sizes = ["small"]
    for size in sizes:
        memory_log = []
        time_log = []
        monitoring = True

        memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        memory_thread.start()

        f = open("../src/secret.json")
        data = json.load(f)

        eratos_key = data['eratos_key']
        eratos_secret = data["eratos_secret"]

        secret = {'id': eratos_key,
                  'secret': eratos_secret}
        end_date = "2025-04-10"

        frost_threshold = 1
        duration_threshold = 1
        geom = get_geometry(size)
        start_date = get_start_date(size)

        daily_frost_metrics(
            geom = geom,
            start_date = start_date,
            end_date = end_date,
            frost_threshold = frost_threshold,
            duration_threshold = duration_threshold,
            secret = secret
        )

        monitoring = False
        memory_thread.join(timeout=1)

        # Plot the memory usage
        plt.figure(figsize=(10, 5))
        plt.plot(time_log, memory_log, label='Memory (MB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage Over Time - Size {size}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"memory_usage_comparison_size_{size}.png", dpi=300)  # Save as high-res PNG

        plt.show()
