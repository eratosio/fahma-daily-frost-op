import json
import logging
import math
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pytest
from as_models.ports import DOCUMENT_PORT, INPUT_PORT, OUTPUT_PORT
from as_models.testing import Context
from docker import APIClient
from dotenv import load_dotenv
from eratos.adapter import Adapter
from eratos.creds import AccessTokenCreds
from eratos_docker.run import ModelRunner
from memory_profiler import memory_usage
from shapely.geometry import MultiPoint

from src.daily_frost_metrics_on_demand import daily_frost_metrics
from senaps_wrapper import eratos_operator_wrapper

load_dotenv()

def test_run_app():
    """Runs the model asynchronously for a given WKT file."""
    load_dotenv()
    docker_client = APIClient()
    runner = ModelRunner(
        "src",
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
                "input_geom": json.dumps(g),
                "input_start_date": json.dumps(start_date),
                "input_end_date": json.dumps(end_date),
                "input_duration_threshold":json.dumps(duration_threshold),
                "input_frost_threshold": json.dumps(frost_threshold),
                "config": {"anything": "e"},
                "secrets": {
                    "id": os.getenv("ERATOS_KEY"),
                    "secret": os.getenv("ERATOS_SECRET"),
                },
            },
            bind_mounts={os.path.realpath('src') : '/opt/model',
                         os.path.realpath('lib/clearnights/clearnights/'): '/usr/local/lib/python3.10/dist-packages/clearnights/',
                         os.path.realpath('lib/clearnights_op/src/clearnights_on_demand/'): '/usr/local/lib/python3.10/dist-packages/clearnights_on_demand/'})
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
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=6)
    elif size == "medium":
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=12)
    elif size == "large":
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8, radius_km=24)
    elif size == 'huge':
        return generate_random_convex_polygon(center_x=144.9, center_y=-37.8,
                                              radius_km=48)

def get_start_date(size):
    if size == 'small':
        return "2024-02-10"
    elif size == 'medium':
        return "2023-12-10"
    elif size == 'large':
        return "2023-8-10"
    elif size == 'huge':
        return "2022-4-15"


def benchmark_run(model_func, geom, start_date, end_date, frost_threshold, duration_threshold, secret):
    result=None

    def wrapped():
        nonlocal result
        result = model_func(
            geom = geom,
            start_date = start_date,
            end_date = end_date,
            frost_threshold = frost_threshold,
            duration_threshold = duration_threshold,
            secret = secret
        )

        return result

    start = time.time()
    mem_usage = memory_usage(wrapped, max_iterations=1)
    elapsed = time.time() - start
    peak_mem = max(mem_usage)
    dims = list(result.sizes.values())
    total_dims = math.prod(dims[:3])

    return elapsed, peak_mem, total_dims


def test_local_frost_metrics_op_geom_size():

    secret = {'id': os.getenv('ERATOS_ID'),
              'secret': os.getenv('ERATOS_SECRET')}


    end_date = "2024-04-10"
    sizes = ["small", "medium", "large", "huge"]
    frost_threshold = 1
    duration_threshold = 1

    elapsed_lst = []
    peak_mem_lst = []
    total_dims_lst = []
    # for size in sizes:
    for size in sizes:
        start_date = get_start_date(size)
        geom = get_geometry(size)
        elapsed, peak_mem, total_dims = benchmark_run(daily_frost_metrics, geom, start_date,
                                          end_date, frost_threshold,
                                          duration_threshold, secret)
        elapsed_lst.append(elapsed)
        peak_mem_lst.append(peak_mem)
        total_dims_lst.append(total_dims)

        print(f"{size.capitalize()} geometry:")
        print(f"  Runtime: {elapsed:.2f} s")
        print(f"  Peak Memory: {peak_mem:.1f} MB\n")
        print(f"  Dimensions: {total_dims} \n")
    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot peak_mem_lst on ax1 (left Y-axis)
    ax1.plot(total_dims_lst, peak_mem_lst, 'o-', color='blue',
             label='Memory (MB)')
    ax1.set_xlabel('Dimension (time*lat*lon)')
    ax1.set_ylabel('Memory (MB) (left Y-axis)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    for i, j in zip(total_dims_lst, peak_mem_lst):
        ax1.text(i, j, str(j), color='blue', fontsize=9, ha='left',
                 va='bottom')

    # Create second Y-axis sharing the same X-axis
    ax2 = ax1.twinx()
    ax2.plot(total_dims_lst, elapsed_lst, 's-', color='red',
             label='Elapsed Time (s)')
    ax2.set_ylabel('Elapsed Time (s) (right Y-axis)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Annotate y2 values
    for i, j in zip(total_dims_lst, elapsed_lst):
        ax2.text(i, j, str(j), color='red', fontsize=9, ha='right',
                 va='bottom')

    # Add grid and title
    ax1.grid(True)
    plt.title('Memory Usage and Elapsed Time Over Dimensions')

    # Add legends (note: we need to manually combine them from both axes)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"peak_mem_elapse_time_v.s_dimension_w_label2.png",
                dpi=300)  # Save as high-res PNG
    plt.show()




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


def test_local_run_like_operator(caplog):
    caplog.set_level(logging.INFO)
    
    test_context = Context()

    test_context.configure_port("config", DOCUMENT_PORT, INPUT_PORT, value=json.dumps({}))

    test_context.configure_port("secrets", DOCUMENT_PORT, INPUT_PORT, value=json.dumps({
        "id": os.environ.get("ERATOS_ID"),
        "secret": os.environ.get("ERATOS_SECRET")
    }))
    # boorowa_area =   'POLYGON((148.680387 -34.477694, 148.709573 -34.477694, 148.709573 -34.46248, 148.680387 -34.46248, 148.680387 -34.477694))'
    boorowa_area = 'POLYGON ((148.582535 -34.515044, 148.868866 -34.515044, 148.868866 -34.392179, 148.582535 -34.392179, 148.582535 -34.515044))'
    test_context.configure_port("input_geom", DOCUMENT_PORT, INPUT_PORT, value=json.dumps(boorowa_area))

    test_context.configure_port("input_duration_threshold", DOCUMENT_PORT, INPUT_PORT, value=json.dumps(0))
    test_context.configure_port("input_frost_threshold", DOCUMENT_PORT, INPUT_PORT, value=json.dumps(0))
    test_context.configure_port("input_start_date", DOCUMENT_PORT, INPUT_PORT, value=json.dumps('2025-08-15'))
    test_context.configure_port("input_end_date", DOCUMENT_PORT, INPUT_PORT, value=json.dumps('2025-08-20'))

    test_context.configure_port("output_frost_metrics_min_temp", DOCUMENT_PORT, OUTPUT_PORT, value='')
    test_context.configure_port("output_frost_metrics_frost_hours", DOCUMENT_PORT, OUTPUT_PORT, value='')
    test_context.configure_port("output_frost_metrics_duration", DOCUMENT_PORT, OUTPUT_PORT, value='')

    eratos_operator_wrapper(test_context)

    min_temp_ern = test_context.ports['output_frost_metrics_min_temp'].value
    frost_hours_ern = test_context.ports['output_frost_metrics_frost_hours'].value
    duration_ern = test_context.ports['output_frost_metrics_duration'].value

    print(min_temp_ern, frost_hours_ern, duration_ern)
