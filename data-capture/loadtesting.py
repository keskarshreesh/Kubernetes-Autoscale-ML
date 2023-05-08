import numpy as np
import pandas as pd
import kube_utils
import requests
from dotenv import load_dotenv, set_key
import subprocess
import datetime as dt
import logging
import os
import time
import json
from kubernetes.client.rest import ApiException
from prometheus_client import Parser, exposition
from kubernetes import client, config, utils
import argparse

# environment
load_dotenv(override=True)

# init logger
p = logging.getLogger(__name__)
p.setLevel(logging.INFO)

def config_env(**kwargs) -> None:

    arguments = locals()
    env_file = os.path.join(os.getcwd(), ".env")
    for i in arguments["kwargs"].keys():
        key = str(i).upper()
        value = str(arguments["kwargs"][i])
        set_key(dotenv_path=env_file, key_to_set=key, value_to_set=value)

def get_istio_metrics(metric_name: str, mode: str, custom: bool, hh: int, mm: int) -> list:
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    ret = None
    try:
        ret = v1.list_namespaced_pod("istio-system")
    except ApiException as e:
        print("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    pods = []
    for i in ret.items:
        pods.append(i.metadata.name)

    api = client.CustomObjectsApi()
    prom_metrics = []
    for pod in pods:
        try:
            response = api.list_namespaced_custom_object(
                group="metrics.istio.io",
                version="v1alpha1",
                namespace="istio-system",
                plural="metrics",
                label_selector=f"app={metric_name},pod={pod}"
            )
        except ApiException as e:
            print("Exception when calling CustomObjectsApi->list_namespaced_custom_object: %s\n" % e)
        for metric in response['items']:
            metric_value = metric['metricValue']
            if mode == "RESOURCES":
                if custom:
                    prom_metrics.append((metric_value['value'], metric_value['type'], metric_value['dimension']['resource'], pod))
                else:
                    prom_metrics.append((metric_value['value'], metric_value['type'], metric_value['name'], pod))
            elif mode == "NETWORK":
                if custom:
                    prom_metrics.append((metric_value['value'], metric_value['type'], metric_value['dimension']['source'], pod))
                    prom_metrics.append((metric_value['value'], metric_value['type'], metric_value['dimension']['destination'], pod))
                else:
                    prom_metrics.append((metric_value['value'], metric_value['type'], metric_value['name'], pod))

    return prom_metrics


def export_istio_metrics_to_csv(folder: str, iteration: int, hh: int, mm: int) -> None:
    # metrics to export
    resource_metrics = [
        "istio_requests_total",
        "istio_request_duration_milliseconds",
        "istio_tcp_received_total",
        "istio_tcp_sent_total",
    ]

    # get resource metric data resources
    resource_metrics_data = get_istio_metrics(metric_name=resource_metrics[0], mode="NETWORK", custom=False,
                                              hh=hh, mm=mm)
    for x in range(1, len(resource_metrics)):
        resource_metrics_data += get_istio_metrics(metric_name=resource_metrics[x], mode="NETWORK", custom=False,
                                                   hh=hh, mm=mm)

    # get custom resource metric data resources
    # memory usage
    custom_memory = get_istio_metrics(metric_name="memory", mode="RESOURCES", custom=True, hh=hh, mm=mm)
    custom_memory = pd.DataFrame(custom_memory, columns=["value", "type", "resource", "pod"])
    custom_memory.insert(0, 'metric', "memory")

    # cpu usage
    custom_cpu = get_istio_metrics(metric_name="cpu", mode="RESOURCES", custom=True, hh=hh, mm=mm)
    custom_cpu = pd.DataFrame(custom_cpu, columns=["value", "type", "resource", "pod"])
    custom_cpu.insert(0, 'metric', "cpu")

    # response time
    custom_resp_time = get_istio_metrics(metric_name="request_duration", mode="RESOURCES", custom=True, hh=hh, mm=mm)
    custom_resp_time = pd.DataFrame(custom_cpu, columns=["value", "type", "resource", "pod"])
    custom_resp_time.insert(0, 'metric', "response_time")

    # write
    custom_metrics_df = pd.concat(
        [custom_cpu, custom_memory, custom_resp_time])
    # write to csv file
    custom_metrics_df.to_csv(rf"{folder}\custom_metrics_{iteration}.csv")

def evaluation(load: int, spawn_rate: int, hh: int, mm: int, load_testing: str) -> None:

    # init date
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # create folder
    folder_path = os.path.join(os.getcwd(), "data", "raw", f"{date}_eval")
    os.mkdir(folder_path)
    # create deployments
    kube_utils.create_kube_ml()
    kube_utils.deploy_autoscaler_docker()
    # config
    kube_utils.set_prometheus_info()
    config_env(
        host=os.getenv("base_url"),
        node_port=kube_utils.k8s_get_app_port(),
        date=date,
        load=load,
        spawn_rate=spawn_rate,
        HH=hh,
        MM=mm
    )
    # evaluation
    logging.info("Starting Evaluation.")
    logging.info("Start Locust.")
    if load_testing == "JMeter":
        start_jmeter(0, date, False, load)
    # get prometheus data
    time.sleep(30)
    get_istio_metrics(folder=folder_path, iteration=0, hh=hh, mm=mm)
    # clean up
    kube_utils.delete_autoscaler_docker()
    kube_utils.k8s_delete_namespace()
    logging.info("Finished Benchmark.")

def parameter_variation_mat(expressions, step, sample, load):
    
    data = None

    # Initialize dictionary to store parameter variation matrices
    matrices = {}

    # Generate parameter variation matrix for each deployment
    for i, l in enumerate(load):
        # Initialize matrix with zeros
        mat = np.zeros((int(l/step), expressions))

        # Calculate parameter variations for each expression
        for j in range(expressions):
            if sample:
                # Sample data from the loaded file
                data_subset = data[j*int(len(data)/expressions):(j+1)*int(len(data)/expressions)]
                for k in range(mat.shape[0]):
                    start_index = k * step
                    end_index = min(start_index + step, len(data_subset))
                    mat[k,j] = max(data_subset[start_index:end_index]) - min(data_subset[start_index:end_index])
            else:
                # Randomly generated data
                mat[:,j] = np.random.randint(1, 100, mat.shape[0])

        # Store matrix in dictionary
        matrices[f'deployment_{i}'] = mat

    return matrices

def benchmark(name: str, load: list, spawn_rate: int, expressions: int,
              step: int, run: int, run_max: int, custom_shape: bool, history: bool,
              sample: bool, locust: bool):
    # init date
    # read new environment data
    load_dotenv(override=True)
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # create folder
    folder_path = os.path.join(os.getcwd(), "data", "raw", date)
    os.mkdir(folder_path)
    kube_utils.create_kube_ml()
    # config
    set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="LAST_DATA", value_to_set=date)
    kube_utils.set_prometheus_info()
    config_env(app_name=name,
               host=os.getenv("HOST"),
               node_port=kube_utils.k8s_get_app_port(),
               date=date,
               load=load,
               spawn_rate=spawn_rate
               )
    iteration = 1
    scale_only = "webui"
    # get variation
    variations = parameter_variation_mat(expressions, step, sample, load)
    c_max, m_max, p_max, l_max = variations[os.getenv("UI")].shape

    # benchmark
    logging.info("Starting Benchmark.")
    for c in range(0, c_max):
        for m in range(0, m_max):
            for p in range(0, p_max):
                for l in range(0, l_max):
                    logging.info(
                        f"Iteration: {iteration}/{c_max * m_max * p_max} run: {run}/ {run_max}")
                    # for every pod in deployment
                    for pod in variations.keys():
                        # check that pod is scalable
                        if scale_only in pod:
                            # get parameter variation
                            v = variations[pod][c, m, p, l]
                            # check if variation is empty
                            if v[0] == 0 or v[1] == 0 or v[2] == 0:
                                break
                            logging.info(f"{pod}: cpu: {int(v[0])}m - memory: {int(v[1])}Mi - # pods: {int(v[2])}")
                            # update resources of pod
                            kube_utils.deploy(deployment_name=pod, cpu_limit=int(v[0]),
                                                      memory_limit=int(v[1]),
                                                      number_of_replicas=int(v[2]), replace=True)
                            # wait for deployment
                            time.sleep(90)
                            while not kube_utils.check_teastore_health():
                                time.sleep(10)
                    # start load test
                    logging.info("Start Load.")
                    start_jmeter(iteration, date, True, l)
                    # get prometheus data
                    export_istio_metrics_to_csv(folder=folder_path, iteration=iteration, hh=int(os.getenv("HH")),
                                        mm=int(os.getenv("MM")))
                    iteration = iteration + 1
    kube_utils.k8s_delete_namespace()


def start(name, load, spawn_rate, expressions, step, runs,
          custom_shape, history, sample, locust):
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="FIRST_DATA", value_to_set=date)
    for i in range(1, runs + 1):
        benchmark(name, load, spawn_rate, expressions, step, i, runs, custom_shape, history, sample,
                  locust)


def start_jmeter(iteration, date, evaluation, rps):
    jmeter_path = os.path.join(os.getcwd(), "data", "loadtest", "jmeter", "bin")
    jmeter_jar = os.path.join(jmeter_path, "ApacheJMeter.jar")
    os.chdir(jmeter_path)

    cmd = ["java", "-jar", jmeter_jar, "-Jhostname", os.getenv("HOST"), "-Jport", os.getenv('NODE_PORT'),
           "-l", f"{date}_{iteration}.log"]

    if evaluation:
        cmd.extend(["-t", "ml_kube_sample.jmx", "-Jjmeterengine.force.system.exit=true", "-n"])
    else:
        cmd.extend(["-t", "ml_kube_sample.jmx", "-Jload_profile", f'const({rps},{int(os.getenv("MM")) * 60}s)', "-n"])

    logging.info("Executing command: %s", " ".join(cmd))
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in process.stdout:
            print(line.decode("utf-8"), end="")
        for line in process.stderr:
            print(line.decode("utf-8"), end="")
    os.chdir(os.getcwd())


def main():
    parser = argparse.ArgumentParser(description="Start load testing")
    parser.add_argument("name", type=str, help="Name of the test")
    parser.add_argument("load", type=int, default=350)
    parser.add_argument("spawn_rate", type=int)
    parser.add_argument("expressions", type=list, default=[])
    parser.add_argument("-s", "--step", type=int, default=6, help="Step duration in seconds (default: 10)")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of test runs (default: 1)")
    parser.add_argument("-c", "--custom-shape", type=str, default=None, help="Custom shape for the load curve (comma-separated list of values)")
    parser.add_argument("-H", "--history", type=str, default=None, help="Path to save the load history")
    parser.add_argument("-S", "--sample", type=str, default=None, help="Path to save the data")
    parser.add_argument("--locust", action="store_true", help="Use Locust instead of JMeter")
    args = parser.parse_args()

    # Call the start function with the parsed arguments
    start(name=args.name, load=args.load, spawn_rate=args.spawn_rate, expressions=args.expressions,
          step=args.step, runs=args.runs, custom_shape=args.custom_shape, history=args.history,
          sample=args.sample, locust=args.locust)

if __name__ == "__main__":
    main()