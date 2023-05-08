import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from datetime import datetime, timedelta
from kubernetes import client, config, utils
import argparse
from typing import Optional

# init
load_dotenv()
logging.getLogger().setLevel(logging.INFO)



def get_all_istio_data() -> list:
    """Gets all Istio metric data between two dates.

    Returns:
        list: List of Istio metric data
    """
    # Load Kubernetes configuration
    config.load_kube_config()

    # Create Istio client object
    api_instance = client.CustomObjectsApi()

    # Get all namespaces
    namespaces = api_instance.list_namespace().items

    # Metrics to export
    resource_metrics = [
        "istio_request_count",
        "istio_request_duration_milliseconds",
        "istio_request_size_bytes",
        "istio_response_size_bytes"
    ]

    # Calculate start and end time
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=30)

    # Get resource metric data for each namespace
    all_data = []
    for ns in namespaces:
        ns_data = []
        namespace = ns.metadata.name

        for metric in resource_metrics:
            try:
                metric_data = api_instance.get_namespaced_custom_object_metric(
                    group="networking.istio.io",
                    version="v1alpha3",
                    namespace=namespace,
                    plural="metrics",
                    name=metric,
                    start_time=start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    end_time=end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                )

                # Append metric data to list
                ns_data.append(metric_data)

            except Exception as e:
                print(f"Error fetching {metric} data for namespace {namespace}: {e}")
                ns_data.append(None)

        # Append namespace data to list
        all_data.append(ns_data)

    return all_data

def get_data(directory):
    # config
    load_dotenv(override=True)
    prometheus_data = None
    prometheus_custom_data = None
    locust_data = None
    # check if folder exists
    data_path = os.path.join(os.getcwd(), "data", "raw", directory)
    if os.path.exists(data_path):
        # search for prometheus metric files
        logging.info(f"Gets data from {directory}.")
        for (dir_path, dir_names, filenames) in os.walk(data_path):
            for file in filenames:
                if "metrics" in file and "custom_metrics" not in file:
                    i = int(str(file).split("_")[1].rstrip(".csv"))
                    prometheus_data = get_data_helper(prometheus_data, file, i, directory)
                elif "custom_metrics" in file:
                    j = int(str(file).split("_")[2].rstrip(".csv"))
                    print(j)
                    prometheus_custom_data = get_data_helper(prometheus_custom_data, file, j, directory)
                elif "locust" in file and "stats" in file and "history" not in file:
                    l = int(str(file).split("_")[2].rstrip(".csv"))
                    locust_data = get_data_helper(locust_data, file, l, directory)
    return prometheus_data, prometheus_custom_data, locust_data


def get_data_helper(data: pd.DataFrame, file: str, iteration: int, directory: str) -> pd.DataFrame:
    """Connects two dataframes.

    Args:
      data: given dataframe
      file: data frame in file
      iteration: number of iteration
      directory: date
      data: pd.DataFrame: 
      file: str: 
      iteration: int: 
      directory: str: 

    Returns:
      connected data frame

    """
    data_path = os.path.join(os.getcwd(), "data", "raw", directory)
    # concat metrics
    tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, file), delimiter=',')
    tmp_data.insert(0, 'Iteration', iteration)
    if data is None:
        data = tmp_data
    else:
        data = pd.concat([data, tmp_data])
    return data


def get_directories() -> list:
    """Gets all directory names between the first and last data date.
    :return: list of directory names

    Args:

    Returns:

    """
    load_dotenv()
    first_date = int(str(os.getenv("FIRST_DATA")).replace('-', "").strip())
    last_date = int(str(os.getenv("LAST_DATA")).replace('-', "").strip())
    base_path = os.path.join(os.getcwd(), "data", "raw")
    dirs = list()
    # get data from each run
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_dir in dir_names:
            if "dataset" not in c_dir:
                c_date = int(str(c_dir).replace('-', "").strip())
                if last_date >= c_date >= first_date:
                    dirs.append(c_dir)
    return dirs


def get_filtered_data(directory: str) -> pd.DataFrame:
    """Returns a data frame of a given filtered data.

    Args:
      directory: date
      directory: str: 

    Returns:
      data frame of filtered data

    """
    base_path = os.path.join(os.getcwd(), "data", "filtered")
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_file in filenames:
            if directory in c_file:
                df = pd.read_csv(os.path.join(base_path, c_file))
                return df


def get_all_filtered_data() -> list:
    """Reads all filtered data between two dates.
    :return: list of filtered data

    Args:

    Returns:

    """
    load_dotenv()
    first_date = int(str(os.getenv("FIRST_DATA")).replace('-', "").strip())
    last_date = int(str(os.getenv("LAST_DATA")).replace('-', "").strip())
    base_path = os.path.join(os.getcwd(), "data", "filtered")
    files = list()
    # get data from each run
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_file in filenames:
            if str(c_file).endswith(".csv"):
                c_date = int(str(c_file).replace('-', "").replace(".csv", "").strip())
                if last_date >= c_date >= first_date:
                    files.append(pd.read_csv(os.path.join(base_path, c_file)))
    return files


def filter_all_data() -> None:
    """Filters all data between two dates.
    :return: None

    Args:

    Returns:

    """
    # init
    i = 1
    dirs = get_directories()
    for d in dirs:
        # filter data in directory
        logging.info(f"Filtering data: {d} {i}/{len(dirs)}")
        filter_data(d)
        i = i + 1


def get_variation_matrix(directory: str) -> np.array:
    """Reads all variation matrices of a directory and puts them in a list.

    Args:
      directory: current directory
      directory: str: 

    Returns:
      variation matrix

    """
    dir_path = os.path.join(os.getcwd(), "data", "raw", directory)
    # find variation files
    for (dir_path, dir_names, filenames) in os.walk(dir_path):
        for file in filenames:
            if "variation" in file:
                # filter name
                name = str(file).split("-")[1].split("_")[0]
                # read variation file
                file_path = os.path.join(dir_path, file)
                res = pd.read_csv(filepath_or_buffer=file_path, delimiter=',')
                # edit table
                res.insert(0, 'pod', name)
                res.rename(columns={"Unnamed: 0": "Iteration"}, inplace=True)
                res.reset_index()
                return res


def save_data(data, directory, mode):
    save_path = os.path.join(os.getcwd(), "data", mode, f"{directory}.csv")
    data.to_csv(path_or_buf=save_path)

def filter_data(directory):
    normal, custom, locust = get_data(directory)
    filtered_data = filter_namespace(normal)
    filtered_custom_data = filter_and_clean_custom(custom)
    variation = get_variation_matrix(directory)
    res_data = merge_tables(filtered_data, filtered_custom_data, variation)
    res_data = clean_and_calculate_ratios(res_data)
    res_data = filter_webui_pod(res_data)
    save_data(res_data, directory, "filtered")
    return res_data

def filter_webui_pod(data):
    return data.loc[data['pod'] == 'webui']

def filter_namespace(data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = pd.concat(objs=[data[data.namespace.eq(os.getenv("NAMESPACE"))]])
    filtered_data["pod"] = filtered_data["pod"].str.split("-", n=2).str[1]
    filtered_data['datapoint'] = filtered_data.groupby(["Iteration"]).cumcount() + 1
    filtered_data = pd.pivot_table(filtered_data, index=["Iteration", "pod", "datapoint"], columns=["__name__"],
                                   values="value").reset_index()
    filtered_data = filtered_data.groupby(["Iteration", "pod"]).mean().reset_index()
    return filtered_data

def filter_and_clean_custom(data: pd.DataFrame) -> pd.DataFrame:
    data['datapoint'] = data.groupby(["Iteration"]).cumcount() + 1
    data['pod'] = data['pod'].fillna("webui")
    filtered_custom_data = pd.pivot_table(data, index=["Iteration", "pod", "datapoint"], columns=["metric"],
                                          values="value").reset_index()
    filtered_custom_data = filtered_custom_data.groupby(["Iteration", "pod"]).mean().reset_index()
    filtered_custom_data.rename(columns={"rps": "average rps"}, inplace=True)
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] < filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data['median_latency'])
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] > filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data['median_latency'])
    return filtered_custom_data

def merge_tables(data1: pd.DataFrame, data2: pd.DataFrame, data3: pd.DataFrame) -> pd.DataFrame:
    res_data = pd.merge(data1, data2, how='left', on=["Iteration", "pod"])
    res_data = pd.merge(res_data, data3, how='left', on=["Iteration", "pod"])
    res_data.drop(columns=["kube_deployment_spec_replicas", "kube_pod_container_resource_limits_cpu_cores",
                           "kube_pod_container_resource_limits_memory_bytes",
                           "kube_pod_container_resource_requests_cpu_cores",
                           "kube_pod_container_resource_requests_memory_bytes",
                           "datapoint_x", "datapoint_y"], inplace=True)
    res_data.rename(
        columns={"cpu": "cpu usage", "memory": "memory usage", "CPU": "cpu limit", "Memory": "memory limit",
                 "Pods": "number of pods", "container_cpu_cfs_throttled_seconds_total": "cpu throttled total",
                 "response_time": "average response time", "median_latency": "median latency"},
        inplace=True)
    return res_data

def clean_and_calculate_ratios(data, custom_data, variation):
    # Filter for namespace
    filtered_data = data[data.namespace.eq(os.getenv("namespace"))]
    
    # Filter for pod name
    filtered_data["pod"] = filtered_data["pod"].str.split("-", n=2).str[1]
    custom_data["pod"] = custom_data["pod"].str.split("-", n=2).str[1]
    
    # Count data points per iteration
    filtered_data['datapoint'] = filtered_data.groupby(["Iteration"]).cumcount() + 1
    custom_data['datapoint'] = custom_data.groupby(["Iteration"]).cumcount() + 1
    
    # Create pivot tables
    filtered_data = pd.pivot_table(filtered_data, index=["Iteration", "pod", "datapoint"], columns=["__name__"],
                                   values="value").reset_index()
    filtered_custom_data = pd.pivot_table(custom_data, index=["Iteration", "pod", "datapoint"], columns=["metric"],
                                          values="value").reset_index()
    
    # Calculate median values
    filtered_data = filtered_data.groupby(["Iteration", "pod"]).mean().reset_index()
    filtered_custom_data = filtered_custom_data.groupby(["Iteration", "pod"]).mean().reset_index()
    filtered_custom_data.rename(columns={"rps": "average rps"}, inplace=True)
    
    # Outliers
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] < filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data['median_latency'])
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] > filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data['median_latency'])
    
    # Merge all tables
    res_data = pd.merge(filtered_data, filtered_custom_data, how='left', on=["Iteration", "pod"])
    res_data = pd.merge(res_data, variation, how='left', on=["Iteration", "pod"])
    
    # Drop unnecessary columns
    res_data.drop(columns=["kube_deployment_spec_replicas", "kube_pod_container_resource_limits_cpu_cores",
                           "kube_pod_container_resource_limits_memory_bytes",
                           "kube_pod_container_resource_requests_cpu_cores",
                           "kube_pod_container_resource_requests_memory_bytes",
                           "datapoint_x", "datapoint_y"], inplace=True)
    
    # Rename columns
    res_data.rename(
        columns={"cpu": "cpu usage", "memory": "memory usage", "CPU": "cpu limit", "Memory": "memory limit",
                 "Pods": "number of pods", "container_cpu_cfs_throttled_seconds_total": "cpu throttled total",
                 "response_time": "average response time", "median_latency": "median latency"},
        inplace=True)
    
    # Calculate ratios
    res_data["rps delta"] = (res_data["average rps"] - res_data["RPS"]) / res_data["RPS"]
    res_data["ratio response time"] = res_data["median latency"] * (1 - res_data["rps delta"])
    res_data["ratio cpu usage"] = res_data["cpu usage"] * (1 - res_data["rps delta"])
    res_data["ratio memory usage"] = res_data["memory usage"] * (1 - res_data["rps delta"])
    
    return res_data

def main(directory: Optional[str] = "../data") -> None:
    filter_data(directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data and filter for webui pod.')
    parser.add_argument('--directory', dest='directory', default="../data",
                        help='The directory containing the data files and the save destination.')
    args = parser.parse_args()
    main(args.directory)