from ast import Str
from pathlib import Path
from datetime import datetime
import string
import subprocess
import argparse
import boto3
import json
import logging
import os
import yaml

def publish_cw_metrics(cw, metrics_namespace, metrics, json_map, model_name, endpoint_type, container, instance_type, concurrency):
    for field_name, field_value in json_map.items():
        if field_name in metrics:
            metric_data = [{
                "MetricName":
                metrics[field_name]["metric_name"],
                    "Dimensions": [
                        {
                            "Name": "Model",
                            "Value": model_name,
                        },
                        {
                            "Name": "Endpoint",
                            "Value": endpoint_type,
                        },
                        {
                            "Name": "Container",
                            "Value": container,
                        },
                        {
                            "Name": "InstanceType",
                            "Value": instance_type,
                        },
                        {
                            "Name": "Concurrency",
                            "Value": concurrency,
                        },
                    ],
                    "Unit":
                    metrics[field_name]["unit"],
                    "Value":
                    float(field_value),
                }]
            response = cw.put_metric_data(Namespace=metrics_namespace, MetricData=metric_data)
            logging.info(
                "publish metric: %s, model: %s, endpoint: %s, container: %s, instance_type: %s, concurrency: %s, response: %s",
                metrics[field_name]["metric_name"],
                model_name,
                endpoint_type,
                container,
                instance_type,
                concurrency,
                response,
            )


def is_valid_device(device):
    if device == "gpu":
        return is_gpu_device()
    elif device == "neuron":
        return is_neuron_device()
    else:
        logging.error("Not supported device: {device}")
        return False


def is_gpu_device():
    try:
        # check the number of GPUs and GPU type.
        bash_command = "nvidia-smi --query-gpu=gpu_name --format=csv,noheader"
        logging.info(bash_command)
        output = subprocess.check_output(bash_command,
                                         shell=True,
                                         stderr=subprocess.STDOUT)
        gpu_count = len(output.decode().strip().split('\n'))
        if gpu_count <= 0:
            logging.warning("GPU not found, gpu_count: {gpu_count}")
            return False
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run nvidia-smi, error: {e}")
        return False

    return True

def run_bash_command(bash_command):
    try:
        logging.info(bash_command)
        output = subprocess.run([bash_command],
                                capture_output=True,
                                text=True,
                                check=True)
        logging.info(output.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run {bash_command}, error: {e}")
        return False
    except FileNotFoundError:
        logging.error("Not found command: {bash_command}")
        return False
    
def is_neuron_device():
    bash_command = "neuron-ls"
    return run_bash_command(bash_command)

def pull_docker_image(image):
    bash_command = f"docker pull {image}"
    return run_bash_command(bash_command)

def build_parameter_str(parameter_list):
    if len(parameter_list) == 0:
        return ""
    return " ".join(parameter_list)
    
def launch_container(container, model, hf_token):
    image = container.get("image")
    docker_parameters = build_parameter_str(container.get("docker_parameters", []))
    server_parameters = build_parameter_str(container.get("server_parameters", []))
    bash_command = f"docker run -e HUGGING_FACE_HUB_TOKEN={hf_token} {docker_parameters} {image} --model {model} {server_parameters}"
    return run_bash_command(bash_command)

def load_json_to_map(file_path):
    try:
        # Open the JSON file
        with open(file_path, 'r') as file:
            # Load the JSON content into a Python dictionary
            data_map = json.load(file)
        
        return data_map
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in the file. error: {e}")
    except Exception as e:
        logging.error("An unexpected error occurred. error: {e}")
    
    return {}

def run_llm_perf(warmup: bool, model, concurrency, llmperf_parameters_others_list, result_outputs):
    if warmup:
        bash_command = f"python token_benchmark_ray.py --model {model} --num-concurrent-requests 2 --mean-input-tokens 550 --stddev-input-tokens 150 --mean-output-tokens 150 --stddev-output-tokens 10 --max-num-completed-requests 50 --timeout 600 \
--num-concurrent-requests 1 --llm-api openai"
        return run_bash_command(bash_command)
    
    llmperf_parameters_others = build_parameter_str(llmperf_parameters_others_list)
    bash_command = f"python token_benchmark_ray.py --results-dir {result_outputs}  --model {model} --num-concurrent-requests {concurrency} {llmperf_parameters_others}"
    return run_bash_command(bash_command)

def shutdown_container():
    bash_command = "docker rm -f $(docker ps -aq)"
    return run_bash_command(bash_command)

def upload_summary_to_s3(s3, result_outputs, s3_bucket, s3_metrics_folder):
    for root, dirs, files in Path(result_outputs).walk(on_error=print):
        for file in files:
            try:
                if file.endswith("_summary.json"):
                    summary_json_file = Path(file)
                s3_metrics_object = f"{s3_metrics_folder}{result_outputs}/{file}"
                s3.upload_file(Path(file), s3_bucket, s3_metrics_object)
            except Exception as e:
                logging.error(
                    f'Failed to upload {Path(file)} to {s3_metrics_object}, error: {e}'
            )
    return summary_json_file
    

            
def run_benchmark(config_yml, instance_type):
    with open(config_yml, "r") as file:
        config = yaml.safe_load(file)

        if config is None:
            logging.fatal("Invalid config.yml")
        region = config.get("region", "us-west-2")
        metrics_namespace = config.get("cloudwatch",
                                       {}).get("metrics_namespace", "Rubikon")
        metrics = config.get("metrics", {})
        hf_token = os.getenv("HF_TOKEN", "")
        s3_bucket = config.get("s3", {}).get("bucket_name", "djl-benchmark-llm")
        s3_folder = config.get("s3", {}).get("folder", "ec2")
        current_date = datetime.now().strftime("%Y-%m-%d")
        s3_metrics_folder = f"{current_date}/{s3_folder}/metrics/"
        s3_config_folder = f"{current_date}/{s3_folder}/config/"
        s3_job_config_object = f"{s3_config_folder}config.yml"
        session = boto3.session.Session()
        cloudwatch = session.client("cloudwatch", region_name=region)
        s3 = session.client("s3", region_name=region)
        s3.upload_file(Path(config_yml), s3_bucket, s3_job_config_object)

        for benchmark in config["benchmarks"]:
            model = benchmark["model"]
            tests = benchmark["tests"]
            for test in tests:
                test_name = test.get("test_name")
                containers = test.get("containers")
                llmperf_parameters = test.get("llmperf_parameters")

                for container in containers:
                    if not container["action"]:
                        continue
                    container["HF_TOKEN"] = hf_token
                    if not is_valid_device(container["device"]):
                        continue

                    if not pull_docker_image(container["image"]):
                        continue

                    container_name = container.get("container")
                    for concurrency in llmperf_parameters[
                            "num-concurrent-requests-list"]:
                        try:
                            launch_container(container, model, hf_token)

                            result_outputs = f"{container_name}/{model}-{test_name}/{concurrency}"
                            if run_llm_perf(True, concurrency, llmperf_parameters, result_outputs):
                                run_llm_perf(False, concurrency, llmperf_parameters, result_outputs)
                            shutdown_container()

                            # parse llmperf json file
                            # upload to s3
                            # publish metrics
                            summary_json_file = upload_summary_to_s3(s3, result_outputs, s3_bucket, s3_metrics_folder)
                            json_map = load_json_to_map(summary_json_file)
                            publish_cw_metrics(cloudwatch, metrics_namespace, metrics, json_map, model, "ec2", container_name, instance_type, concurrency)
                        except Exception as e:
                            logging.error(
                                f'Error in test: {test["test_name"]} on {container["container"]} with concurrency {concurrency}, error: {e}'
                            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",
                        "-j",
                        type=str,
                        required=True,
                        help="Specify the config.yml path")

    parser.add_argument(
        "--instance",
        "-i",
        type=string,
        required=True,
        help="Specify the ec2 instance type",
    )
    args = parser.parse_args()

    run_benchmark(args.jobs, args.configs, args.metrics, args.instance)


if __name__ == "__main__":
    main()
