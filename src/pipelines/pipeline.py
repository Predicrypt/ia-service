from get_data import get_all_data_component
from process_data import process_data_component
from kfp.compiler import Compiler
import os

def pipeline(symbol, interval, limit):
    get_data_task = get_all_data_component(symbol, interval, limit)
    process_data_task = process_data_component(get_data_task.output)

Compiler().compile(
    pipeline_func=pipeline,
    package_path=f'{os.getcwd()}/ia-service/src/pipelines/definitions/GetAndProcessPipeline.yaml'
)