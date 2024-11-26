import json
import os
from typing import Dict

import httpx

from llamafactory.extras.logging import get_logger
from llamafactory.webui.utils import load_args

logger = get_logger(__name__)
user_config = load_args("cache/user_config.yaml")
finetuning_type_map = {"full": 1, "lora": 2, "freeze": 3}


class TaskApi:
    base_url = user_config.get("base_host_url")
    api_client = httpx.Client(base_url=base_url)

    @classmethod
    def get_url_params(cls) -> Dict:
        url_params = json.loads(os.environ.get("URL_PARAMS"))
        logger.info(f"url_params: {url_params}")
        return url_params

    @classmethod
    def all_tasks(cls):
        api = "/llm/task/list"
        url_params = cls.get_url_params()
        resp = cls.api_client.get(api, params=url_params)
        logger.debug(f"{resp.url}: {resp.text}")
        return resp.json()

    @classmethod
    def add_train_task(cls, training_args: Dict) -> int:
        """插入训练任务, 返回训练任务ID"""
        api = "/llm/task/add"
        payload = {
            "name": training_args.get("train.output_dir"),
            "baseModel": training_args.get("top.model_name"),
            "baseModelPath": training_args.get("top.model_path"),
            "finetuningType": finetuning_type_map.get(
                training_args.get("top.finetuning_type")
            ),
            "trainingArgs": json.dumps(training_args, ensure_ascii=False, indent=None),
            "trainingStatus": training_args.get("train.status"),
            "createTime": training_args.get("train.create_time"),
            "finishTime": training_args.get("train.finish_time"),
            "trainedModelName": training_args.get("export_model_name"),
            "trainedModelPath": training_args.get("export_dir"),
        }
        payload.update(cls.get_url_params())
        resp = cls.api_client.post(api, data=payload)
        logger.info(f"add_train_task: {resp.text}")
        return resp.json().get("data", {}).get("id")

    @classmethod
    def update_train_task(cls, update_payload: Dict) -> None:
        api = "/llm/task/update"
        update_payload.update(cls.get_url_params())
        resp = cls.api_client.post(api, data=update_payload)
        logger.info(f"update_train_task_info: {resp.text}")

    @classmethod
    def fetch_train_task_detail(cls, cond: Dict) -> Dict:
        api = "/llm/task/detail"
        cond.update(cls.get_url_params())
        resp = cls.api_client.get(api, params=cond)
        logger.info(f"fetch_train_task_info: {resp.text}")
        return resp.json()["data"]

    @classmethod
    def add_model(cls, model_detail: Dict) -> int:
        api = "/llm/result/add"
        payload = {
            "name": model_detail.get("name"),
            "path": model_detail.get("path"),
            "quantizationType": model_detail.get("quantization_type"),
            "deployStatus": model_detail.get("deploy_status", 1),
            "createTime": model_detail.get("create_time"),
            "baseModel": model_detail.get("base_model"),
            "trainTaskId": model_detail.get("train_task_id"),
        }
        payload.update(cls.get_url_params())
        resp = cls.api_client.post(api, data=payload)
        logger.info(f"add_model: {resp.text}")
        return resp.json()["data"].get("id")

    @classmethod
    def fetch_model_detail(cls, cond: Dict) -> Dict:
        api = "/llm/result/detail"
        cond.update(cls.get_url_params())
        resp = cls.api_client.get(api, params=cond)
        logger.info(f"fetch_model_detail: {resp.text}")
        return resp.json()["data"]

    @classmethod
    def update_model(cls, update_payload: Dict):
        api = "/llm/result/update"
        update_payload.update(cls.get_url_params())
        resp = cls.api_client.post(api, data=update_payload)
        logger.info(f"update_model: {resp.text}")
