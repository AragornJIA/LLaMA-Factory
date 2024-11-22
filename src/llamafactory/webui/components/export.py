# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os.path
from subprocess import Popen
from typing import TYPE_CHECKING, Dict, Generator, List, Union

from transformers.trainer_utils import get_last_checkpoint

from ...api.task_api import TaskApi, finetuning_type_map
from ..common import GPTQ_BITS, get_save_dir, load_config
from ..locales import ALERTS
from ..utils import get_cur_datetime, load_args, save_args
from ...extras.constants import PEFT_METHODS
from ...extras.logging import get_logger
from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model

logger = get_logger(__name__)

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def can_quantize(checkpoint_path: Union[str, List[str]]) -> "gr.Dropdown":
    if isinstance(checkpoint_path, list) and len(checkpoint_path) != 0:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def quantize_model(model_name: str, quantization_type: int = 2):
    p1 = Popen(f"bash bash_scripts/quantize_model.bash {model_name} {quantization_type}", shell=True)
    p1.communicate()


def deploy_model(model_name: str, model_path: str):
    prefix = f"FROM {model_name}-Q4_0.gguf\n"
    ollama_makefile_path = model_path + "/ollama_Makefile"
    with (open('assets/ollama_makefile_template', 'r', encoding='utf-8') as fr,
          open(ollama_makefile_path, 'w', encoding='utf-8') as fw):
        fw.writelines([prefix, fr.read()])

    p2 = Popen(f"ollama create {model_name} -f {ollama_makefile_path}", shell=True)
    p2.communicate()


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: Union[str, List[str]],
    template: str,
    visual_inputs: bool,
    export_size: int,
    export_quantization_bit: str,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
    export_hub_model_id: str,
) -> Generator[str, None, None]:
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif export_quantization_bit in GPTQ_BITS and checkpoint_path and isinstance(checkpoint_path, list):
        error = ALERTS["err_gptq_lora"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    user_config = load_config()

    args = dict(
        model_name_or_path=model_path,
        finetuning_type=finetuning_type,
        template=template,
        visual_inputs=visual_inputs,
        export_dir=user_config.get("trained_models_dir") + export_dir,
        export_hub_model_id=export_hub_model_id or None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
    )

    if checkpoint_path:
        if finetuning_type in PEFT_METHODS:  # list
            args["adapter_name_or_path"] = ",".join(
                [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
            )
        else:  # str
            args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

    last_checkpoint_dir = get_last_checkpoint(args.get("adapter_name_or_path"))
    train_args_path = os.path.join(last_checkpoint_dir, "llamaboard_config.yaml")
    train_args = load_args(train_args_path)
    train_args.update(args)
    train_args['export_model_name'] = export_dir
    save_args(train_args_path, train_args)

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]

    train_args['train.status'] = 5
    train_args['export.trained_model_create_time'] = get_cur_datetime()
    train_args['deploy_status'] = 1
    logger.info(f"train_args:{train_args}")

    trained_model_detail: Dict = {
        "name": train_args.get("export_model_name"),
        "path": train_args.get("export_dir"),
        "train_task_id": train_args['task_id'],
        "deploy_status": train_args.get("deploy_status", 1),
        "create_time": train_args.get("export.trained_model_create_time"),
        "base_model": train_args.get("top.model_name"),
    }

    # 1. 根据 checkpoint 文件夹中的 llamaboard_config.yaml 中的 task_id, 如果该 task_id 对应 trainedModelName 字段为空, 说明该模型还没有被导出过, 就根据 task_id 进行更新;
    task_info = TaskApi.fetch_train_task_detail({"id": train_args.get("task_id")})
    if not task_info.get("trainedModelName"):
        logger.info("第1分支")
        TaskApi.update_train_task(
            {"id": train_args.get("task_id"),
             "trainingStatus": train_args['train.status'],
             "trainedModelName": train_args.get("export_model_name"),
             "trainedModelPath": train_args.get("export_dir"),
             }
        )
        trained_model_detail['model_id'] = TaskApi.add_model(trained_model_detail)

    # 2. 如果该模型已经被导出过, 就检查 trainedModelName 是否已经存在, 如果存在, 就更新该记录;
    elif task_id := TaskApi.fetch_train_task_detail({"trainedModelName": train_args.get("export_model_name")}).get(
        "id"):
        logger.info("第2分支")
        train_args['task_id'] = task_id
        TaskApi.update_train_task(
            {"id": task_id,
             "name": train_args.get("train.output_dir"),
             "baseModel": train_args.get("top.model_name"),
             "baseModelPath": train_args.get("top.model_path"),
             "finetuningType": finetuning_type_map.get(train_args.get("top.finetuning_type")),
             "trainingArgs": json.dumps(train_args, ensure_ascii=False, indent=None),
             "trainingStatus": train_args.get("train.status"),
             "createTime": train_args.get('train.create_time'),
             "finishTime": train_args.get('train.finish_time'),
             "trainedModelName": train_args.get("export_model_name"),
             "trainedModelPath": train_args.get("export_dir"),
             })
        trained_model_detail['base_model'] = train_args.get("top.model_name")
        trained_model_detail['name'] = train_args.get("export_model_name")
        trained_model_detail['path'] = train_args.get("export_dir")
        trained_model_detail['create_time'] = train_args.get("export.trained_model_create_time")
        model_id = TaskApi.fetch_model_detail({"trainTaskId": train_args.get("task_id")}).get("id")
        trained_model_detail['model_id'] = model_id
        TaskApi.update_model({
            "id": model_id,
            "baseModel": trained_model_detail['base_model'],
            "path": trained_model_detail['path'],
            "createTime": trained_model_detail['create_time'],
        })

    # 3. 如果该模型已经被导出过, 并且 trainedModelName 还不存在, 就新增一条导出记录;
    else:
        logger.info("第3分支")
        task_id = TaskApi.add_train_task(train_args)
        train_args['task_id'] = trained_model_detail['train_task_id'] = task_id
        model_id = TaskApi.add_model(trained_model_detail)
        trained_model_detail['model_id'] = model_id

    save_args(os.path.join(train_args.get("export_dir"), "llamaboard_config.yaml"), train_args)

    yield ALERTS["info_deploying"][lang]

    logger.info(f"trained_model_detail: {trained_model_detail}")

    quantize_model(trained_model_detail.get("name"))
    train_args['export.quantization_type'] = trained_model_detail['quantization_type'] = 2
    trained_model_detail['deploy_status'] = 2
    TaskApi.update_model({
        "id": trained_model_detail.get("model_id"),
        "quantizationType": trained_model_detail['quantization_type'],
        "deployStatus": trained_model_detail['deploy_status'],
        "trainTaskId": trained_model_detail['train_task_id'],
    })

    deploy_model(train_args.get("export_model_name"), train_args.get("export_dir"))
    trained_model_detail['deploy_status'] = 3
    TaskApi.update_model(
        {"id": trained_model_detail.get("model_id"), "deployStatus": trained_model_detail['deploy_status']})

    yield ALERTS["info_deployed"][lang]


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=100, value=5, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none"] + GPTQ_BITS, value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.json")
        export_device = gr.Radio(choices=["cpu", "auto"], value="cpu")
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()
        export_hub_model_id = gr.Textbox()

    checkpoint_path: gr.Dropdown = engine.manager.get_elem_by_id("top.checkpoint_path")
    checkpoint_path.change(can_quantize, [checkpoint_path], [export_quantization_bit], queue=False)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            engine.manager.get_elem_by_id("top.visual_inputs"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
            export_hub_model_id,
        ],
        [info_box],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id,
        export_btn=export_btn,
        info_box=info_box,
    )
