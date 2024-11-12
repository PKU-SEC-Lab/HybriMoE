# Copyright 2024 Shaoyuan Chen
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

import os
import platform
import sys
import time
import threading


project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import torch.nn as nn
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
from ktransformers.operators.experts import KExpertsCache
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
# use gpu or cpu
use_gpu = False

default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat-gpu.yaml" if use_gpu else ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-single-gpu.yaml" if use_gpu else ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral-gpu.yaml" if use_gpu else ktransformer_rules_dir + "Mixtral.yaml",
}


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # debug用，加上会慢不少


def local_chat(
    # model_path: str,
    optimize_rule_path: str = None,
    # gguf_path: str = None,
    max_new_tokens: int = 2000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
):
    start = time.time()
    # switch model to set path

    model_name = 0
    if model_name == 0:
        model_path = "/opt/pretrained_models/DeepSeek-V2-Lite-Chat"
        gguf_path = "/home/syf/ktransformers/Deepseek-GGUF"
    elif model_name == 1:
        model_path = "/opt/pretrained_models/Mixtral-8x7B-Instruct-v0.1"
        gguf_path = "/home/syf/ktransformers/Mixtral-GGUF"
    elif model_name == 2:
        model_path = "/opt/pretrained_models/Qwen2-57B-A14B-Instruct"
        gguf_path = "/data/pretrained_models/Qwen-GGUF"
    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer
    print("cpu_infer:", Config().cpu_infer)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if "Qwen2Moe" in config.architectures[0]:  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="flash_attention_2")

    if optimize_rule_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = input("please input the path of your rule file(yaml file containing optimize rules):")

    if gguf_path is None:
        gguf_path = input("please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):")

    # setting
    if model_name == 1:
        load_size = [4] * config.num_hidden_layers
    else:
        load_size = [16] * config.num_hidden_layers
    prefetch_size = 0
    start = time.time()
    config.load_size = load_size
    config.prefetch_size = prefetch_size
    optimize_and_load_gguf(
        model,
        optimize_rule_path,
        gguf_path,
        config,
    )
    end = time.time()
    print("optimize_and_load_gguf time:", end - start)

    model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    logging.basicConfig(level=logging.INFO)

    if use_gpu:
        print("using gpu")
    else:
        print("using cpu")

    # content = "write quick sort algorithm in python"
    cache = KExpertsCache._instance
    loop = 1

    while True:
        if loop % 3 == 0:
            content = input("继续？")
            if content == "n":
                break
        # content = "请完整的给出每种排序的c语言代码"
        content = "请给出2的1到10次方的值"
        print("content:", content)
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        torch.set_default_dtype(torch.bfloat16)  # TODO: Remove this, replace dtype using config

        generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph)
        if cache is not None:
            cache.print_time()
        break
    if cache is not None:
        cache.print_time()

    print("fin")


import signal


def timeout_handler(signum, frame):
    print("程序超时，终止进程")
    os.kill(os.getpid(), signal.SIGTERM)


def set_timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)


def clear_timeout():
    signal.alarm(0)


if __name__ == "__main__":
    # fire.Fire(local_chat)
    set_timeout(6000)
    # try:
    local_chat()
    # except Exception as e:
    #     print(e)
    #     clear_timeout()
    #     # trace
    #     import traceback

    #     traceback.print_exc()
    # finally:
    #     clear_timeout()
    #     os._exit(0)
