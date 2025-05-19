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
import sys
import time
import argparse

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
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
use_gpu = True

default_optimize_rules = {
    "DeepseekV2ForCausalLM": (
        ktransformer_rules_dir + "DeepSeek-V2-Chat-gpu.yaml"
        if use_gpu
        else ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml"
    ),
    "Qwen2MoeForCausalLM": (
        ktransformer_rules_dir + "Qwen2-single-gpu.yaml"
        if use_gpu
        else ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml"
    ),
    "MixtralForCausalLM": (
        ktransformer_rules_dir + "Mixtral-gpu.yaml" if use_gpu else ktransformer_rules_dir + "Mixtral.yaml"
    ),
}


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # debug用，加上会慢不少


def local_chat(
    # model_path: str,
    optimize_rule_path: str = None,
    # gguf_path: str = None,
    max_new_tokens: int = 1024,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    model_name: int = 0,
    optimize_rule_name: str = None,
    load_size: int = None,
):
    start = time.time()
    # switch model to set path
    if model_name == 0:
        print("using DeepseekV2ForCausalLM")
        model_path = "/opt/pretrained_models/DeepSeek-V2-Lite-Chat"
        gguf_path = "/data/home/yanfansun/ktrans/ktransformers/Deepseek-GGUF"
    elif model_name == 1:
        print("using MixtralForCausalLM")
        model_path = "/opt/pretrained_models/Mixtral-8x7B-Instruct-v0.1"
        gguf_path = "/data/home/yanfansun/ktrans/ktransformers/Mixtral-GGUF"
    elif model_name == 2:
        print("using Qwen2MoeForCausalLM")
        model_path = "/opt/pretrained_models/Qwen2-57B-A14B-Instruct"
        gguf_path = "/data/home/yanfansun/ktrans/ktransformers/Qwen-GGUF"
    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer
    #print("cpu_infer:", Config().cpu_infer)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            #print("using custom modeling_xxx.py.")
            if "Qwen2Moe" in config.architectures[0]:  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_rule_path is None:
        if config.architectures[0] in default_optimize_rules:
            #print("using default_optimize_rule for", config.architectures[0])
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = input("please input the path of your rule file(yaml file containing optimize rules):")

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to"
            " current model):"
        )

    # setting
    if load_size is None:
        if model_name == 1:
            load_size = [2] * config.num_hidden_layers
        else:
            load_size = [32] * config.num_hidden_layers
    else:
        load_size = [load_size] * config.num_hidden_layers
    prefetch_size = 0
    if model_name == 1:
        prefetch_size = 0
    else:
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
    #print("optimize_and_load_gguf time:", end - start)

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
    # 打开并读取tasks.jsonl文件
    if model_name == 0:
        t = "DeepseekV2"
    elif model_name == 1:
        t = "Mixtral"
    elif model_name == 2:
        t = "Qwen2Moe"
    file_name = "result/new_" + t + "_" + str(load_size[0]) + ".txt"
    # optimize_rule_name = optimize_rule_name[:-5]
    # file_name = "result/" + optimize_rule_name + ".txt"
    # file_name = "result/our_" + t + "_" + str(load_size[0]) + ".txt"
    # file_name = "result/all.txt"
    # file_name = "test.log"

    #"user input: "
    print("User: ", end="")
    # user_input = input()
    user_input = "write quick sort algorithm in python"
    #print("\n")
    messages = [{"role": "user", "content":user_input}]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    torch.set_default_dtype(torch.bfloat16)  # TODO: Remove this, replace dtype using config

    generated, prefill_time, tokens_generated, total_time = prefill_and_generate(
        model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph
    )


    # with open(
    #     "/data/home/yanfansun/ktrans/ktransformers/ktransformers/tasks_full.jsonl", "r", encoding="utf-8"
    # ) as file:
    #     # try:
    #     totall_prefill_time = 0
    #     cnt = 0
    #     totall_tokens_generated = 0
    #     totall_total_time = 0
    #     lengths = [32, 128, 512, 1024]
    #     # lengths = [128]
    #     grand_total_prefill_time = 0
    #     grand_total_tokens_generated = 0
    #     grand_total_total_time = 0
    #     cache = KExpertsCache._instance
    #     for line in file:
    #         # 解析每一行的字符串
    #         content = line.strip()
    #         print("content:", content)
    #         print("cnt:", cnt)
    #         messages = [{"role": "user", "content": content}]
    #         input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    #         torch.set_default_dtype(torch.bfloat16)  # TODO: Remove this, replace dtype using config

    #         generated, prefill_time, tokens_generated, total_time = prefill_and_generate(
    #             model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph
    #         )
    #         # cache.print_time()
    #         if cnt > 0:
    #             totall_prefill_time += prefill_time
    #             totall_tokens_generated += tokens_generated
    #             totall_total_time += total_time

    #             grand_total_prefill_time += prefill_time
    #             grand_total_tokens_generated += tokens_generated
    #             grand_total_total_time += total_time

    #         cnt += 1
    #         if cnt % 10 == 1 and cnt > 1:
    #             print(f"Total prefill time for last 10 iterations: {grand_total_prefill_time}")
    #             print(f"Total tokens generated for last 10 iterations: {grand_total_tokens_generated}")
    #             print(f"Total time for last 10 iterations: {grand_total_total_time}")

    #             # 将结果写入文件
    #             with open(file_name, "a") as f:
    #                 f.write(
    #                     f"Lengths: {lengths[cnt // 10 - 1]}\n"
    #                     f"Total prefill time for last 10 iterations: {grand_total_prefill_time}\n"
    #                     f"Total tokens generated for last 10 iterations: {grand_total_tokens_generated}\n"
    #                     f"Total time for last 10 iterations: {grand_total_total_time}\n"
    #                 )

    #             # 重置累加器
    #             grand_total_prefill_time = 0
    #             grand_total_tokens_generated = 0
    #             grand_total_total_time = 0

    # print(f"prompt eval duration: {totall_prefill_time/cnt}s")
    # print(f"eval count:           {totall_tokens_generated} token(s)")
    # print(f"eval duration:        {totall_total_time}s")
    # tokens_per_second = totall_tokens_generated / totall_total_time
    # print(f"eval rate:            {tokens_per_second} tokens/s")
    # print("fin")
    # with open(file_name, "a") as f:
    #     f.write(
    #         "Summary:\n"
    #         f"prompt eval duration: {totall_prefill_time/cnt}s\n"
    #         f"eval count:           {totall_tokens_generated} token(s)\n"
    #         f"eval duration:        {totall_total_time}s\n"
    #         f"eval rate:            {tokens_per_second} tokens/s\n"
    #     )


if __name__ == "__main__":
    # 解析args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=int, default=None)
    parser.add_argument("--optimize_rule_name", type=str, default=None)
    parser.add_argument("--load_size", type=int, default=None)
    model_name = parser.parse_args().model_name
    optimize_rule_name = parser.parse_args().optimize_rule_name
    if model_name is None:
        model_name = 2
    load_size = parser.parse_args().load_size
    if optimize_rule_name is not None:
        optimize_rule_path = ktransformer_rules_dir + optimize_rule_name
    else:
        optimize_rule_path = None
    print("model_name:", model_name)
    print("optimize_rule_name:", optimize_rule_name)
    print("load_size:", load_size)
    local_chat(
        model_name=model_name,
        optimize_rule_path=optimize_rule_path,
        optimize_rule_name=optimize_rule_name,
        load_size=load_size,
    )
