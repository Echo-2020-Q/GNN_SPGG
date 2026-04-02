from __future__ import annotations

import copy
import faulthandler
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any

import torch

from Project1.env import SPGGEnv
from Project1.td3 import GraphTD3Trainer
from run_experiment import (
    BASE_EXPERIMENT,
    build_domain_randomization_config,
    build_env_config,
    build_evaluation_env_factories,
    build_graph,
    build_gnn_policy,
    build_output_dir,
    build_trainer_config,
    build_training_curriculum,
)


DEBUG_CONFIG: dict[str, Any] = {
    # True: 只跑 demo collection + actor/critic pretrain
    # False: 继续进入完整 trainer.train()
    "pretrain_only": True,

    # True: 只隔离 demo collection，本轮不做 actor/critic pretrain，最快定位崩点
    "collection_only": True,

    # 调试时建议先小一点
    "demo_collection_env_steps": 10_000,

    # 可选覆盖；None 表示沿用 BASE_EXPERIMENT
    "demo_collection_runtime": None,

    # 可选输出目录名后缀
    "experiment_name_suffix": "__debug_demo_collection",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _prepare_debug_spec() -> dict[str, Any]:
    spec = copy.deepcopy(BASE_EXPERIMENT)
    spec["experiment_name"] = str(spec["experiment_name"]) + str(DEBUG_CONFIG["experiment_name_suffix"])
    spec["run_mode"] = "gnn_train"
    spec["training"]["demo_pretrain_enabled"] = True
    spec["training"]["demo_collection_env_steps"] = int(DEBUG_CONFIG["demo_collection_env_steps"])
    if DEBUG_CONFIG["demo_collection_runtime"] is not None:
        spec["training"]["demo_collection_runtime"] = str(DEBUG_CONFIG["demo_collection_runtime"])
    if bool(DEBUG_CONFIG["collection_only"]):
        spec["training"]["actor_bc_pretrain_updates"] = 0
        spec["training"]["critic_pretrain_updates"] = 0
    return spec


def _print_system_info() -> None:
    print("=== Debug Demo Collection ===")
    print("Python     :", sys.executable)
    print("Version    :", sys.version.replace("\n", " "))
    print("Platform   :", platform.platform())
    print("PID        :", os.getpid())
    print("CWD        :", os.getcwd())
    print("Torch      :", torch.__version__)
    print("CUDA avail :", torch.cuda.is_available())
    print("CUDA count :", torch.cuda.device_count())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print("CUDA[{0}]   : {1}".format(index, torch.cuda.get_device_name(index)))
    print("========================================")


def main() -> None:
    spec = _prepare_debug_spec()
    output_dir = build_output_dir(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "debug_demo_collection.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.__stdout__, log_file)
        sys.stdout = tee
        sys.stderr = tee
        faulthandler.enable(file=log_file, all_threads=True)

        print("Debug Log  :", log_path)
        _print_system_info()
        print("Step 1     : build_graph")
        graph = build_graph(spec)
        print("Step 2     : build_env_config")
        env_config = build_env_config(spec, graph)
        print("Step 3     : build_env")
        env = SPGGEnv(env_config, graph)
        print("Step 4     : build_policy")
        policy = build_gnn_policy(spec)
        print("Step 5     : build_trainer_config")
        trainer_config = build_trainer_config(spec)
        print("Step 6     : build_domain_randomization_config")
        randomization_config = build_domain_randomization_config(spec)
        print("Step 7     : build_evaluation_env_factories")
        eval_env_factories = build_evaluation_env_factories(spec)
        print("Step 8     : build_training_curriculum")
        curriculum_stages = build_training_curriculum(spec)
        print("Step 9     : construct GraphTD3Trainer")
        trainer = GraphTD3Trainer(
            env=env,
            policy=policy,
            config=trainer_config,
            eval_env=env,
            randomization=randomization_config,
            eval_env_factories=eval_env_factories,
            curriculum_stages=curriculum_stages,
        )
        try:
            if bool(DEBUG_CONFIG["pretrain_only"]):
                print("Step 10    : trainer._run_demo_pretrain()")
                summary = trainer._run_demo_pretrain()
                print("Pretrain summary:", summary)
            else:
                print("Step 10    : trainer.train()")
                history = trainer.train(num_updates=1)
                print("Train history length:", len(history))
        except BaseException:
            print("=== Python Exception ===")
            traceback.print_exc()
            raise
        finally:
            print("Step 11    : trainer.close()")
            try:
                trainer.close()
            except Exception:
                traceback.print_exc()
        print("Done.")


if __name__ == "__main__":
    main()
