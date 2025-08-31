# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
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

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""


from datetime import timedelta
from typing import Any, Optional, Union

import os
import ray
import torch
import torch.distributed
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from torch._C._distributed_c10d import ProcessGroup
import vllm
from vllm.worker.worker import Worker
from vllm.logger import init_logger
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from logging import getLogger


logger = init_logger(__name__)
import pickle
import datetime
from torch.distributed import ProcessGroup, TCPStore

class CustomStatelessProcessGroup(StatelessProcessGroup):
    # New field to isolate barrier rounds.
    barrier_counter: int = 0

    def barrier(self):
        """
        A more resilient barrier that uses a dedicated key namespace.
        """
        # Obtain a unique identifier for this barrier round
        barrier_id = self.barrier_counter
        self.barrier_counter += 1
        print("Construct barrier with id:", self.barrier_counter)
        print("Num keys:", self.store.num_keys())

        # Construct a unique namespace for this barrier round.
        barrier_prefix = f"barrier/{barrier_id}"
        my_key = f"{barrier_prefix}/{self.rank}"

        # Each process signals its arrival by setting its token.
        self.store.set(my_key, pickle.dumps("ready"))

        print(f"Getting keys in {my_key}\nThe store: {self.store}")

        # Wait for tokens from all processes.
        for r in range(self.world_size):
            key = f"{barrier_prefix}/{r}"
            # The get call will block until the key is available.
            pickle.loads(self.store.get(key))


    @staticmethod
    def create(
            host: str,
            port: int,
            rank: int,
            world_size: int,
            data_expiration_seconds: int = 3600,
            store_timeout: int = 3600,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """  # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=datetime.timedelta(seconds=store_timeout),
        )

        return CustomStatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds
        )

# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")
    logger.info("Set up backend: %s", str(backend))
    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        logger.info("Setting up rendezvous")
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        logger.info("Rendezvous iterator: %s", str(rendezvous_iterator))
        store, rank, world_size = next(rendezvous_iterator)
        logger.info("Store: %s", str(store))
        logger.info("Rank: %d", rank)
        logger.info("World size: %d", world_size)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)
        print("Set up store: ", store)
    logger.info("Making new process group")
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class WorkerWrap(Worker):
    def _assert_memory_footprint_increased_during_profiling(self):
        return

    def init_process_group(self, master_address: str, master_port: int, rank_offset: int, world_size: int, group_name: str, backend: str = "nccl") -> None:

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        self._model_update_group = CustomStatelessProcessGroup.create(host=master_address, port=master_port, rank=rank_offset, world_size=world_size)
        self.weight_sync_comm = PyNcclCommunicator(
            self._model_update_group,
            device=torch.cuda.current_device(),
        )
        logger.info(
            "Using patched worker supporting weight loading."
            "init_process_group: master_address=%s, master_port=%s, "
            "rank=%d, world_size=%d, group_name=%s. The process binds "
            "to gpus: %s. Default group rank: %d.",
            master_address,
            str(master_port),
            rank_offset,
            world_size,
            str(group_name),
            str(ray.get_gpu_ids()),
            torch.distributed.get_rank(),
        )

    def update_weights(self, name, dtype, shape) -> None:
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.weight_sync_comm.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])
        # logger.info("Weight update is finished for %s.", name)

    def wait_and_barrier(self) -> None:
        torch.cuda.synchronize()
        self._model_update_group.barrier()

    def sync_model_keys(self) -> dict:
        model_keys = self._model_update_group.broadcast_obj(None, src=0)
        return model_keys


if __name__ == "__main__":
    print("VLLM version: ", vllm.__version__)

    llm = vllm.LLM(model="meta-llama/Llama-3.2-3B-Instruct", tensor_parallel_size=2)
    output = llm.generate("San Franciso is a")
    print(f"output: {output}")

    llm = vllm.LLM(model="meta-llama/Llama-3.2-3B-Instruct", tensor_parallel_size=2)
    output = llm.generate("Berkeley is a")
    print(f"output: {output}")
