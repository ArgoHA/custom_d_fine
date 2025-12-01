import os

import numpy as np
import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed_mode() -> None:
    """
    Single-node torchrun-friendly DDP init.

    torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK.
    We:
    - read LOCAL_RANK
    - set CUDA device
    - init the default process group
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # not running under torchrun
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"

        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=local_rank,  # this removes the barrier() warning
            )
        except TypeError:
            # for older PyTorch versions that don't support device_id
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "RANK" in os.environ and torch.cuda.is_available():
        return int(os.environ["RANK"]) % torch.cuda.device_count()
    return 0


def all_gather_object(obj):
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    object_list = [None for _ in range(world_size)]
    # Use gloo backend for CPU-based object gathering to avoid CUDA memory issues
    if dist.get_backend() == "nccl":
        # For NCCL, we need to use a CPU-based group or handle it differently
        # all_gather_object internally uses gloo for object serialization
        pass
    dist.all_gather_object(object_list, obj)
    return object_list


def reduce_dict(input_dict, average: bool = True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys])
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


def broadcast_scalar(value, src: int = 0):
    world_size = get_world_size()
    if world_size == 1:
        return value

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor([float(value)], device=device)
    if get_rank() == src:
        tensor.fill_(float(value))
    dist.broadcast(tensor, src=src)
    return tensor.item()


def _preds_to_serializable(preds_list):
    """
    Convert prediction/gt dicts with tensors to serializable format (numpy arrays).
    This avoids CUDA memory issues during all_gather_object.
    """
    serializable = []
    for item in preds_list:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                new_item[k] = v.cpu().numpy()
            else:
                new_item[k] = v
        serializable.append(new_item)
    return serializable


def _serializable_to_preds(serializable_list):
    """
    Convert serializable format back to tensors.
    """
    preds = []
    for item in serializable_list:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                new_item[k] = torch.from_numpy(v)
            else:
                new_item[k] = v
        preds.append(new_item)
    return preds


def gather_predictions(local_preds, local_gt):
    """
    Gather predictions and ground truth from all ranks to rank 0.

    Args:
        local_preds: List of prediction dicts from this rank
        local_gt: List of ground truth dicts from this rank

    Returns:
        On rank 0: (all_preds, all_gt) gathered from all ranks
        On other ranks: (None, None)
    """
    world_size = get_world_size()
    if world_size == 1:
        return local_preds, local_gt

    # Convert tensors to numpy to avoid CUDA memory issues during serialization
    local_preds_np = _preds_to_serializable(local_preds)
    local_gt_np = _preds_to_serializable(local_gt)

    # Gather all predictions and gt to rank 0
    all_preds_list = all_gather_object(local_preds_np)
    all_gt_list = all_gather_object(local_gt_np)

    if get_rank() == 0:
        # Flatten the lists from all ranks and convert back to tensors
        gathered_preds = []
        gathered_gt = []
        for preds, gt in zip(all_preds_list, all_gt_list):
            gathered_preds.extend(_serializable_to_preds(preds))
            gathered_gt.extend(_serializable_to_preds(gt))
        return gathered_preds, gathered_gt

    return None, None


def synchronize():
    """
    Synchronize all processes. Use as a barrier.
    """
    if not is_dist_available_and_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()
