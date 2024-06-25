import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from resnet_1_data import *
from ResNet import Bottleneck, ResNet, ResNet152

bs = 128
NUM_FEATURES = 128 * 128
NUM_CLASSES = 10
GRAYSCALE = False
RANDOM_SEED = 1

torch.manual_seed(RANDOM_SEED)

############################

def set_nccl_environment():
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "2"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.epoch_losses = []

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        if os.path.exists(snapshot_path):
            snapshot = torch.load(snapshot_path, map_location=lambda storage, loc: storage.cuda(self.local_rank))

            # Check if the model is wrapped in DataParallel or DistributedDataParallel
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model

            model_to_load.load_state_dict(snapshot["MODEL_STATE"], strict=False)
            self.epochs_run = snapshot["EPOCHS_RUN"]
            print(f"Loaded snapshot from {snapshot_path} and running from {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
        return epoch_loss / len(self.train_data)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self.model.train()
            epoch_loss = self._run_epoch(epoch)
            self.epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=8
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot_3.pt"):
    set_nccl_environment()
    ddp_setup()

    dataset = train_dataset
    train_data = prepare_dataloader(dataset, batch_size)
    model = ResNet152(10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    st_time = time.time()  # start_time
    trainer.train(total_epochs)
    end_time = time.time()
    total_time = end_time - st_time
    print(f"Total training time on multinode: {total_time:.2f} seconds")
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=512, type=int, help='Input batch size on each device (default: 256)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size)
