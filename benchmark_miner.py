import numpy as np
import torch

from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletMiner, NumpyTripletMiner
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.callback.base import BaseCallback
from timm import create_model

from jina import Document, DocumentArray
from time import perf_counter

class Timer(BaseCallback):
    def __init__(self):
        self.times = []

    def on_train_batch_begin(self, tuner):
        self.t0 = perf_counter()

    def on_train_batch_end(self, tuner):
        torch.cuda.synchronize()
        self.times.append(perf_counter() - self.t0)

def run_test(data, model_name='resnet18', batch_size=32, device='cuda', miner='torch'):
    model = create_model(model_name, num_classes=100)

    timer = Timer()
    if miner == 'torch':
        tuner = PytorchTuner(model, TripletLoss(miner=TripletMiner()), callbacks=[timer])
    else:
        tuner = PytorchTuner(model, TripletLoss(miner=NumpyTripletMiner()), callbacks=[timer])
    tuner.fit(data, epochs=1, batch_size=batch_size, num_items_per_class=4, device=device)
    return timer

def gen_fn(n: int, dim: int, n_cls: int = 4):
        docs = DocumentArray()
        for i in range(n):
            d = Document(
                blob=np.random.rand(3, dim, dim).astype(np.float32),
                tags={'finetuner_label': i % n_cls},
            )
            docs.append(d)
        return docs


settings = [
    {'batch_size': 512, 'dimension': 224, 'model_name': 'resnet18', 'device': 'cuda'},
    {'batch_size': 128, 'dimension': 224, 'model_name': 'resnet18', 'device': 'cuda'},

    {'batch_size': 256, 'dimension': 128, 'model_name': 'resnet18', 'device': 'cuda'},

    {'batch_size': 256, 'dimension': 224, 'model_name': 'resnet18', 'device': 'cpu'},

    {'batch_size': 256, 'dimension': 224, 'model_name': 'resnet34', 'device': 'cuda'},
    {'batch_size': 128, 'dimension': 224, 'model_name': 'resnet50', 'device': 'cuda'},
]

for setting in settings:
    data = gen_fn(10_000, setting['dimension'], 200)

    timer = run_test(data, batch_size=setting['batch_size'], model_name=setting['model_name'], device=setting['device'], miner='torch')

    print(setting)
    print(np.mean(timer.times))
    print(np.var(timer.times))

    timer = run_test(data, batch_size=setting['batch_size'], model_name=setting['model_name'], device=setting['device'], miner='numpy')

    print(setting)
    print(np.mean(timer.times))
    print(np.var(timer.times))
