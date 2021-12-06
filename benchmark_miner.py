import numpy as np
import torch

from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletMiner, NumpyTripletMiner
from finetuner.tuner.pytorch import PytorchTuner
from timm import create_model

from jina import Document, DocumentArray


def run_test(model_name='resnet_18', batch_size=32, dimension=224):

    def gen_fn(n: int, dim: int, n_cls: int = 4):
        docs = DocumentArray()
        for i in range(n):
            d = Document(
                blob=np.random.rand(3, dim, dim).astype(np.float32),
                tags={'finetuner_label': i % n_cls},
            )
            docs.append(d)
        return docs


    data = gen_fn(10_000, 224, 100)
    model = create_model(model_name, num_classes=100)

    tuner = PytorchTuner(model, TripletLoss(miner=TripletMiner()))
    tuner.fit(data, epochs=2, batch_size=32, num_items_per_class=4)
