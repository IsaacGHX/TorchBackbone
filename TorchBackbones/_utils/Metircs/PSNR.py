import torch

from ignite.engine import *
from ignite.metrics import *
from ignite.utils import *


# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

manual_seed(666)

if __name__ == '__main__':

    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')
    preds = torch.rand([50000, 3, 16, 16])
    target = preds*0.75

    state = default_evaluator.run([[preds, target]])
    print(state.metrics['psnr'])
