
import torch
import argparse
import numpy as np
from torch.autograd import Variable

from train import run_epoch
from models.model import make_model
from utils.optimizer import NoamOpt
from utils.helpers import greedy_decode
from utils.regularizer import LabelSmoothing

from data.generate_data import data_gen
from utils.losses import SimpleLossCompute


parser = argparse.ArgumentParser()


# Settings Parameters
parser.add_argument('--V', default=11, type=int)
parser.add_argument('--smoothing', default=0.0, type=float)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--num_epochs', default=10, type=int)

args = parser.parse_args()

criterion = LabelSmoothing(size=args.V, padding_idx=0, smoothing=args.smoothing)
model = make_model(args.V, args.V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9))

print("Start training...")
for epoch in range(args.num_epochs):
    model.train()
    epoch_tr_loss = run_epoch(data_gen(args.V,30,20), model, 
                              SimpleLossCompute(model.generator,criterion, model_opt), epoch)
    model.eval()
    epoch_val_loss = run_epoch(data_gen(args.V,30,5), model, 
                           SimpleLossCompute(model.generator,criterion,None), epoch, "Val")
print("Done.")
print()

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
