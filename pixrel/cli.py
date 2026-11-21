from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from pixrel.models.encoder import PixrelEncoder


def _loader(data, batch, workers, train, augment=True, download=True):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train and augment else []
    ops += [transforms.ToTensor(), transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))]
    ds = datasets.CIFAR100(data, train=True if train else False, download=download, transform=transforms.Compose(ops))
    return ds


def _data(args):
    train_all = _loader(args.data, args.batch_size, args.workers, True)
    g = torch.Generator().manual_seed(args.seed)
    n_val = 5000
    train_set, val_set = random_split(train_all, [len(train_all)-n_val, n_val], generator=g)
    # Validation data must not inherit random train augmentations.
    val_base = _loader(args.data, args.batch_size, args.workers, True, augment=False)
    val_set.dataset = val_base
    train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=g)
    val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test = DataLoader(_loader(args.data, args.batch_size, args.workers, False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train, val, test


def _accuracy(model, loader, device, limit=None):
    model.eval(); correct = total = 0; loss_sum = 0.; batches = 0
    with torch.inference_mode():
        for i, (x,y) in enumerate(loader):
            if limit and i >= limit: break
            x,y=x.to(device),y.to(device); logits=model(x, classify=True)
            loss_sum += nn.functional.cross_entropy(logits,y).item(); correct += (logits.argmax(1)==y).sum().item(); total += y.numel(); batches += 1
    return {"loss": loss_sum/max(batches,1), "accuracy": correct/max(total,1)}


def train_main():
    p=argparse.ArgumentParser(description="Train a PIXREL CIFAR-100 variant")
    p.add_argument("--variant",choices=["adaptive","fixed","conv"],default="adaptive"); p.add_argument("--seed",type=int,default=17)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--threads",type=int,default=2)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto",help="auto uses CUDA when available")
    p.add_argument("--init-encoder",help="optional PIXREL encoder state dict to initialize from")
    p.add_argument("--freeze-encoder",action="store_true",help="train only the classifier; without --init-encoder, probe random frozen features")
    p.add_argument("--workers",type=int,default=0); p.add_argument("--data",default="./data"); p.add_argument("--output",default="./outputs/run")
    p.add_argument("--lr",type=float,default=.1); p.add_argument("--weight-decay",type=float,default=5e-4)
    p.add_argument("--limit-train-batches",type=int); p.add_argument("--limit-val-batches",type=int)
    args=p.parse_args(); _train(args)


def _train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable. In Colab, select a GPU runtime and verify torch.cuda.is_available().")
    device=torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train,val,_=_data(args)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Data ready: {len(train.dataset)} train / {len(val.dataset)} validation examples; device={device_name}", flush=True)
    model=PixrelEncoder(variant=args.variant).to(device)
    if args.init_encoder:
        if args.freeze_encoder is False:
            print("Initializing encoder weights; encoder will be fine-tuned.", flush=True)
        encoder_state=torch.load(args.init_encoder,map_location="cpu",weights_only=False)
        model.load_state_dict(encoder_state,strict=True)
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("classifier."))
        source = "initialized encoder" if args.init_encoder else "randomly initialized encoder"
        print(f"Encoder frozen ({source}); training classifier only.", flush=True)
    optimizer=torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=args.lr,momentum=.9,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    best=float("-inf"); history=[]; start=time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start=time.perf_counter()
        print(f"Epoch {epoch+1}/{args.epochs} started ({len(train)} batches)", flush=True)
        model.train(); loss_total=0.; seen=0
        for i,(x,y) in enumerate(train):
            if args.limit_train_batches and i>=args.limit_train_batches: break
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x,classify=True),y); loss.backward(); optimizer.step()
            loss_total += loss.item(); seen += 1
            if seen % 10 == 0:
                elapsed=time.perf_counter()-epoch_start
                print(f"  batch {seen}/{min(len(train), args.limit_train_batches or len(train))} loss={loss.item():.4f} elapsed={elapsed/60:.1f}m", flush=True)
        metrics=_accuracy(model,val,device,args.limit_val_batches); scheduler.step()
        row={"epoch":epoch+1,"train_loss":loss_total/max(seen,1),"val_loss":metrics["loss"],"val_accuracy":metrics["accuracy"],"lr":scheduler.get_last_lr()[0]}
        history.append(row); print(json.dumps(row),flush=True)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch+1,"seed":args.seed,"variant":args.variant,"config":vars(args)}
        torch.save(state,output/"last.pt")
        if metrics["accuracy"]>best:
            best=metrics["accuracy"]; torch.save(state,output/"best.pt")
        (output/"history.json").write_text(json.dumps(history,indent=2))
    elapsed=time.perf_counter()-start
    summary={"variant":args.variant,"seed":args.seed,"elapsed_seconds":elapsed,"parameters":sum(p.numel() for p in model.parameters()),"best_val_accuracy":best,"device":str(device),"device_name":device_name,"config":vars(args)}
    (output/"run.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)


def eval_main():
    p=argparse.ArgumentParser(description="Evaluate a PIXREL checkpoint on CIFAR-100 test split")
    p.add_argument("--checkpoint",required=True); p.add_argument("--data",default="./data"); p.add_argument("--output",default="eval.json")
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--threads",type=int,default=2); p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto",help="auto uses CUDA when available")
    args=p.parse_args(); torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device=torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=PixrelEncoder(variant=state["variant"]).to(device); model.load_state_dict(state["model"])
    ds=_loader(args.data,args.batch_size,args.workers,False); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    result={**_accuracy(model,loader,device),"checkpoint":str(args.checkpoint),"seed":state["seed"],"variant":state["variant"],"device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result))
