################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################

import numpy as np
from argparse import Namespace
import yaml
import pdb
import torch
from pathlib import Path
from functools import partial
import deepchem as dc
import argparse
import random
# random.seed(12345)
import time
random.seed(time.perf_counter_counter())
import multiprocessing as mp
import sys
sys.path.append("/g/g13/jones289/workspace/hd-cuda-master")
from hdpy.ecfp import compute_fingerprint_from_smiles
parser = argparse.ArgumentParser()
parser.add_argument("--include-ecfp", action="store_true")
parser.add_argument("--ecfp-length", type=int, default=1024)
parser.add_argument("--ecfp-radius", type=int, default=1)
parser.add_argument("--dataset", choices=["profile", "lit-pcba", "bace", "hiv", "bbbp", "tox21", "clintox", "sider"])
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--out-dir", default="molformer_variorum_extract")

args = parser.parse_args()

with open('/g/g13/jones289/workspace/hd-cuda-master/hdpy/molformer/notebooks/pretrained_molformer/hparams.yaml', 'r') as f:
    config = Namespace(**yaml.safe_load(f))

from tokenizer.tokenizer import MolTranBertTokenizer

tokenizer = MolTranBertTokenizer('/g/g13/jones289/workspace/hd-cuda-master/hdpy/molformer/notebooks/pretrained_molformer/bert_vocab.txt')

from train_pubchem_light import LightningModule

ckpt = Path('/usr/WS1/jones289/lee218/molformer/data/Pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt')
lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)
print(lm)
total_params = sum(p.numel() for p in lm.parameters())
print(f"MoLFormer has {total_params} parameters.")
# todo: need to 
# lm = torch.nn.DataParallel(lm)

import torch
from tqdm import tqdm
from fast_transformers.masking import LengthMask as LM

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def embed(model, smiles, tokenizer, batch_size=64, return_time=False):
    # pdb.set_trace()
    model.eval()
    model = model.cuda()
    # model.blocks = model.blocks.cuda()

    time_list = []

    embeddings = []
    for batch in tqdm(batch_split(smiles, batch_size=batch_size), total=int(np.ceil(len(smiles)/batch_size))):
        # if return_time:
        # start_time = time.perf_counter_counter()
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']).cuda(), torch.tensor(batch_enc['attention_mask']).cuda()
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():

            starter.record()
            if isinstance(model, torch.nn.DataParallel):
                # todo: this doesn't work yet
                # token_embeddings = model.module.blocks.cuda()(model.tok_emb(idx.cuda()), length_mask=LM(mask.sum(-1).cuda()))
                token_embeddings = model.module.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            else:
                # token_embeddings = model.blocks.cuda()(model.tok_emb(idx.cuda()), length_mask=LM(mask.sum(-1).cuda()))
                token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            # token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float().cuda()
        # pdb.set_trace()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        ender.record()
        torch.cuda.synchronize()
        batch_time = starter.elapsed_time(ender) / 1000 # convert milliseconds to seconds

        embeddings.append(embedding.detach().cpu())

        if return_time:
            time_list.append(batch_time)
    if return_time:
        # import pdb
        # pdb.set_trace()
        return torch.cat(embeddings), np.array(time_list) # the first batch is usally the slowest and can mess up the timing information
    else:
        return torch.cat(embeddings)

from rdkit import Chem
from sklearn.linear_model import LogisticRegression

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

from pathlib import Path

import pandas as pd


def energy_profile_main():
    # import pdb 
    # pdb.set_trace()
    # input_dir = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")
    # path_list = list(input_dir.glob("*/"))
    # path_list = [Path("/p/vast1/jones289/lit_pcba/AVE_unbiased/VDR")]
    # random.shuffle(path_list)
    # for path in path_list:
        # target_name = path.name


        # for split in ["full"]:


            # input_path = input_dir / target_name / Path(f"{split}_data.csv")

            # I'm appending the name of the molformer checkpoint so its easier to compare multiple
            # output_path = input_dir / target_name / Path(f"molformer_embedding_{ckpt.stem}_{split}-time.npy")

            # if args.include_ecfp:

                # output_path = input_dir / target_name / Path(f"molformer_embedding_ecfp_{args.ecfp_length}_{args.ecfp_radius}_{ckpt.stem}_{split}-time.npy")


            # print(input_path, output_path)
            # smiles_col = '0'
            

            # df = pd.read_csv(input_path)

            # print(f"{path}\t{input_path} {input_path.exists()} {df.shape}")

            # smiles = df[smiles_col].apply(canonicalize)
            # labels = df['label']
            output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/hiv")

            if output_molnet_dir.exists():
                pass
            else:
                output_molnet_dir.mkdir(parents=True)

            try: 
                dataset = dc.molnet.load_hiv()
            except ValueError as e:
                print(f"{e}. trying with reload=False")
                dataset = dc.molnet.load_hiv(reload=False)

            smiles_train = dataset[1][0].ids
            smiles_test = dataset[1][1].ids
            # labels_train = dataset[1][0].y
            # labels_test = dataset[1][1].y

            smiles = np.concatenate([smiles_train, smiles_test])

            # start=time.perf_counter()
            X, time_arr = embed(lm, smiles, tokenizer, batch_size=args.batch_size, return_time=True)
            # import pdb
            # pdb.set_trace()
            total_time = np.sum(time_arr)
            mean_time = total_time / X.shape[0]
            # mean_time = np.mean(time_arr)
            # std_time = np.std(time_arr)
            # end = time.perf_counter()
            # total_time = end - start 
            # mean_time = total_time / len(smiles)

            print(f"extracting data took {total_time} seconds for {len(smiles)} molecules (mean: {mean_time}s/mol)")


            out_dir = Path(args.out_dir)
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)

            np.savetxt(out_dir / Path("molformer_time.txt"), [total_time, X.shape[0]])
            # np.save(out_dir / Path("molformer_time.npy"), np.array([time_arr]))

            # data = np.concatenate([X, labels.values.reshape(-1,1)], axis=1)

            # just commenting out for the timing info runs
            # np.save(output_path, data)

            # print(f"lit-pcba VDR profile done.")

def lit_pcba_ave_main():
    # import pdb 
    # pdb.set_trace()
    input_dir = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")
    path_list = list(input_dir.glob("*/"))
    random.shuffle(path_list)
    for path in path_list:
        target_name = path.name


        for split in ["train", "test", "full"]:


            input_path = input_dir / target_name / Path(f"{split}_data.csv")

            # I'm appending the name of the molformer checkpoint so its easier to compare multiple
            output_path = input_dir / target_name / Path(f"molformer_embedding_{ckpt.stem}_{split}-time.npy")

            if args.include_ecfp:

                output_path = input_dir / target_name / Path(f"molformer_embedding_ecfp_{args.ecfp_length}_{args.ecfp_radius}_{ckpt.stem}_{split}-time.npy")


            print(input_path, output_path)
            smiles_col = '0'
            
            if output_path.exists():
                print(f"{output_path} exists, skipping.")
                continue
            else:


                df = pd.read_csv(input_path)
            


                print(f"{path}\t{input_path} {input_path.exists()} {df.shape}")

                smiles = df[smiles_col].apply(canonicalize)
                labels = df['label']



                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # import time

                # start=time.perf_counter()
                starter.record()
                X = embed(lm, smiles, tokenizer, batch_size=args.batch_size).numpy()
                # end = time.perf_counter()
                # total_time = end - start
                ender.record()
                torch.cuda.synchronize()

                total_time = starter.elapsed_time(ender)

                print(f"extracting data took {total_time} seconds for {X.shape[0]} molecules ({total_time/X.shape[0]}s/mol)")

                if args.include_ecfp:
                    print(f"computing ECFPs")

                    fp_job = partial(compute_fingerprint_from_smiles, length=args.ecfp_length, radius=args.ecfp_radius)
                    with mp.Pool(int(mp.cpu_count()/2)) as pool:
                        result = list(tqdm(pool.imap(fp_job, smiles), total=len(smiles)))
                        pool.close()
                        pool.join()

                    fps = torch.from_numpy(np.asarray([x for x in result if x is not None])).int().reshape(-1, args.ecfp_length)

                    X = np.concatenate([X, fps], axis=1)

                else:

                    # I'm appending the name of the molformer checkpoint so its easier to compare multiple
                    # output_path = input_dir / target_name / Path(f"molformer_embedding_{ckpt.stem}.npy")
                    pass

                data = np.concatenate([X, labels.values.reshape(-1,1)], axis=1)

                # just commenting out for the timing info runs
                np.save(output_path, data)


                print(f"{output_path} done.")

def bace_main():

    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/bace")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)
    
    dataset = dc.molnet.load_bace_classification()

    smiles_train = dataset[1][0].ids
    smiles_test = dataset[1][1].ids
    labels_train = dataset[1][0].y
    labels_test = dataset[1][1].y

    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train.reshape(-1,1)], axis=1)
    data_test = np.concatenate([X_test, labels_test.reshape(-1,1)], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)

    print(f"done.")


def hiv_main():

    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/hiv")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)
    
    dataset = dc.molnet.load_hiv()

    smiles_train = dataset[1][0].ids
    smiles_test = dataset[1][1].ids
    labels_train = dataset[1][0].y
    labels_test = dataset[1][1].y

    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train.reshape(-1,1)], axis=1)
    data_test = np.concatenate([X_test, labels_test.reshape(-1,1)], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)

    print(f"done.")


def bbbp_main():

    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/bbbp")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)
    
    dataset = dc.molnet.load_bbbp()

    smiles_train = dataset[1][0].ids
    smiles_test = dataset[1][1].ids
    labels_train = dataset[1][0].y
    labels_test = dataset[1][1].y

    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train.reshape(-1,1)], axis=1)
    data_test = np.concatenate([X_test, labels_test.reshape(-1,1)], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)

    print(f"done.")


def sider_main():

    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/sider")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)

    sider_dataset = dc.molnet.load_sider()

    target_list = sider_dataset[0]


    smiles_train = sider_dataset[1][0].ids
    smiles_test = sider_dataset[1][1].ids

    labels_train = sider_dataset[1][0].y
    labels_test = sider_dataset[1][1].y


    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train], axis=1)
    data_test = np.concatenate([X_test, labels_test], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)


def clintox_main():
    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/clintox")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)

    dataset = dc.molnet.load_clintox(splitter="scaffold")
    smiles_train = dataset[1][0].ids
    smiles_test = dataset[1][1].ids

    labels_train = dataset[1][0].y
    labels_test = dataset[1][1].y

    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train], axis=1)
    data_test = np.concatenate([X_test, labels_test], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)


def tox21_main():
    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/tox21")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)

    dataset = dc.molnet.load_tox21()
    
    train_dataset = dataset[1][0]
    test_dataset = dataset[1][1]

    smiles_train = train_dataset.ids
    labels_train = train_dataset.y

    smiles_test = test_dataset.ids
    labels_test = test_dataset.y

    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train], axis=1)
    data_test = np.concatenate([X_test, labels_test], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)

def sider_main():
    output_molnet_dir = Path("/p/vast1/jones289/molformer_embeddings/molnet/sider")

    if output_molnet_dir.exists():
        pass
    else:
        output_molnet_dir.mkdir(parents=True)
    sider_dataset = dc.molnet.load_sider()

    target_list = sider_dataset[0]


    smiles_train = sider_dataset[1][0].ids
    smiles_test = sider_dataset[1][1].ids

    labels_train = sider_dataset[1][0].y
    labels_test = sider_dataset[1][1].y
    
    X_train = embed(lm, smiles_train, tokenizer, batch_size=args.batch_size).numpy()
    X_test = embed(lm, smiles_test, tokenizer, batch_size=args.batch_size)

    data_train = np.concatenate([X_train, labels_train], axis=1)
    data_test = np.concatenate([X_test, labels_test], axis=1)

    output_train_path = output_molnet_dir / Path(f"train_{ckpt.stem}.npy")
    output_test_path = output_molnet_dir / Path(f"test_{ckpt.stem}.npy")
    output_train_smiles_path = output_molnet_dir / Path(f"train_{ckpt.stem}_smiles.npy")
    output_test_smiles_path = output_molnet_dir / Path(f"test_{ckpt.stem}_smiles.npy")
    np.save(output_train_path, data_train)
    np.save(output_test_path, data_test)
    np.save(output_train_smiles_path, smiles_train)
    np.save(output_test_smiles_path, smiles_test)

if args.dataset == "lit-pcba":
    lit_pcba_ave_main()

# lit_pcba_main()
if args.dataset == "bace":

    bace_main()

if args.dataset == "hiv":
    hiv_main()

if args.dataset == "bbbp":
    bbbp_main()

if args.dataset == "tox21":
    tox21_main()

if args.dataset == "clintox":
    clintox_main()

if args.dataset == "sider":
    sider_main()


if args.dataset == "profile":

    energy_profile_main()
