################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from cProfile import run
import torch
print(f"PyTorch is using {torch.get_num_threads()} thread(s)")
import random
from hdpy.utils import seed_rngs
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from hdpy.dataset import ECFPFromSMILESDataset, StreamingECFPDataset, StreamingComboDataset, SMILESDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import TensorDataset
from pathlib import Path
from hdpy.model import run_mlp, run_hdc, get_model
from hdpy.metrics import compute_enrichment_factor, compute_roc_enrichment


SCRATCH_DIR = "/p/vast1/jones289/"


def main(
    args,
    config,
    model,
    train_dataset,
    test_dataset,
    encode=True,
    result_dict=None,
    result_path=None,
):
    
    # import pdb 
    # pdb.set_trace()
    if config.model in ["molehd", "selfies", "ecfp", "rp", "directecfp", "combo"]:
        result_dict = run_hdc(
            model=model,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            random_state=args.random_state,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            encode=encode,
            result_dict=result_dict,
            result_path=result_path,
        )
    # elif config.model in ["mlp"]:
    elif "mlp" in config.model:
        result_dict = run_mlp(
            config=config,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            random_state=args.random_state,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

    elif config.model in ["logistic"]:
        # model = LogisticRegression(solver="liblinear")

        x_train, y_train = train_dataset.fps, train_dataset.labels
        x_test, y_test = test_dataset.fps, test_dataset.labels

        result_dict = {"trials": {}}



        for i in range(args.n_trials):
            print(f"{args.config} trial {i}")
            seed = args.random_state + i
            # this should force each call of .fit to be different..
            seed_rngs(seed)

            model.fit(X=x_train, y=y_train)

            y_ = model.predict(X=x_test)
            scores = model.predict_proba(X=x_test)



            # import pdb
            # pdb.set_trace()
            class_report = classification_report(y_true=y_test, y_pred=y_)
            roc_auc = roc_auc_score(y_score=scores[:,1], y_true=y_test) 
            roc_enrich_1 = compute_roc_enrichment(scores=scores[:,1], labels=y_test, fpr_thresh=.01)
            enrich_1 = compute_enrichment_factor(scores=scores[:,1], labels=y_test, n_percent=.01)
            enrich_10 = compute_enrichment_factor(scores=scores[:,1], labels=y_test, n_percent=.1)
            
            result_dict["trials"][i] = {
                                "y_true": y_test, 
                                "y_pred": y_, 
                                "y_score": scores,
                                "class_report": class_report, 
                                "roc-auc": roc_auc,
                                "er-1": roc_enrich_1,
                                "enrich-1": enrich_1,
                                "enrich-10": enrich_10,
                            }
            
            # result_dict = {
                # "trials": {
                        # 0:
                            # {
                                # "y_true": y_test, 
                                # "y_pred": y_, 
                                # "y_score": scores,
                                # "class_report": class_report, 
                                # "roc-auc": roc_auc,
                                # "er-1": roc_enrich_1,
                                # "enrich-1": enrich_1,
                                # "enrich-10": enrich_10,
                            # }
                        # }
            # }


    else:
        raise NotImplementedError

    return result_dict




# def run_trials(model, target_path, target_name, lit_pcba_ave_p, result_dict=None, result_path=None):
def load_litpcba_dataset(model, lit_pcba_ave_p, target_path, target_name):

    smiles_train, smiles_test, y_train, y_test = None, None, None, None

    df = pd.read_csv(
        target_path / Path("full_data.csv")
    )
    
    df["index"] = df.index


    #TODO: have line for liblinear model here?



    # load the smiles strings, if split type is ave then the split has already been computed, other wise load the
    # corresponding file and do the split
    if args.split_type == "random":

        _, test_idxs = train_test_split(
            list(range(len(df))),
            stratify=df["label"],
            random_state=args.random_state,
        )

        df["split"] = ["train"] * len(df)
        df.loc[test_idxs, "split"] = "test"

        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]


        smiles_train = (df["0"][df[df["split"] == "train"]["index"]]).values
        smiles_test = (df["0"][df[df["split"] == "test"]["index"]]).values

        y_train = (df["label"][df[df["split"] == "train"]["index"]]).values
        y_test = (df["label"][df[df["split"] == "test"]["index"]]).values

    else:
        train_df = pd.read_csv(
            lit_pcba_ave_p / Path(f"{target_name}/train_data.csv"),
        )
        smiles_train = train_df['0'].values

        test_df = pd.read_csv(
            lit_pcba_ave_p / Path(f"{target_name}/test_data.csv")
        )

        #todo: there may be an issue with how smiles_test is being saved for molformer
        smiles_test = test_df['0'].values

        y_train = train_df["label"].values
        y_test = test_df["label"].values

    if config.embedding == "ecfp":

        train_dataset = ECFPFromSMILESDataset(
            smiles=smiles_train,
            labels=y_train,
            ecfp_length=config.ecfp_length,
            ecfp_radius=config.ecfp_radius,
        )

        test_dataset = ECFPFromSMILESDataset(
            smiles=smiles_test,
            labels=y_test,
            ecfp_length=config.ecfp_length,
            ecfp_radius=config.ecfp_radius,
        )

    elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
        # its assumed in this case you are using an HD model, this could change..
        train_dataset = SMILESDataset(
            smiles=smiles_train,
            labels=y_train,
            D=config.D,
            tokenizer=config.embedding,
            ngram_order=config.ngram_order,
            num_workers=16,
            device=device,
        )
        # use the item_memory generated by the train_dataset as a seed for the test, then update both?
        test_dataset = SMILESDataset(
            smiles=smiles_test,
            labels=y_test,
            D=config.D,
            tokenizer=config.embedding,
            ngram_order=config.ngram_order,
            item_mem=train_dataset.item_mem,
            num_workers=1,
            device=device,
        )

        train_dataset.item_mem = test_dataset.item_mem
        model.item_mem = train_dataset.item_mem

    elif config.embedding in ["molformer", "molformer-ecfp-combo"]:

        if args.split_type == "random":


            full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
            "molformer_embedding_N-Step-Checkpoint_3_30000_full.npy"
            )
            if config.embedding == "molformer-ecfp-combo":
                print("loading combo model")
                full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_full.npy"
                )


            data = np.load(full_molformer_path)

            train_data = data[train_df["index"].values, :]
            test_data = data[test_df["index"].values, :]

            
            x_train = train_data[:, :-1] 
            y_train = train_data[:, -1]

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            train_dataset = TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).int(),
            )
            test_dataset = TensorDataset(
                torch.from_numpy(x_test).float(),
                torch.from_numpy(y_test).int(),
            )

        else:
            train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
            "molformer_embedding_N-Step-Checkpoint_3_30000_train.npy"
            )

            test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
            "molformer_embedding_N-Step-Checkpoint_3_30000_test.npy"
            )

            if config.embedding == "molformer-ecfp-combo":
                print("loading combo model")
                train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_train.npy"
                )
                test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_test.npy"
                )


            train_data = np.load(train_molformer_path)
            test_data = np.load(test_molformer_path)

            x_train = train_data[:, :-1] 
            y_train = train_data[:, -1]

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            train_dataset = TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).int(),
            )
            test_dataset = TensorDataset(
                torch.from_numpy(x_test).float(),
                torch.from_numpy(y_test).int(),
            )
    
    elif config.embedding == "molclr": 
        # we're just using the GIN model always 
        if args.split_type == "random":

            data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/lit-pcba/full_{target_name}.npy")
            
            train_data = data[train_df["index"].values, :]
            test_data = data[test_df["index"].values, :]

            train_dataset = TensorDataset(torch.from_numpy(normalize(train_data[:, :-1], norm="l2", axis=0)).float(), 
                                        torch.from_numpy(train_data[:, -1]).float())
            
            test_dataset = TensorDataset(torch.from_numpy(normalize(test_data[:, :-1], norm="l2", axis=0)).float(), 
                                        torch.from_numpy(test_data[:, -1]).float())
        else:
            train_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/lit-pcba/train_{target_name}.npy")
            test_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/lit-pcba/test_{target_name}.npy")

            train_dataset = TensorDataset(torch.from_numpy(normalize(train_data[:, :-1], norm="l2", axis=0)).float(), 
                                        torch.from_numpy(train_data[:, -1]).float())
            test_dataset = TensorDataset(torch.from_numpy(normalize(test_data[:, :-1], norm="l2", axis=0)).float(), 
                                        torch.from_numpy(test_data[:, -1]).float())

    elif config.embedding == "directecfp":


        train_dataset = StreamingECFPDataset(smiles_list=smiles_train, 
                                                labels=y_train, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)
        test_dataset = StreamingECFPDataset(smiles_list=smiles_test, 
                                                labels=y_test, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)

    elif config.embedding == "molformer-decfp-combo":


        if args.split_type == "random":


            full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
            "molformer_embedding_N-Step-Checkpoint_3_30000_full.npy"
            )
            if config.embedding == "molformer-ecfp-combo":
                print("loading combo model")
                full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_full.npy"
                )


            smiles_df = pd.read_csv(full_molformer_path.with_name("full_data.csv"))

            data = np.load(full_molformer_path)

            train_data = data[train_df["index"].values, :]
            test_data = data[test_df["index"].values, :]


            smiles_train = (smiles_df.loc[train_df["index"].values])['0'].values.tolist() 
            smiles_test = (smiles_df.loc[test_df["index"].values])['0'].values.tolist()

            x_train = train_data[:, :-1] 
            y_train = train_data[:, -1]

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]


        else:

            train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                "molformer_embedding_N-Step-Checkpoint_3_30000_train.npy"
                )

            test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
            "molformer_embedding_N-Step-Checkpoint_3_30000_test.npy"
            )

            train_data = (np.load(train_molformer_path)).astype(float)
            test_data = (np.load(test_molformer_path)).astype(float)


            smiles_train_df = pd.read_csv(train_molformer_path.with_name("train.csv"))
            smiles_test_df = pd.read_csv(test_molformer_path.with_name("test.csv"))

            smiles_train = smiles_train_df['0'].values.tolist()
            smiles_test = smiles_test_df['0'].values.tolist()

            x_train = train_data[:, :-1] 
            y_train = train_data[:, -1]

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

        print(train_data.shape, test_data.shape)
        train_dataset = StreamingComboDataset(smiles_list=smiles_train,
                                                feats=x_train,
                                                labels=y_train, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)
        test_dataset = StreamingComboDataset(smiles_list=smiles_test,
                                                feats=x_test, 
                                                labels=y_test, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)



    else:
        raise NotImplementedError


    return train_dataset, test_dataset, smiles_train, smiles_test, y_train, y_test

def run_trials(model, target_path, target_name, lit_pcba_ave_p, result_dict=None, result_path=None):



    train_dataset, test_dataset, smiles_train, smiles_test, y_train, y_test = load_litpcba_dataset(model=model, lit_pcba_ave_p=lit_pcba_ave_p, target_path=target_path, target_name=target_name)


    encode = True
    if config.embedding == "directecfp":
        encode = False
    result_dict = main(args=args, config=config,
        model=model, train_dataset=train_dataset, test_dataset=test_dataset, encode=encode,
        result_path=result_path, result_dict=result_dict,

    )

    result_dict["smiles_train"] = smiles_train
    result_dict["smiles_test"] = smiles_test
    result_dict["y_train"] = y_train
    result_dict["y_test"] = y_test

    result_dict["args"] = config
    torch.save(result_dict, result_path)
    print(f"done. output file: {result_path}")
    return result_dict






def driver():

    model = get_model(config) 

    if config.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        # transfer the model to GPU memory
        model = model.to(device).float()

        print("model is on the gpu")

    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(config)

    result_dict = None
    roc_values = (
        []
    )  # some datasets contain multiple targets, store these values then print at end
    std_values = []


    enrich_1_values, enrich_10_values = [], []
    er_1_mean_values = []
    er_1_std_values = []

    lit_pcba_ave_p = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")

    target_list = list(lit_pcba_ave_p.glob("*/"))

    random.shuffle(target_list)
    for target_path in tqdm(target_list):
        target_name = target_path.name

        dataset = args.dataset

        if dataset == "lit-pcba-ave":
            dataset = "lit-pcba"

        output_file = Path(
            f"{output_result_dir}/{exp_name}.{dataset}-{target_path.name}-{args.split_type}.{args.random_state}.pkl"
        )
        print(f"{output_file}\t{output_file.exists()}")

        if output_file.exists():

        
            print(f"output_file: {output_file} exists. skipping.")
            result_dict = torch.load(output_file)

            result_dict = run_trials(model=model, target_path=target_path, target_name=target_name, lit_pcba_ave_p=lit_pcba_ave_p, result_dict=result_dict,
                            result_path=output_file)
        else:
            result_dict = run_trials(model=model, target_path=target_path, target_name=target_name, lit_pcba_ave_p=lit_pcba_ave_p,
                            result_path=output_file) 




        roc_values.append(
            np.mean([value["roc-auc"] for value in result_dict["trials"].values()])
        )

        std_values.append(np.std([value["roc-auc"] for value in result_dict["trials"].values()]))
        
        try:
            enrich_1_values.append(
                np.mean([value["enrich-1"] for value in result_dict["trials"].values()])
            )

            enrich_10_values.append(
                np.mean([value["enrich-10"] for value in result_dict["trials"].values()])
            )
        
        except KeyError as e:
            print(f"{e}. result missing enrichment metrics. computing these now.")

            for trial_idx, _ in result_dict['trials'].items():

                scores = None
                if config.model in ["molehd", "selfies", "ecfp", "rp", "directecfp"]:
                    scores = result_dict["trials"][trial_idx]["eta"]
                elif config.model == "mlp":
                    scores = result_dict["trials"][trial_idx]["eta"][:, 1]
                
                result_dict["trials"][trial_idx]["enrich-1"]  = compute_enrichment_factor(scores=scores, 
                                                        labels=result_dict["trials"][trial_idx]["y_true"],
                                                        n_percent=.01)
                result_dict["trials"][trial_idx]["enrich-10"] = compute_enrichment_factor(scores=scores,
                                                                    labels=result_dict["trials"][trial_idx]["y_true"],
                                                                    n_percent=.1)


            enrich_1_values.append(
                np.mean([value["enrich-1"] for value in result_dict["trials"].values()])
            )

            enrich_10_values.append(
                np.mean([value["enrich-10"] for value in result_dict["trials"].values()])
            )
            torch.save(result_dict, output_file)

        try:
            mean_er_1 = np.mean([value["er-1"] for value in result_dict["trials"].values()])
            std_er_1 = np.std([value["er-1"] for value in result_dict["trials"].values()])
            er_1_mean_values.append(
               mean_er_1 
            )
            er_1_std_values.append(
               std_er_1 
            )

        except KeyError as e:
            
            print(f"{e}. result missing roc-enrichment metrics. computing these now.")
            for trial_idx in result_dict["trials"].keys():
                scores = None 
                
                if "er-1" not in result_dict["trials"][trial_idx].keys():

                    if config.model in ["molehd", "selfies", "ecfp", "rp", "directecfp", "combo"]:
                        scores = result_dict["trials"][trial_idx]["eta"]
                    elif "mlp" in config.model:
                        scores = result_dict["trials"][trial_idx]["eta"][:, 1]
                    
                    labels = result_dict["trials"][trial_idx]['y_true']


                    er_1 = compute_roc_enrichment(scores=scores, labels=labels, fpr_thresh=.01)

                    result_dict["trials"][trial_idx]["er-1"] = er_1
            

            mean_er_1 = np.mean([value["er-1"] for value in result_dict["trials"].values()])
            std_er_1 = np.std([value["er-1"] for value in result_dict["trials"].values()])
            er_1_mean_values.append(
               mean_er_1 
            )
            er_1_std_values.append(
               std_er_1 
            )
            torch.save(result_dict, output_file)           

    print(f"Average ROC-AUC is {np.mean(roc_values)} +/- ({np.mean(std_values)}) \t {np.mean(roc_values)*100:.2f} ({np.mean(std_values)*100:.2f})")
    print(f"Median EF-1% is {np.median(enrich_1_values)}")
    print(f"Median EF-10% is {np.median(enrich_10_values)}")
    print(f"Mean ER-1%: {np.mean(er_1_mean_values)}")

if __name__ == "__main__":
    import hdpy.hdc_args as hdc_args

    # args contains things that are unique to a specific run
    args = hdc_args.parse_args()
    assert args.split_type is not None and args.dataset is not None
    assert args.dataset == "lit-pcba-ave"


    # config contains general information about the model/data processing
    config = hdc_args.get_config(args)

    if config.device == "cpu":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    exp_name = f"{Path(args.config).stem}"

    driver()
