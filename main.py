import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Main Script to Prepare data, train + validate LR model and produce results
import hydra
import wandb
import lir
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from tqdm import tqdm
from sklearn.metrics import f1_score
from omegaconf import OmegaConf

from utils import load_data, split_train_select, setup_scorer, setup_calibrator, CalibratedScorer, compute_cmxe

def train_select_validate(cfg, wandb_available):
    # Use selection sets defined in config
    if cfg.eval.do_cv:
        pp_list = cfg.eval.pp_list
        n = cfg.eval.sel_set_size
        sel_sets = [pp_list[i:i + n] for i in range(0, len(pp_list), n)]
    else:
        sel_sets = [cfg.eval.sel_pp]

    # Store all LRs and labels across selection sets
    lrs_sel = []
    labels_sel = []
    pps_sel = []
    phones_sel = []
    carrylocs_sel = []
    preds_sel = []
    likelihoods_sel = []

    print("Using data path: ", cfg.eval.data_path)
    print(f"Loading datasets w/ following activities: {cfg.eval.activity_pair}\n")
    
    # Cross-validation over selection sets
    for sel_set in sel_sets:
        data = load_data(cfg)
        try:
            train_data, train_labels, train_pp, train_phone, train_carryloc,\
            sel_data, sel_labels, sel_pp, sel_phone, sel_carryloc, n_clusters = split_train_select(cfg, data, sel_set)
        except Exception as e:
            print(f"Error: `{e}` occured while splitting data for selection set {sel_set}.")
            print("Skipping this selection set.")
            print()
            continue

        # Get scorer and calibrator
        scorer = setup_scorer(cfg, train_data, train_labels)
        calibrator = setup_calibrator(cfg)

        # Set up LR system
        lr_system = CalibratedScorer(cfg, scorer, calibrator)
        lr_system.fit(train_data, train_labels)
        sel_lrs = lr_system.predict_lr(sel_data)
        if cfg.eval.is_multiclass:
            sel_likelihoods = lr_system.calibrator.log_likelihoods
        sel_predictions = lr_system.predict_class(sel_data)

        # Save LRs and labels
        lrs_sel.extend(sel_lrs)
        labels_sel.extend(sel_labels)
        preds_sel.extend(sel_predictions)
        pps_sel.extend(sel_pp)
        phones_sel.extend(sel_phone)
        carrylocs_sel.extend(sel_carryloc)
        if cfg.eval.is_multiclass:
            likelihoods_sel.append(sel_likelihoods)

        print()

    # C_llr / C_mxe calculations
    if not cfg.eval.is_multiclass:
        # Save LRs, labels, and other metadata to dataframe    
        out_df = pd.DataFrame({
            'lrs': lrs_sel,
            'labels': labels_sel,
            'pps': pps_sel,
            'phones': phones_sel,
            'carrylocs': carrylocs_sel
        })

        # Evaluate all systems together
        lrs = np.array(lrs_sel)
        labels = np.array(labels_sel)
        lr_system = CalibratedScorer(cfg, None, None)  # Dummy system for evaluation
        evaluate_all_systems(cfg, lrs, labels, out_df)
    else: 
        likelihoods_sel = np.vstack(likelihoods_sel)
        out_df = pd.DataFrame({
            'labels': labels_sel,
            'pps': pps_sel,
            'phones': phones_sel,
            'carrylocs': carrylocs_sel
        })
        # Calculate C_mxe
        cmxes = multiway_bootstrap_cmxe(likelihoods_sel, out_df, B=cfg.eval.n_bootstraps)
        cmxe_mean = np.mean(cmxes)
        cmxe_std = np.std(cmxes)
        cmxe_ref = np.log2(n_clusters)
        cmxe_normed = cmxe_mean / cmxe_ref
        if wandb_available:
            wandb.log({
                "C_mxe": cmxe_mean,
                "C_mxe_std": cmxe_std,
                "C_mxe_normed": cmxe_normed,
                })
        print(f"C_mxe: {cmxe_mean:.4f} ± {cmxe_std:.4f} (reference: {cmxe_ref:.4f}, normed: {cmxe_normed:.4f})")
    
    # Log overall accuracy
    overall_accuracy = np.mean(np.array(preds_sel) == np.array(labels_sel))
    overall_f1  = f1_score(labels_sel, preds_sel, average='macro')
    if wandb_available:
        wandb.log({
            "accuracy": overall_accuracy,
            "f1_score": overall_f1
        })

    print(f'Overall Accuracy: {overall_accuracy:.3f}')
    print(f'Overall F1 Score: {overall_f1:.3f}')


def evaluate_all_systems(cfg, lrs, labels, out_df, wandb_available=False):
    # Show the distribution of LRs together with the ELUB values
    lrhist_save_loc = "figs/lrhist_plot.png"
    with lir.plotting.show() as ax:
        ax.lr_histogram(lrs, labels)
        H1_legend = mpatches.Patch(color='tab:blue', alpha=.25, label='$H_1$-true')
        H2_legend = mpatches.Patch(color='tab:orange', alpha=.25, label='$H_2$-true')
        ax.legend(handles=[H1_legend, H2_legend])
        ax.savefig(lrhist_save_loc, dpi=300)
        print(f"Saved LR histogram to {lrhist_save_loc}")
        if cfg.eval.save_plots_wandb and wandb_available:
            wandb.log({"lr_histogram": wandb.Image(lrhist_save_loc)})
    
    # Show the PAV plot (closer to the line y=x is better)
    pav_save_loc = "figs/pav_plot.png"
    with lir.plotting.show() as ax:
        ax.pav(lrs, labels)
        ax.savefig(pav_save_loc, dpi=300)
        print(f"Saved PAV plot to {pav_save_loc} (closer to the line y=x is better)")
        if cfg.eval.save_plots_wandb and wandb_available:
            wandb.log({"pav_plot": wandb.Image(pav_save_loc)})

    # Show Tippet plot
    tipp_save_loc = "figs/tippet_plot.png"
    with lir.plotting.show() as ax:
        ax.tippett(lrs, labels)
        ax.savefig(tipp_save_loc, dpi=300)
    if cfg.eval.save_plots_wandb and wandb_available:
        wandb.log({"tippet_plot": wandb.Image(tipp_save_loc)})
    print(f"Saved Tippet plot to {tipp_save_loc}")

    # Show ECE plot
    ece_save_loc = "figs/ece_plot.png"
    with lir.plotting.show() as ax:
        ax.ece(lrs, labels)
        ax.savefig(ece_save_loc, dpi=300)
        print(f"Saved ECE plot to {ece_save_loc}")
        if cfg.eval.save_plots_wandb and wandb_available:
            wandb.log({"ece_plot": wandb.Image(ece_save_loc)})

    # Calculate log likelihood ratio cost (lower is better)
    # Use multiway bootstrap to get confidence intervals while mainiting dependence
    cllrs, cllrs_min, cllrs_cal = multiway_bootstrap_cllr(out_df, B=cfg.eval.n_bootstraps)
    cllr_mean = np.mean(cllrs)
    cllr_std = np.std(cllrs)
    cllr_min_mean = np.mean(cllrs_min)
    cllr_cal_mean = np.mean(cllrs_cal)
    cllr_25, cllr_975 = np.percentile(cllrs, [2.5, 97.5])
    cllr_min_25, cllr_min_975 = np.percentile(cllrs_min, [2.5, 97.5])
    cllr_cal_25, cllr_cal_975 = np.percentile(cllrs_cal, [2.5, 97.5])
    
    print(f'\nC_llr:\t {cllr_mean:.3f} ({cllr_std:.3f}) [{cllr_25:.3f},{cllr_975:.3f}] (lower is better)')
    print(f'C_llr min:\t {cllr_min_mean:.3f}')
    print(f'C_llr cal:\t {cllr_cal_mean:.3f}')
    if wandb_available:
        wandb.log({"cllr": cllr_mean,
                "cllr_25": cllr_25,
                "cllr_975": cllr_975,
                "cllr_std": cllr_std
                }),
        wandb.log({"cllr_min": cllr_min_mean,
                "cllr_min_25": cllr_min_25,
                "cllr_min_975": cllr_min_975})
        wandb.log({"cllr_cal": cllr_cal_mean,
                "cllr_cal_25": cllr_cal_25,
                "cllr_cal_975": cllr_cal_975})

def multiway_bootstrap_cllr(out_df, B=1000):
    persons = out_df['pps'].unique()
    phones = out_df['phones'].unique()
    carrylocs = out_df['carrylocs'].unique()
    Np, Nph, Ncl = len(persons), len(phones), len(carrylocs)

    cllrs = []
    cllrs_min = []
    cllrs_cal = []

    print("performing bootstrap...")
    for _ in tqdm(range(B)):
        sampled_persons = np.random.choice(persons, size=Np, replace=True)
        sampled_phones = np.random.choice(phones, size=Nph, replace=True)
        sampled_carrylocs = np.random.choice(carrylocs, size=Ncl, replace=True)

        # Create bootstrap sample by resampling rows
        boot_indices = []
        
        for pp in sampled_persons:
            for phone in sampled_phones:
                for carryloc in sampled_carrylocs:
                    # Find all rows matching this combination
                    mask = (out_df['pps'] == pp) & (out_df['phones'] == phone) & (out_df['carrylocs'] == carryloc)
                    matching_indices = out_df.index[mask].tolist()
                    boot_indices.extend(matching_indices)
        
        if not boot_indices:
            continue

        boot_df = out_df.iloc[boot_indices].copy()
        
        if boot_df.empty:
            continue
        if len(boot_df['labels'].unique()) < 2:
            # Skip if there are not enough unique labels
            continue

        cllr = lir.metrics.cllr(boot_df['lrs'], boot_df['labels'])
        cllr_min = lir.metrics.cllr_min(boot_df['lrs'], boot_df['labels'])
        cllr_cal = cllr - cllr_min

        cllrs.append(cllr)
        cllrs_min.append(cllr_min)
        cllrs_cal.append(cllr_cal)

    return np.array(cllrs), np.array(cllrs_min), np.array(cllrs_cal)

def multiway_bootstrap_cmxe(likelihoods_sel, sel_df, B=1000):
    persons = sel_df['pps'].unique()
    phones = sel_df['phones'].unique()
    carrylocs = sel_df['carrylocs'].unique()
    Np, Nph, Ncl = len(persons), len(phones), len(carrylocs)

    cmxes = []

    print("performing bootstrap...")
    for _ in tqdm(range(B)):
        sampled_persons = np.random.choice(persons, size= Np, replace=True)
        sampled_phones = np.random.choice(phones, size= Nph, replace=True)
        sampled_carrylocs = np.random.choice(carrylocs, size= Ncl, replace=True)

        mask = (sel_df['pps'].isin(sampled_persons) &
                sel_df['phones'].isin(sampled_phones) & 
                sel_df['carrylocs'].isin(sampled_carrylocs))

        if mask.sum() == 0:  # Check if any rows match
            continue
            
        # Fix: Use column access instead of .loc with string
        boot_labels = sel_df[mask]['labels'].values
        boot_likelihoods = likelihoods_sel[mask]  # Apply mask to likelihoods too
        
        if len(np.unique(boot_labels)) < 2:
            # Skip if there are not enough unique labels
            continue

        cmxe = compute_cmxe(boot_likelihoods, boot_labels)  # Use masked likelihoods
        cmxes.append(cmxe)

    return np.array(cmxes)

@hydra.main(config_path="./conf", config_name="config_main", version_base=None)
def main(cfg): 
    try: 
        wandb.init(
            project=cfg.wandb.project,
            job_type=cfg.wandb.job_type,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            settings=wandb.Settings(start_method="thread")
        )
        wandb_available = True
    except Exception as e:
        print(f"Warning: wandb init failed with error `{e}`. \nContinuing without wandb logging.")
        wandb_available = False

    train_select_validate(cfg, wandb_available)

if __name__ == "__main__":
    main()