import argparse
import os
import json
import operator
from tqdm import tqdm

import wget
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import logsumexp


from scipy.optimize import minimize
from scipy.special import softmax  # 用于概率混合

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA
from utilsa import get_model_from_sd, test_model_on_dataset, test_cached_predictions_on_dataset


# -----------------------------
#  Argument parsing
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--ensemble", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-ensemble", action="store_true", default=False,
        help="Greedy ensemble in output space using cached logits and val accuracy on ImageNet2p.",
    )
    parser.add_argument(
        "--opt-ensemble", action="store_true", default=False,
        help="Optimize ensemble weights on ImageNet2p via convex prob-mixture cross-entropy.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
    )
    parser.add_argument(
        "--cache-predictions", action="store_true", default=False,
        help="Cache per-model logits for each dataset under model-location/cache/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()


def dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return {"images": batch[0], "labels": batch[1]}
    raise ValueError("Unrecognized batch format, expected dict or (images, labels).")


def get_targets_from_dataset(dataset):
    loader = dataset.test_loader
    if type(dataset).__name__ == 'ImageNet2p':
        loader = dataset.train_loader

    all_targets = []
    for batch in loader:
        batch = dictionarize_batch(batch)
        labels = batch["labels"]
        device = labels.device
        if hasattr(dataset, "project_labels"):
            labels = dataset.project_labels(labels, device)
        all_targets.append(labels.detach().cpu().numpy())

    targets_np = np.concatenate(all_targets, axis=0)
    return targets_np


def load_cached_predictions_for_dataset(model_location, dataset_name, num_models):
    first_path = os.path.join(model_location, "cache", dataset_name, "model_0.pt")
    assert os.path.exists(first_path), f"Cached predictions not found: {first_path}"
    first = torch.load(first_path)  # list[Tensor[B,C]] 或 Tensor[N,C]

    if isinstance(first, list):
        first_tensor = torch.cat(first, dim=0)  # [N, C]
    elif torch.is_tensor(first):
        first_tensor = first
    else:
        raise ValueError(f"Unexpected cached prediction type: {type(first)}")

    N, C = first_tensor.shape
    logits_np = np.zeros((num_models, N, C), dtype=np.float32)
    logits_np[0] = first_tensor.detach().cpu().numpy()

    for m in range(1, num_models):
        path = os.path.join(model_location, "cache", dataset_name, f"model_{m}.pt")
        assert os.path.exists(path), f"Cached predictions not found: {path}"
        preds = torch.load(path)

        if isinstance(preds, list):
            preds_tensor = torch.cat(preds, dim=0)  # [N, C]
        elif torch.is_tensor(preds):
            preds_tensor = preds
        else:
            raise ValueError(f"Unexpected cached prediction type for model_{m}: {type(preds)}")

        assert preds_tensor.shape == (N, C), \
            f"Shape mismatch for cached preds of model_{m} on {dataset_name}: {preds_tensor.shape} vs {(N, C)}"

        logits_np[m] = preds_tensor.detach().cpu().numpy()

    print(f"[cache] Loaded logits for dataset={dataset_name}: shape={logits_np.shape}")
    return logits_np


def build_weighted_predictions_from_cache(model_location, dataset_name, weights):
    
    num_models = len(weights)
    base_path = os.path.join(model_location, 'cache', dataset_name, 'model_0.pt')
    base = torch.load(base_path) 
    num_batches = len(base)

    w0 = float(weights[0])
    predictions = [b * w0 for b in base]


    for m in range(1, num_models):
        w = float(weights[m])
        if w == 0.0:
            continue
        path = os.path.join(model_location, 'cache', dataset_name, f'model_{m}.pt')
        preds_m = torch.load(path)
        assert len(preds_m) == num_batches
        for i in range(num_batches):
            predictions[i] += preds_m[i] * w

    return predictions



def eval_ensemble_accuracy_from_logits(logits_np, weights, targets_np, use_probs=False):
   
    if use_probs:
        probs = softmax(logits_np, axis=2)  # [M, N, C]
        mixed = np.tensordot(weights, probs, axes=(0, 0))  # [N, C]
        preds = mixed.argmax(axis=1)
    else:
        mixed_logits = np.tensordot(weights, logits_np, axes=(0, 0))  # [N, C]
        preds = mixed_logits.argmax(axis=1)

    acc = float((preds == targets_np).mean())
    return acc



def project_to_simplex(v):

    v = np.maximum(v, 0)
    if v.sum() == 0:
        return np.ones_like(v) / len(v)
    return v / v.sum()


def optimize_ce_logits_pgd(logits_np, targets_np,
                           seed=0, maxiter=2000, lr=1e-2):
   
    M, N, C = logits_np.shape

    rng = np.random.RandomState(seed)
    w = rng.rand(M)
    w /= w.sum()

    loss_history = []

    def mixed_logits(w):
        return np.tensordot(w, logits_np, axes=(0, 0))   # [N, C]

    def ce_loss_and_grad(w):
  
        ml = mixed_logits(w)       

        logZ = logsumexp(ml, axis=1)             # [N]
        loss = - np.mean(ml[np.arange(N), targets_np] - logZ)

       
        sm = np.exp(ml - logZ[:, None])          # [N, C]

       
        grad = np.zeros(M, dtype=np.float64)
        for j in range(M):
            lj = logits_np[j]                     # [N, C]
            term1 = lj[np.arange(N), targets_np]  # [N]
            term2 = (sm * lj).sum(axis=1)         # [N]
            grad[j] = - np.mean(term1 - term2)

        return loss, grad

    # PGD iterations
    pbar = tqdm(range(maxiter), desc="Opt-ensemble (logits CE)", ncols=80)
    for _ in pbar:
        loss, grad = ce_loss_and_grad(w)
        loss_history.append(loss)

        w = w - lr * grad
        w = project_to_simplex(w)

        pbar.set_postfix(loss=loss)

    best_idx = np.argmin(loss_history)
    best_loss = loss_history[best_idx]

    print(f"[opt-logits-CE] best_loss={best_loss:.6f}")
    print(f"[opt-logits-CE] best_w[:10]={np.round(w[:10], 4)}")

    return w.astype(np.float32), best_loss, loss_history

# -----------------------------
if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'
    ENSEMBLE_RESULTS_FILE = 'ensemble_results.jsonl'
    GREEDY_ENSEMBLE_RESULTS_FILE = 'greedy_ensemble_results.jsonl'
    OPT_ENSEMBLE_RESULTS_FILE = 'opt_ensemble_results.jsonl'

    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
            )

    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]

    # Step 2: base CLIP model & preprocess
    if (
        args.eval_individual_models
        or args.uniform_soup
        or args.greedy_soup
        or args.ensemble
        or args.greedy_ensemble
        or args.opt_ensemble
    ):
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    # Step 2: Evaluate individual models (optionally caching predictions).
    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model = get_model_from_sd(state_dict, base_model)

            results = {'model_name': f'model_{j}'}
            # ImageNet2p 是 held-out minival
            for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
                print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')
                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                cache_loc = None
                if args.cache_predictions:
                    cache_loc = os.path.join(
                        args.model_location,
                        'cache',
                        dataset_cls.__name__,
                        f'model_{j}.pt'
                    )
                    os.makedirs(os.path.dirname(cache_loc), exist_ok=True)

                accuracy = test_model_on_dataset(
                    model,
                    dataset,
                    cache_loc=cache_loc,
                )
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup (weight-space)
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):
            print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k: v * (1. / NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k: v * (1. / NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = get_model_from_sd(uniform_soup, base_model)

        results = {'model_name': 'uniform_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
            print(f'Evaluating on {dataset_cls.__name__}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    # Step 4: Greedy Soup (weight-space)
    if args.greedy_soup:
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row['ImageNet2p']
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]

        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
        best_val_acc_so_far = individual_model_val_accs[0][1]
        held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)

        # Now, iterate through all models and consider adding them to the greedy soup.
        for i in range(1, NUM_MODELS):
            print(f'Testing model {i} of {NUM_MODELS}')
            new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                   new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

            print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
            if held_out_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = held_out_val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

        # Finally, evaluate the greedy soup.
        model = get_model_from_sd(greedy_soup_params, base_model)
        results = {'model_name': 'greedy_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
            print(f'Evaluating on {dataset_cls.__name__}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    # Step 5: Simple uniform ensemble using cached predictions
    if args.ensemble:
        assert os.path.exists(os.path.join(args.model_location, 'cache')), \
            "Cache directory not found. Run with --eval-individual-models --cache-predictions first."
        if os.path.exists(ENSEMBLE_RESULTS_FILE):
            os.remove(ENSEMBLE_RESULTS_FILE)

        results = {'model_name': 'ensemble'}
        weights = np.ones(NUM_MODELS, dtype=np.float32) / float(NUM_MODELS)

        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
            name = dataset_cls.__name__
            print(f'Evaluating ensemble on {name}.')
            predictions = build_weighted_predictions_from_cache(
                args.model_location, name, weights
            )
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_cached_predictions_on_dataset(predictions, dataset)
            results[name] = accuracy
            print(accuracy)

        with open(ENSEMBLE_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    if args.greedy_ensemble:
        assert os.path.exists(os.path.join(args.model_location, 'cache')), \
            "Cache directory not found. Run with --eval-individual-models --cache-predictions first."
        if os.path.exists(GREEDY_ENSEMBLE_RESULTS_FILE):
            os.remove(GREEDY_ENSEMBLE_RESULTS_FILE)

        
        val_dataset = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)
        val_targets_np = get_targets_from_dataset(val_dataset)  # [N]
        val_logits_np = load_cached_predictions_for_dataset(
            args.model_location, 'ImageNet2p', NUM_MODELS
        )  

        M = NUM_MODELS

        def eval_with_weights_on_val_np(w):
            return eval_ensemble_accuracy_from_logits(
                val_logits_np, w, val_targets_np, use_probs=False
            )

   
        best_idx = 0
        best_acc = -1.0
        for m in range(M):
            w = np.zeros(M, dtype=np.float32)
            w[m] = 1.0
            acc = eval_with_weights_on_val_np(w)
            print(f"[greedy-ens] single model {m} val acc = {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_idx = m

        selected = [best_idx]
        print(f"[greedy-ens] start from model {best_idx}, acc = {best_acc:.4f}")

       
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_candidate_acc = best_acc

            for m in range(M):
                if m in selected:
                    continue
                cand = selected + [m]
                w = np.zeros(M, dtype=np.float32)
                w[cand] = 1.0 / len(cand)
                acc = eval_with_weights_on_val_np(w)
                print(f"[greedy-ens] try add {m}, val acc = {acc:.4f}")
                if acc > best_candidate_acc:
                    best_candidate_acc = acc
                    best_candidate = m

            if best_candidate is not None:
                selected.append(best_candidate)
                best_acc = best_candidate_acc
                improved = True
                print(f"[greedy-ens] add model {best_candidate}, new selected={selected}, acc={best_acc:.4f}")

        weights = np.zeros(M, dtype=np.float32)
        weights[selected] = 1.0 / len(selected)

        results = {
            'model_name': 'greedy_ensemble',
            'indices': selected,
            'val_ImageNet2p': float(best_acc),
        }

        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
            name = dataset_cls.__name__
            print(f'[greedy-ensemble] Evaluating on {name}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            preds = build_weighted_predictions_from_cache(
                args.model_location, name, weights
            )
            acc = test_cached_predictions_on_dataset(preds, dataset)
            results[name] = acc
            print(f'[greedy-ensemble] {name} acc = {acc:.4f}')

        with open(GREEDY_ENSEMBLE_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    # Step 7: Optimal ensemble (convex prob mixture via CE on ImageNet2p)
    
    if args.opt_ensemble:
        assert os.path.exists(os.path.join(args.model_location, 'cache')), \
            "Cache directory not found. Run with --eval-individual-models --cache-predictions first."
        if os.path.exists(OPT_ENSEMBLE_RESULTS_FILE):
            os.remove(OPT_ENSEMBLE_RESULTS_FILE)

      
        val_dataset = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)
        val_targets_np = get_targets_from_dataset(val_dataset)
        val_logits_np = load_cached_predictions_for_dataset(
            args.model_location, 'ImageNet2p', NUM_MODELS
        )

     
        best_w, best_loss, loss_history = optimize_ce_logits_pgd(
            val_logits_np,
            val_targets_np,
            seed=42,
            maxiter=300,   
            lr=0.01
        )
            
     
        val_top1 = eval_ensemble_accuracy_from_logits(
            val_logits_np, best_w, val_targets_np, use_probs=False
        )
        print(f"[opt-ensemble] ImageNet2p val top1 = {val_top1:.4f}, loss = {best_loss:.6f}")



      
        plt.figure(figsize=(6, 4))
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Opt-ensemble PGD Loss Curve (ImageNet2p)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("opt_ensemble_loss_curve.png", bbox_inches="tight")
        print("Saved opt-ensemble loss curve to opt_ensemble_loss_curve.png")

        results = {
            'model_name': 'opt_ensemble',
            'weights': best_w.tolist(),
            'val_ImageNet2p_top1': val_top1,
            'val_ImageNet2p_loss': best_loss,
        }

      
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA]:
            name = dataset_cls.__name__
            print(f'[opt-ensemble] Evaluating on {name}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            preds = build_weighted_predictions_from_cache(
                args.model_location, name, best_w
            )
            acc = test_cached_predictions_on_dataset(preds, dataset)
            results[name] = acc
            print(f'[opt-ensemble] {name} acc = {acc:.4f}')

        with open(OPT_ENSEMBLE_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    
    if args.plot:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_db['OOD'] = 1. / 4 * (
            individual_model_db['ImageNetV2'] +
            individual_model_db['ImageNetR'] +
            individual_model_db['ImageNetSketch'] +
            individual_model_db['ImageNetA']
        )
        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db['OOD'] = 1. / 4 * (
            uniform_soup_db['ImageNetV2'] +
            uniform_soup_db['ImageNetR'] +
            uniform_soup_db['ImageNetSketch'] +
            uniform_soup_db['ImageNetA']
        )
        greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
        greedy_soup_db['OOD'] = 1. / 4 * (
            greedy_soup_db['ImageNetV2'] +
            greedy_soup_db['ImageNetR'] +
            greedy_soup_db['ImageNetSketch'] +
            greedy_soup_db['ImageNetA']
        )
        ensemble_db = pd.read_json(ENSEMBLE_RESULTS_FILE, lines=True)
        ensemble_db['OOD'] = 1. / 4 * (
            ensemble_db['ImageNetV2'] +
            ensemble_db['ImageNetR'] +
            ensemble_db['ImageNetSketch'] +
            ensemble_db['ImageNetA']
        )

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            greedy_soup_db['ImageNet'],
            greedy_soup_db['OOD'],
            marker='*',
            color='C4',
            s=400,
            label='Greedy Soup',
            zorder=10
        )
        ax.scatter(
            uniform_soup_db['ImageNet'],
            uniform_soup_db['OOD'],
            marker='o',
            color='C0',
            s=200,
            label='Uniform Soup',
            zorder=10
        )
        ax.scatter(
            ensemble_db['ImageNet'],
            ensemble_db['OOD'],
            marker='s',
            color='C3',
            s=90,
            label='Ensemble (more compute)',
            zorder=10
        )
        ax.scatter(
            individual_model_db['ImageNet'].values[0],
            individual_model_db['OOD'].values[0],
            marker='h',
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=10
        )
        ax.scatter(
            individual_model_db['ImageNet'].values[1:],
            individual_model_db['OOD'].values[1:],
            marker='d',
            color='C2',
            s=130,
            label='Various hyperparameters',
            zorder=10
        )

        ax.set_ylabel('Avg. accuracy on 4 distribution shifts', fontsize=16)
        ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig('figure.png', bbox_inches='tight')
        print("Saved plot to figure.png")

