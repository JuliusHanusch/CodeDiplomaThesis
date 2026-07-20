from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# IMPORT CHRONOS
# ============================================================

root_dir = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir/"src").resolve()))
sys.path.append(str((root_dir/"chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline



# ============================================================
# DATA
# ============================================================

def load_dataset(path):

    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        dtype=np.float32
    )

    values = data[:,0]
    labels = data[:,1].astype(int)

    return values, labels



# ============================================================
# WINDOWING
# ============================================================

def sliding_window(
        arr,
        window_size,
        stride
):

    for start in range(
        0,
        len(arr)-window_size+1,
        stride
    ):
        yield start, arr[start:start+window_size]





def compute_window_score(
        model,
        tokenizer,
        window,
        device,
        span_length=16,
        repetitions=20
):

    values = torch.tensor(
        window,
        dtype=torch.float32
    )


    input_ids, attention_mask, _ = tokenizer.context_input_transform(
        values
    )
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    mask_token_id = getattr(
        model.config,
        "mask_token_id",
        None
    )

    if mask_token_id is None:
        raise ValueError(
            "Missing mask_token_id in model config"
        )


    seq_len = input_ids.shape[1]


    scores = torch.zeros(
        1,
        seq_len,
        device=device
    )
    counts = torch.zeros(
        1,
        seq_len,
        device=device
    )


    for _ in range(repetitions):

        masked_input = input_ids.clone()
        labels = input_ids.clone()
        start = np.random.randint(
            0,
            seq_len - span_length + 1
        )

        end = start + span_length


        mask = torch.zeros(
            input_ids.shape,
            dtype=torch.bool,
            device=device
        )

        mask[:, start:end] = True
        masked_input[mask] = mask_token_id

        with torch.no_grad():
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        span_logits = logits[:, start:end, :]
        span_labels = labels[:, start:end]


        loss = F.cross_entropy(
            span_logits.reshape(
                -1,
                span_logits.size(-1)
            ),
            span_labels.reshape(-1),
            reduction="none"
        )

        loss = loss.reshape(span_labels.shape)

        scores[:, start:end] += loss
        counts[:, start:end] += 1

    scores /= counts.clamp(min=1)

    return scores.squeeze(0).cpu().numpy()



def evaluate(model, tokenizer, dataset, window_size=512, stride=128, device="cuda"):

    series, labels = load_dataset(dataset)

    scores = np.zeros(len(series))
    counts = np.zeros(len(series))

    for start, window in sliding_window(series, window_size, stride):

        token_scores = compute_window_score(model, tokenizer, window, device)

        length = min(len(token_scores), window_size)

        scores[start:start+length] += token_scores[:length]
        counts[start:start+length] += 1

    counts[counts == 0] = 1
    scores /= counts

    auprc = average_precision_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    idx = np.argmax(f1_scores)

    threshold = thresholds[idx] if idx < len(thresholds) else scores.mean()

    preds = (scores >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "scores": scores,
    }


if __name__=="__main__":


    model_path = (
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/FineTunedModels/Anomaly/run-5/checkpoint-final"
    )


    dataset = (
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/data/finetuning/Anomaly/"
        "ucr_processed/1sddb40_test.out"
    )


    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="mlm"
    )


    model = pipeline.model
    tokenizer = pipeline.tokenizer


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model.to(device)
    model.eval()



    result = evaluate(
        model,
        tokenizer,
        dataset,
        window_size=512,
        stride=128,
        device=device
    )


    print("\n===================")
    print("RESULTS")
    print("===================")


    for k,v in result.items():

        if k!="scores":
            print(
                f"{k}: {v}"
            )