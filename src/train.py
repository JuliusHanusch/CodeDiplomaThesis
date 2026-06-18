# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import logging
import os
import re

# Include Parent Directory to load packages from
import sys  
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

import json
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

import typer

from typer_config import use_yaml_config

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    BertConfig,
    Trainer,
    TrainingArguments,  
    AutoModelForMaskedLM,
)

import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)
from src.utils import make_dict_storable, get_expected_model_size, get_model_size, ModelTooBig


from chronos_pkg.src.chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


app = typer.Typer(pretty_exceptions_enable=False)


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    training_config = make_dict_storable(training_config)
    print(training_config)
    print(ckpt_path / "training_info.json")
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + 1
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


#modified load model fúnction to implement additional hyperparameter
def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id = 0,
    bos_token_id = 1,
    eos_token_id = 2,
    mask_token_id = 3, 
    config_overrides=None,
    hidden_size = 768,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    intermediate_size = 3072,
    hidden_act = "gelu",
    hidden_dropout_prob = 0.1,
    attention_probs_dropout_prob = 0.1,
    layer_norm_eps = 1e-12,
    bolt = False,
    is_gated_act= False,
    task="mlm",
    num_labels=6,
    context_length = 512,

):
    """
    Load a HuggingFace model, adjusting vocab size, token IDs, and config overrides.

    Compatible with both pretrained and randomly initialized models.
    """

    

    assert model_type in ["seq2seq", "causal", "mlm"]

    if model_type == "mlm":
        AutoModelClass = AutoModelForMaskedLM
    elif model_type == "seq2seq":
        AutoModelClass = AutoModelForSeq2SeqLM
    elif model_type == "causal":
        AutoModelClass = AutoModelForCausalLM
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    

    # Load from scratch
    if random_init:
        log_on_main("Using random initialization", logger)
        config = AutoConfig.from_pretrained(model_id)
        
        '''
        Params are not set in search space. For optimal results use defautl values from Auto Config.
        '''
        # config.hidden_size = hidden_size
        # config.num_hidden_layers = num_hidden_layers
        # config.num_attention_heads = num_attention_heads
        # config.intermediate_size = intermediate_size
        # config.hidden_dropout_prob = hidden_dropout_prob
        # config.hidden_act = hidden_act
        # config.attention_probs_dropout_prob = attention_probs_dropout_prob
        # config.layer_norm_eps = layer_norm_eps
        # config.hidden_act = hidden_act
        # config.is_gated_act = is_gated_act
        #config.max_position_embeddings = context_length


        print("Config:", print(type(config).__name__))

        if isinstance(config, T5Config):
            config.initializer_factor = 0.05
            print("Config: Confirmed T5")
        if isinstance(config, BertConfig):
            config.initializer_range = 0.05
            print("Config: Confirmed BERT")

        config.tie_word_embeddings = tie_embeddings

        if config_overrides:
            log_on_main(f"Overriding config: {config_overrides}", logger)
            config.update_from_string(config_overrides)

        if bolt:
            return config
        
        if task == "classification":
            from chronos_pkg.src.chronos.chronos_classification import ChronosModelForClassification


            config = AutoConfig.from_pretrained(model_id)
            base_model = AutoModelForMaskedLM.from_config(config)

            model = ChronosModelForClassification(
                config=ChronosConfig(model_type="mlm"),
                model=base_model,
                num_labels=num_labels,
            )
        elif task == "anomaly":
            from chronos_pkg.src.chronos.chronos_anomaly import ChronosModelForAnomalyDetection
       
            model = ChronosModelForAnomalyDetection(
                config=ChronosConfig(model_type="mlm"),
                model=base_model,
            )
        else:
            model = AutoModelClass.from_config(config)

    # Load from pretrained
    else:
        log_on_main(f"Using pretrained initialization from {model_id}", logger)
        if config_overrides:
            raise ValueError("--config_overrides cannot be used with pretrained models")

        if task == "classification":
            log_on_main("Loading classification model via ChronosPipeline", logger)

            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                task="classification",
                num_labels=num_labels
            )
            model = pipeline.model

        if task == "anomaly":
            log_on_main("Loading anomaly model via ChronosPipeline", logger)

            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                task="anomaly",
            )
            model = pipeline.model


        else:
            # ✅ Standard HF loading for MLM / seq2seq / causal
            model = AutoModelClass.from_pretrained(model_id)


    # Resize token embeddings to match vocab size
    target_model = model.model if hasattr(model, "model") else model

    if hasattr(target_model, "resize_token_embeddings"):
        target_model.resize_token_embeddings(vocab_size)


    # Set special token IDs (on the HF backbone!)
    hf_config = target_model.config
    hf_config.pad_token_id = pad_token_id
    hf_config.eos_token_id = eos_token_id
    hf_config.mask_token_id = mask_token_id
    hf_config.bos_token_id = bos_token_id


    return model


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
        span_masking: bool = False,
        mean_span_length: int = 3,
        masking_prob: float = 0.15,
        task: str = "mlm",


    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal", "mlm")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype
        self.span_masking = span_masking
        self.mean_span_length = mean_span_length
        self.masking_prob = masking_prob
        self.task = task

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        #logger.info(f"RAW ENTRY KEYS: {list(entry.keys())}")

        # CLASSIFICATION PATH
        if self.task == "classification":
            assert "label" in entry, f"Missing label. Keys: {list(entry.keys())}"

            return {
                "target": np.asarray(entry["target"], dtype=self.np_dtype),
                "label": int(entry["label"]),
            }
        elif self.task == "anomaly":
            assert "anomaly_mask" in entry, f"Missing timestep mask. Keys: {list(entry.keys())}"

            target = np.asarray(entry["target"], dtype=self.np_dtype)
            label = np.asarray(entry["anomaly_mask"], dtype=np.float32)  # (T,) binary mask

            # safety checks (important for debugging shape bugs later)
            assert target.shape[0] == label.shape[0], (
                f"Target/label length mismatch: {target.shape} vs {label.shape}"
            )

        return {
            "target": target,
            "label": label,   # (T,) float mask for BCEWithLogitsLoss
        }
        # FORECASTING / MLM PATH
        entry = {
            "start": entry["start"],
            "target": np.asarray(entry["target"], dtype=self.np_dtype),
        }

        assert entry["target"].ndim == 1

        # causal handling
        if self.model_type == "causal":
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice([True, False], size=len(target), p=[drop_p, 1 - drop_p])
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        if self.task == "classification":
            return data
        elif self.task == "anomaly":
            return data
        # forecasting path (unchanged)
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        if self.task == "mlm":
            return self._to_mlm(entry)
        elif self.task == "classification":
            return self._to_classification(entry)
        elif self.task == "anomaly":
            return self._to_anomaly(entry)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
    
    def _to_classification(self, entry: dict) -> dict:
        target = np.asarray(entry["target"], dtype=np.float32)

        context = torch.tensor(target[-self.context_length:]).unsqueeze(0)

        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(context)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": entry["label"],
        }
    
    def _to_anomaly(self, entry: dict) -> dict:
        target = np.asarray(entry["target"], dtype=np.float32)
        label = np.asarray(entry["anomaly_mask"], dtype=np.float32)

        context = torch.tensor(target[-self.context_length:]).unsqueeze(0)
        label = torch.tensor(label[-self.context_length:]).unsqueeze(0)

        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(context)

        print("context:", context.shape)
        print("labels :", label.shape)
        print("input_ids:", input_ids.shape)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": label.squeeze(0),   # (T,) binary mask
        }
        
        
    def _to_mlm(self, entry: dict) -> dict:
        max_len = self.context_length
        context = torch.concat([torch.tensor(entry["past_target"]), torch.tensor(entry["future_target"])])[-max_len:].unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            context
        )
        if False:
            pass
        
        elif self.model_type == "mlm":            
            # --- ensure mask token exists ---
            if not hasattr(self.tokenizer.config, "mask_token_id"):
                self.tokenizer.config.mask_token_id = self.tokenizer.config.n_special_tokens 
                self.tokenizer.config.n_special_tokens += 1
                self.tokenizer.config.n_tokens += 1
                print(f"Added <mask> token with id {self.tokenizer.config.mask_token_id}")

            labels = input_ids.clone()

            # --- span masking parameters ---
            masking_prob = self.masking_prob
            mask_token_id = self.tokenizer.config.mask_token_id
            mean_span_length = self.mean_span_length

            special_tokens_mask = input_ids < self.tokenizer.config.n_special_tokens 
            valid_positions = ~special_tokens_mask
            
            # --- compute number of tokens to mask ---
            n_tokens = valid_positions.sum().item()
            n_to_mask = int(masking_prob * n_tokens)
            mask = torch.zeros_like(input_ids, dtype=torch.bool)

            valid_indices = valid_positions.nonzero(as_tuple=True)[1]

            total_masked = 0

            while total_masked < n_to_mask:
                span_len = max(1, int(torch.poisson(torch.tensor(mean_span_length, dtype=torch.float32)).item()))
                if total_masked + span_len > n_to_mask:
                    span_len = n_to_mask - total_masked

                # pick random start
                start_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
                end_idx = min(start_idx + span_len, input_ids.size(1))

                if mask[0, start_idx:end_idx].any():
                    continue

                mask[0, start_idx:end_idx] = True
                total_masked += span_len
            # ensure we never mask special tokens
            mask &= valid_positions
            random_prob = torch.rand_like(input_ids.float())
            # 80% [MASK]
            input_ids[mask & (random_prob < 0.8)] = mask_token_id
        

        random_mask = mask & (random_prob >= 0.8) & (random_prob < 0.9)
        if random_mask.any():
            #print("random_mask", random_mask)
            indices = torch.nonzero(random_mask, as_tuple=False)
            for idx_pair in indices:
                idx = int(idx_pair[1].item())  # token index only
                low = max(0, idx - 10)
                high = min(input_ids.size(1) - 1, idx + 10)
                nearby_tokens = input_ids[0, low:high+1]
                if nearby_tokens.numel() > 0:
                    replacement = nearby_tokens[torch.randint(0, nearby_tokens.numel(), (1,), device=input_ids.device)]
                    input_ids[0, idx] = replacement

        # loss only computed on masked positions
        labels[~mask] = -100


        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        if self.task == "classification":
            preprocessed_datasets = self.datasets
        if self.task == "anomaly":
            preprocessed_datasets = self.datasets
        else:
            preprocessed_datasets = [
                Map(
                    partial(self.preprocess_entry, mode=self.mode),
                    dataset,
                )
                for dataset in self.datasets
            ]

        if self.mode == "training":
            if self.task == "classification":
                iterables = preprocessed_datasets
            elif self.task == "anomaly":
                iterables = preprocessed_datasets
            else:
                iterables = [
                    self.create_training_data(dataset) for dataset in preprocessed_datasets
                ]
        elif self.mode == "test":
            if self.task == "classification":
                iterables = preprocessed_datasets
            elif self.task == "anomaly":
                iterables = preprocessed_datasets
            else:
                iterables = [
                    self.create_test_data(dataset) for dataset in preprocessed_datasets
                ]
        else:
            if self.task == "classification":
                iterables = preprocessed_datasets
            if self.task == "anomaly":
                iterables = preprocessed_datasets
            else:
                iterables = [
                    self.create_validation_data(dataset)
                    for dataset in preprocessed_datasets
                ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    entry = next(iterators[idx])
                    yield self.to_hf_format(entry)

                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [p / sum(probs) for p in probs]
                else:
                    for entry in itertools.chain(*iterators):
                        yield self.to_hf_format(entry)

class BoltDataset(ChronosDataset):
    def to_hf_format(self, entry: dict) -> dict:
        return {
            "context": torch.tensor(entry["past_target"]),
            "target":  torch.tensor(entry["future_target"]),
        }
        

#modified main fúnction to implement additional hyperparameter
@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    drop_prob: float = 0.2,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 4,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    mask_token_id: int = 3,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    fp16: bool = False,
    bolt: bool = False,
    patch_size: int = 16,
    patch_stride: int = 16,
    use_reg_token: bool = True,
    limit_model_size: bool = True, # Breaks when model size > 110% of original model size defined by model_id - Goal: Not improve perf simply by scaling
    is_gated_act=False,
    config_overrides: Optional[str] = None,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    layer_norm_eps: float = 1e-12,
    span_masking: bool = False,
    mean_span_length: int = 3,
    masking_prob: float = 0.15,
    individual_dropout:float =0.05,
    perturbation_prob: float=0.05,
    perturbation_strength: float=0.1, 
    unchanged_patch_prob: float =0.05, 
    patch_perturbation_prob: float=0.1,
    patch_perturbation_scale_strength: float=0.1,
    patch_perturbation_noise_strength:float=0.1,
    debug_patching: bool = False,
    debug_batch_idx: int = 0,
    debug_max_patches: int = 3,
    task: str = "mlm",
    num_labels: int = 6,


    #T5 Parameter, not available in BERT
    #d_model: int = 512,
    #dropout_rate: float = 0.1,
    #feed_forward_proj: str = "relu",
    #layer_norm_epsilon: float = 1e-06,
    #is_encoder_decoder: bool = True,
    #num_layers: int = 6,
    #num_decoder_layers: int = 6,
    #num_heads: int = 8,
    #d_kv: int = 6,
    #d_ff: int = 2048,
):
    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)

    assert len(training_data_paths) == len(probability)

    if dataloader_num_workers > len(training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(training_data_paths)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(training_data_paths)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal", "mlm"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    print("model_id", model_id)


    model_or_config = load_model( # TODO Go find a cleaner solution instead of model_OR_config
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        bos_token_id = bos_token_id,
        eos_token_id=eos_token_id,
        mask_token_id= mask_token_id,
        config_overrides=config_overrides,
        bolt=bolt,
        is_gated_act=is_gated_act,
        hidden_size = hidden_size,
        num_hidden_layers = num_hidden_layers,
        num_attention_heads = num_attention_heads,
        intermediate_size = intermediate_size,
        hidden_act = hidden_act,
        hidden_dropout_prob = hidden_dropout_prob,
        attention_probs_dropout_prob =attention_probs_dropout_prob,
        layer_norm_eps = layer_norm_eps,
        task=task,
        num_labels = num_labels,
        context_length = context_length,
        

        #T5 Options
        #d_model=d_model,
        #dropout_rate = dropout_rate,
        #feed_forward_proj = feed_forward_proj,
        #layer_norm_epsilon = layer_norm_epsilon,
        #is_encoder_decoder = is_encoder_decoder,
        #num_layers = num_layers,
        #num_decoder_layers=num_decoder_layers,
        #num_heads = num_heads,
        #d_ff = d_ff,
        #d_kv = d_kv,

    )
    if bolt: # Then it is a config and we still need to load the model 
        chronos_bolt_config = ChronosBoltConfig(
            context_length=context_length,
            prediction_length=prediction_length,
            input_patch_size=patch_size,
            input_patch_stride=patch_stride, 
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_reg_token=use_reg_token,
            individual_dropout = individual_dropout,
            masking_prob = masking_prob,
            perturbation_prob = perturbation_prob,
            perturbation_strength = perturbation_strength,
            unchanged_patch_prob = unchanged_patch_prob,
            patch_perturbation_prob = patch_perturbation_prob,
            patch_perturbation_scale_strength = patch_perturbation_scale_strength,
            patch_perturbation_noise_strength = patch_perturbation_noise_strength,
            model_id=model_id,
            debug_patching = debug_patching,
            debug_batch_idx = debug_batch_idx,
            debug_max_patches=debug_max_patches,
            


        )
        model_or_config.chronos_config = chronos_bolt_config.__dict__

        print("model_or_config" )

        model = ChronosBoltModelForForecasting(model_or_config)
    else:
        chronos_config = ChronosConfig(
            tokenizer_class=tokenizer_class,
            tokenizer_kwargs=tokenizer_kwargs,
            n_tokens=n_tokens,
            n_special_tokens=n_special_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            mask_token_id = mask_token_id,
            bos_token_id = bos_token_id,
            use_eos_token=use_eos_token,
            model_type=model_type,
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        model = model_or_config
        # Add extra items to model config so that it's saved in the ckpt
        model.config.chronos_config = chronos_config.__dict__
    
    model_size = get_model_size(model)
    print("Number Non-Embedding Params: ", model_size)
    expected_model_size = get_expected_model_size(model_id=model_id)
    # if model_size > expected_model_size * 1.1 and limit_model_size:
    #     raise ModelTooBig(
    #         f"""ModelTooBig 
    #         The model may only be 10% larger than the config allows else it's not an instance of {model_id} anymore
    #         But: {model_size} >> {expected_model_size}
    #         """)

    if bolt:
        DatasetClass = BoltDataset
    else:
        DatasetClass = ChronosDataset

    shuffled_train_dataset = DatasetClass( 
        datasets=train_datasets,
        probabilities=probability, 
        tokenizer=None if bolt else chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
        drop_prob=drop_prob,
        span_masking = span_masking,
        mean_span_length = mean_span_length,
        masking_prob=masking_prob,
        task=task
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    print("Steps:", max_steps)

    # Define training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        fp16=fp16,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        save_safetensors=False,
    )
    if "min_lr" in lr_scheduler_type:
        training_args.lr_scheduler_kwargs = {
            "min_lr": learning_rate/10 # According to Hoffman best choice  
        }
    if "reduce_lr_on_plateau" == lr_scheduler_type:
        training_args.eval_strategy = "steps"


    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )
    log_on_main(f"Training with {trainer.args.n_gpu} GPU(s)", logger)
    print(f"Training with {trainer.args.n_gpu} GPU(s)")

    trainer.train()

    if is_main_process():
        save_path = output_dir / "checkpoint-final"

        if task == "classification":
            model.model.save_pretrained(save_path)
            torch.save(
                model.classifier.state_dict(),
                save_path / "classifier.pt"
            )
        if task == "anomaly":
            model.model.save_pretrained(save_path)
            torch.save(
                model.classifier.state_dict(),
                save_path / "anomaly.pt"
            )

        else:
            model.save_pretrained(save_path)

        save_training_info(
            save_path,
            training_config=raw_training_config
        )

        return save_path


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()