from ConfigSpace import ConfigurationSpace, Constant, Integer, Float, Categorical, EqualsCondition
from ast import literal_eval
from pathlib import Path
from transformers import AutoConfig
import math
from ConfigSpace.forbidden import ForbiddenRelation
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter
from ConfigSpace.types import Array, Mask, f64
import numpy as np
from src.utils import get_expected_model_size, estimate_transformer_size
from typing import Any, ClassVar, Mapping,  TYPE_CHECKING
from typing_extensions import Self

import ConfigSpace.read_and_write.dictionary as cs_registry

_SENTINEL = object()

def ceil_log2(x: int) -> int:
    return math.ceil(math.log2(x))


class ForbidTooBig(ForbiddenRelation):
    _RELATION_STR = "TOO_BIG"

    hyperparameters: dict[str, Hyperparameter]    

    vector_ids: tuple[None, None] | tuple[np.intp, np.intp]

    def __init__(
            self, 
            d_model_expo: Hyperparameter,
            d_ff_expo: Hyperparameter,
            num_heads: Hyperparameter,
            num_layers: Hyperparameter,
            num_decoder_layers: Hyperparameter,
            d_kv_expo: Hyperparameter,
            bolt: Hyperparameter,
            patch_size_expo: Hyperparameter,
            # feed_forward_proj: Hyperparameter,
            # context_length_expo: Hyperparameter,
            is_gated_act: Hyperparameter,
            model_id: CategoricalHyperparameter,
            ):

        self.hyperparameters = {
            "d_model_expo": d_model_expo,
            "d_ff_expo": d_ff_expo,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "num_decoder_layers": num_decoder_layers,
            "d_kv_expo": d_kv_expo,
            "is_gated_act": is_gated_act,
            "bolt": bolt,
            "patch_size_expo": patch_size_expo,
            # "feed_forward_proj": feed_forward_proj,
            # "context_length_expo": context_length_expo,
            "model_id": model_id,
        }

        self.model_sizes = {mid: get_expected_model_size(mid) for mid in model_id.choices}

        self.vector_ids: tuple[None] | tuple[np.intp] = (None, None, None, None, None, None, None)

        # Artifacts from Forbidden Relation 
        self.left = d_model_expo
        self.right = num_heads


    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if any([value != other.hyperparameters[key] for key, value in self.hyperparameters.items()]):
            return False
        return True
    

    def __copy__(self) -> Self:
        return self.__class__(**self.hyperparameters)
    
    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, int]) -> None:
        """Set the vector index of the hyperparameters."""
        self.vector_ids = tuple(
            np.intp(hyperparameter_to_idx[hp.name]) for hp in self.hyperparameters.values()
        )
        self.vector_id_dict = {
            key: np.intp(hyperparameter_to_idx[hp.name]) for key, hp in self.hyperparameters.items()
        }

    def __repr__(self) -> str:
        return f"Forbidden Model Size"
    

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        """Check if a value is forbidden."""
        raise Exception(f"This method should never get used??? {values}") # TODO Check if "Any" means values or internal floats
        estimated_size = estimate_transformer_size(**{key: values.get(hp.name, _SENTINEL) for key, hp in self.hyperparameters.items()})
        mid = values.get(self.hyperparameters["model_id"].name, _SENTINEL)
        if estimated_size > self.model_sizes[mid] * 1.1:
            return True # It is truly forbidden
        return False # It's probably ok


    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        """Check if a vector is forbidden."""
        parameters = {key: hp.to_value(vector[self.vector_id_dict[key]]) for key, hp in self.hyperparameters.items()}
        estimated_size = estimate_transformer_size(**parameters)
        mid = parameters["model_id"]

        if estimated_size > self.model_sizes[mid] * 1.1:
            return True
        return False
    

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        """Check if a vector is forbidden."""
        # convert arr in config space to real world parameter values
        parameter_vectors = {key: hp.to_value(arr[self.vector_id_dict[key]]) for key, hp in self.hyperparameters.items()}
        mask = []
        for i in range(arr.shape[1]):
            params = {key: vals[i] for key, vals in parameter_vectors.items()}
            estimated_size = estimate_transformer_size(**params)
            mid = parameter_vectors["model_id"][i]

            if estimated_size > self.model_sizes[mid] * 1.1:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    

    def to_dict(self) -> dict[str, Any]:
        """Convert the forbidden relation to a dictionary representation."""
        return {
            **{hp_name: hp.name for hp_name, hp in self.hyperparameters.items()},
            "type": "TOO_BIG",
            "lambda": self._RELATION_STR,
        }

def _decode_forbidden_model_too_big(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode,  # noqa: ARG001
) -> ForbidTooBig:
    return ForbidTooBig(
        d_model_expo=cs[item["d_model_expo"]],
        d_ff_expo=cs[item["d_ff_expo"]],
        num_heads=cs[item["num_heads"]],
        num_layers=cs[item["num_layers"]],
        num_decoder_layers=cs[item["num_decoder_layers"]],
        d_kv_expo=cs[item["d_kv_expo"]],
        bolt=cs[item["bolt"]],
        patch_size_expo=cs[item["patch_size_expo"]],
        is_gated_act=cs[item["is_gated_act"]],
        # feed_forward_proj=cs[item["feed_forward_proj"]],
        # context_length_expo=cs[item["context_length_expo"]],
        model_id=cs[item["model_id"]],
        )

def _encode_forbidden_model_too_big(
    cond: ForbidTooBig,
    encode,
) -> dict[str, Any]:
    encoding = cond.to_dict()
    encoding.pop("type")
    encoding.pop("lambda")
    return encoding

def get_config_space(training_folder: str, model_ids: str = '["google/t5-efficient-tiny"]', max_batch_size=32, limit_model_size=1) -> ConfigurationSpace:
    # Find all Corpora in training_folder
    datasets = []
    training_folder_ = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/train/")
    for file in training_folder_.iterdir():
       if file.is_file() and file.suffix == '.arrow':
            datasets.append(file)
    # For Tracking which HPs are "just" datasets and which are for algorithm tuning
    training_data_paths = str([str(corpus.resolve()) for corpus in datasets])
    probability = str([(0.9), (0.1)])


    # Fixed parameters to be added as constants
    fixed_config = {
        "training_data_paths": training_data_paths,
        "probability": probability,
        "max_per_device_train_batch_size": max_batch_size,
        "save_steps": 200_000,
        "log_steps": 500,
        "num_samples": 20,
        "shuffle_buffer_length": 100_000,
        "model_id": "prajjwal1/bert-small",#"huggingface/CodeBERTa-small-v1",#FacebookAI/roberta-base",
        "model_type": "mlm",
        "task": "mlm",
        "fp16":True,
        "random_init": True,
        "tf32": True,
        "torch_compile": True,
        "dataloader_num_workers": 0,
        "use_eos_token": True,
        "limit_model_size": limit_model_size,
        "prediction_length_expo": 6,
        "n_special_tokens": 4,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 3,
        "bolt": 1,    #TODO True or False
        "context_length": 512,

        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-12,

        "use_reg_token": True,
        "span_masking": True,
        "n_tokens": 4096, # Bert vs Chronos? 4096
        "tokenizer_limit": 15,
        "debug_patching": True,
        "debug_batch_idx": 0,
        "debug_max_patches": 3,
        "tokenizer_class": "MeanScaleUniformBins",


        #set to default fixed config
        "batch_size": 32,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.01,
        "optim": "adamw_torch_fused",
        "max_missing_prop": 0.9,
        "drop_prob": 0.2,
        "lr_scheduler_type": "linear",
        "min_past": 60,
        "mean_span_length": 20,
        "masking_prob": 0.25,


    }


    cs = ConfigurationSpace()

    # Add fixed parameters as constants if compatible
    for key, value in fixed_config.items():
        if isinstance(value, (int, float, str, bool)):
            cs.add(Constant(key, value))
    
    # Add A variable for every dataset
    for ds_path in datasets:
        ds_name =  ds_path.stem
        cs.add(Integer(ds_name, (0, 1), default=1)) # Binary Decision: include or not

    # Add tunable hyperparameters
    model_ids = literal_eval(model_ids)

    # Adjust Search Space when limit_model_size is true
    max_num_layers = 12
    max_num_decoder_layers = 12
    max_num_heads = 12
    max_d_kv_expo = 10
    max_d_ff_expo = 12
    max_d_model_expo = 12
    def_num_layers = 6
    def_num_decoder_layers = 6
    def_num_heads = 8
    def_d_kv_expo = 6
    def_d_ff_expo = 11
    def_d_model_expo = 9

    #Bert Default Values:
    def_hidden_size = 768
    def_num_hidden_layers = 12
    def_num_attention_heads = 12
    def_intermediate_size = 3072
    def_hidden_act = "gelu"
    def_hidden_dropout_prob = 0.1
    def_attention_probs_dropout_prob = 0.1
    def_layer_norm_eps = 1e-12
    def_span_masking = False
    def_mean_span_length = 3

    def_individual_dropout=0.05
    def_mask_prob=0.15
    def_perturbation_prob=0.05  
    def_perturbation_strength=0.1 
    def_unchanged_patch_prob=0.05 
    def_patch_perturbation_prob=0.1
    def_patch_perturbation_scale_strength=0.1
    def_patch_perturbation_noise_strength=0.1 

    #BERT Max Values - def * 2
    max_hidden_size = 1536
    max_num_hidden_layers = 24,
    max_num_attention_heads = 24
    max_intermediate_size = 6144
    max_mean_span_length = 64,
    if limit_model_size:
        nlayers = []
        ndeclayers = []
        nheads = []
        e_dkv = []
        e_dff = []
        e_dmodel = []
        for model_id in model_ids:
            base_config = AutoConfig.from_pretrained(model_id)
            nlayers.append(base_config.num_layers)
            ndeclayers.append(base_config.num_decoder_layers)
            nheads.append(base_config.num_heads)
            e_dkv.append(ceil_log2(base_config.d_kv))
            e_dff.append(ceil_log2(base_config.d_ff))
            e_dmodel.append(ceil_log2(base_config.d_model))
        max_num_layers = max(nlayers) + 2
        max_num_decoder_layers = max(ndeclayers) + 2
        max_num_heads = max(nheads) + 2
        max_d_kv_expo = max(e_dkv) + 2
        max_d_ff_expo = max(e_dff) + 2
        max_d_model_expo = max(e_dmodel) + 2
        def_num_layers = nlayers[0]
        def_num_decoder_layers = ndeclayers[0]
        def_num_heads = nheads[0] 
        def_d_kv_expo = e_dkv[0] 
        def_d_ff_expo = e_dff[0] 
        def_d_model_expo = e_dmodel[0] 

    #cs.add(Integer("bolt", (0, 1),default=0))

    # Search Space
    # cs.add(Float("learning_rate", (0.00005,  0.01), log = True, default=0.001))
    # cs.add(Float("warmup_ratio",(1e-7, 0.1), log = True, default=1e-7))
    # cs.add(Categorical("optim", ["adamw_torch_fused", "adafactor"], default="adamw_torch_fused"))
    # cs.add(Integer("batch_size_expo", (1, 11), log = False, default=5))
    # cs.add(Float("max_missing_prop", (0.8, 1.0), log = True, default=0.9))
    # cs.add(Float("drop_prob", (0.0, 0.5), log = False, default=0.2))
    # cs.add(Categorical("lr_scheduler_type", ["linear", "cosine"],default="linear")) # "cosine_with_restarts", "polynomial", "constant","constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr
    # cs.add(Integer("min_past_expo", (4, 10),default=6)) # to default
    
    # #SpanMasking Params
    #cs.add(Integer("mean_span_length", (1, max_mean_span_length), default=def_mean_span_length))
    # cs.add(Float("masking_prob", (0.1, 0.3), log = False, default=def_mask_prob))


    #cs.add(Categorical("model_id", model_ids, default=model_ids[0]))
    #cs.add(Integer("is_gated_act", (0, 1), default=0))
    #cs.add(Integer("context_length_expo", (4, 14), log = False, default=9))
    #cs.add(Integer("prediction_length_expo", (3, 7), log = False, default=6))
    #cs.add(Categorical("tokenizer_class", ["MeanScaleUniformBins", "MeanScaleQuantileBins"], default="MeanScaleUniformBins"))


    #BERT Parameters
    #cs.add(Integer("hidden_size", (384, max_hidden_size), log = False, default=def_hidden_size))
    #cs.add(Integer("num_hidden_layers", (1, max_num_hidden_layers), log = False, default=def_num_hidden_layers))
    #cs.add(Float("hidden_dropout_prob", (1e-9, 0.2), log = True, default=def_hidden_dropout_prob)) 
    #cs.add(Categorical("hidden_act", ["gelu", "relu", "gelu_new", "silu"], default=def_hidden_act))
    #cs.add(Float("layer_norm_eps", (1e-12, 1e-03), log = True, default=def_layer_norm_eps))
    #cs.add(Integer("num_attention_heads", (1, max_num_attention_heads), log = False, default=def_num_attention_heads))
    #cs.add(Integer("intermediate_size", (1536, max_intermediate_size), log = False, default=def_intermediate_size))
    #cs.add(Float("attention_probs_dropout_prob", (0.01, 0.2), log = False, default=def_attention_probs_dropout_prob))



    #Patching Paramas
    cs.add(Integer("patch_size_expo", (0, 6), log = False, default=4))
    cs.add(Integer("patch_stride_expo", (0, 7), log = False, default=4)) 
    cs.add(Float("individual_dropout", (0.025, 0.1), log = True, default=def_individual_dropout))
    cs.add(Float("perturbation_prob", (0.025, 0.1), log = True, default=def_perturbation_prob))
    cs.add(Float("perturbation_strength", (0.05, 0.2), log = True, default=def_perturbation_strength))
    cs.add(Float("unchanged_patch_prob", (0.025, 0.1), log = True, default=def_unchanged_patch_prob))
    cs.add(Float("patch_perturbation_prob", (0.05, 0.2), log = True, default=def_patch_perturbation_prob))
    cs.add(Float("patch_perturbation_scale_strength", (0.05, 0.2), log = True, default=def_patch_perturbation_scale_strength))
    cs.add(Float("patch_perturbation_noise_strength", (0.05, 0.2), log = True, default=def_patch_perturbation_noise_strength))
    # Add if  Bolt = 1
    cs.add(EqualsCondition(cs["patch_size_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_stride_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["individual_dropout"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["perturbation_prob"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["perturbation_strength"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["unchanged_patch_prob"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_perturbation_prob"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_perturbation_scale_strength"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_perturbation_noise_strength"], cs["bolt"], 1))


    if limit_model_size == 2:
        forbid_large = ForbidTooBig(
            d_model_expo=cs["d_model_expo"],
            d_ff_expo=cs["d_ff_expo"],
            num_heads=cs["num_heads"],
            num_layers=cs["num_layers"],
            num_decoder_layers=cs["num_decoder_layers"],
            d_kv_expo=cs["d_kv_expo"],
            bolt=cs["bolt"],
            patch_size_expo=cs["patch_size_expo"],
            is_gated_act=cs["is_gated_act"],
            # feed_forward_proj=cs["feed_forward_proj"],
            # context_length_expo=cs["context_length_expo"],
            model_id=cs["model_id"],
        )
        cs.add(forbid_large)

        # Register an Encoder and Decoder for our clause
        cs_registry.FORBIDDEN_ENCODERS[ForbidTooBig] = ("TOO_BIG", _encode_forbidden_model_too_big)
        cs_registry.FORBIDDEN_DECODERS["TOO_BIG"] = _decode_forbidden_model_too_big

    return cs


if __name__ == "__main__":
    cs = get_config_space()
    print(cs)
