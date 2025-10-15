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
    training_folder_ = Path(training_folder)
    for file in training_folder_.iterdir():
        if file.is_file() and file.suffix == '.arrow':
            datasets.append(file)
    # For Tracking which HPs are "just" datasets and which are for algorithm tuning
    training_data_paths = str([str(corpus.resolve()) for corpus in datasets])

    # Fixed parameters to be added as constants
    fixed_config = {
        "training_data_paths": training_data_paths,
        "max_per_device_train_batch_size": max_batch_size,
        #"min_past": 60,
        "save_steps": 200_000,
        "log_steps": 500,
        "num_samples": 20,
        "shuffle_buffer_length": 100_000,
        #"model_id": model_id,
        #"model_type": "seq2seq",
        "fp16":True,
        "random_init": True,
        "tf32": True,
        "torch_compile": True,
        "dataloader_num_workers": 0,
        "use_eos_token": True,
        "limit_model_size": limit_model_size,
        "prediction_length_expo": 6,
        #"bolt": True,
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

    cs.add(Categorical("model_id", model_ids, default=model_ids[0]))
    cs.add(Float("learning_rate", (0.00005,  0.01), log = True, default=0.001))
    cs.add(Float("warmup_ratio",(1e-7, 0.1), log = True, default=1e-7))
    cs.add(Float("dropout_rate", (1e-9, 0.2), log = True, default=1e-7)) 
    cs.add(Categorical("feed_forward_proj", ["relu", "gated-gelu", "gated-relu"], default="relu"))
    cs.add(Categorical("optim", ["adamw_torch_fused", "adafactor"], default="adamw_torch_fused"))
    cs.add(Float("layer_norm_epsilon", (1e-07, 1e-03), log = True, default=1e-6))
    cs.add(Integer("d_model_expo", (6, max_d_model_expo), log = False, default=def_d_model_expo))
    cs.add(Integer("num_layers", (1, max_num_layers), log = False, default=def_num_layers))
    cs.add(Integer("num_decoder_layers", (1, max_num_decoder_layers), log = False, default=def_num_decoder_layers))
    cs.add(Integer("num_heads", (1, max_num_heads), log = False, default=def_num_heads))
    cs.add(Integer("d_kv_expo", (3, max_d_kv_expo), log = False, default=def_d_kv_expo))
    cs.add(Integer("d_ff_expo", (5, max_d_ff_expo), log = False, default=def_d_ff_expo))
    cs.add(Integer("is_gated_act", (0, 1), default=0))
    cs.add(Integer("context_length_expo", (4, 14), log = False, default=9))
    # cs.add(Integer("prediction_length_expo", (3, 7), log = False, default=6))
    cs.add(Integer("batch_size_expo", (1, 11), log = False, default=5))
    cs.add(Float("max_missing_prop", (0.8, 1.0), log = True, default=0.9))
    cs.add(Float("drop_prob", (0.0, 0.5), log = False, default=0.2))
    #cs.add(Categorical("tokenizer_class", ["MeanScaleUniformBins", "MeanScaleQuantileBins"], default="MeanScaleUniformBins"))
    cs.add(Categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant","constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr"],default="linear"))
    cs.add(Integer("bolt", (0, 1),default=0))
    cs.add(Integer("min_past_expo", (4, 10),default=6))

    # Base Chronos Only
    cs.add(Float("tokenizer_limit", (5.0, 50), log=False, default=15))
    cs.add(Integer("n_tokens_expo", (8, 13), log=False, default=12))
    cs.add(EqualsCondition(cs["tokenizer_limit"], cs["bolt"], 0))
    cs.add(EqualsCondition(cs["n_tokens_expo"], cs["bolt"], 0))

    # Bolt Only Stuff
    cs.add(Integer("patch_size_expo", (0, 6), log = False, default=4))
    cs.add(Integer("patch_stride_expo", (0, 7), log = False, default=4)) 
    cs.add(Integer("use_reg_token", (0, 1),default=1))
    cs.add(EqualsCondition(cs["patch_size_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_stride_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["use_reg_token"], cs["bolt"], 1))
    # TODO Add Patch use_layer_norm HP 

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
