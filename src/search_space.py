from ConfigSpace import ConfigurationSpace, Constant, Integer, Float, Categorical, EqualsCondition
from ast import literal_eval
from pathlib import Path
from transformers import AutoConfig
import math

def ceil_log2(x: int) -> int:
    return math.ceil(math.log2(x))

# TODO if limit_model_size then adjust max layercount and co depending on model id
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
    max_num_heads = 12
    max_d_kv_expo = 10
    max_d_ff_expo = 12
    def_num_layers = 6
    def_num_heads = 8
    def_d_kv_expo = 6
    def_d_ff_expo = 11

    if limit_model_size:
        nlayers = []
        nheads = []
        e_dkv = []
        e_dff = []
        for model_id in model_ids:
            base_config = AutoConfig.from_pretrained(model_id)
            nlayers.append(base_config.num_layers)
            nheads.append(base_config.num_heads)
            e_dkv.append(ceil_log2(base_config.d_kv))
            e_dff.append(ceil_log2(base_config.d_ff))
        max_num_layers = max(nlayers) + 2
        max_num_heads = max(nheads) + 2
        max_d_kv_expo = max(e_dkv) + 2
        max_d_ff_expo = max(e_dff) + 2
        def_num_layers = max(nlayers) 
        def_num_heads = max(nheads) 
        def_d_kv_expo = max(e_dkv) 
        def_d_ff_expo = max(e_dff) 

    cs.add(Categorical("model_id", model_ids, default=model_ids[0]))
    cs.add(Float("learning_rate", (0.00005,  0.01), log = True, default=0.001))
    cs.add(Float("warmup_ratio",(1e-7, 0.1), log = True, default=1e-7))
    cs.add(Float("dropout_rate", (1e-9, 0.2), log = True, default=1e-7)) 
    cs.add(Categorical("feed_forward_proj", ["relu", "gated-gelu", "gated-relu"], default="relu"))
    cs.add(Categorical("optim", ["adamw_torch_fused", "adafactor"], default="adamw_torch_fused"))
    cs.add(Float("layer_norm_epsilon", (1e-07, 1e-03), log = True, default=1e-6))
    cs.add(Integer("d_model_expo", (6, 12), log = False, default=9))
    cs.add(Integer("num_layers", (1, max_num_layers), log = False, default=def_num_layers))
    cs.add(Integer("num_heads", (1, max_num_heads), log = False, default=def_num_heads))
    cs.add(Integer("d_kv_expo", (3, max_d_kv_expo), log = False, default=def_d_kv_expo))
    cs.add(Integer("d_ff_expo", (5, max_d_ff_expo), log = False, default=def_d_ff_expo))
    cs.add(Integer("context_length_expo", (4, 14), log = False, default=9))
    cs.add(Integer("prediction_length_expo", (3, 7), log = False, default=6))
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


    return cs


if __name__ == "__main__":
    cs = get_config_space()
    print(cs)
