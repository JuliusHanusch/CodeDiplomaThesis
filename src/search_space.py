from ConfigSpace import ConfigurationSpace, Constant, Integer, Float, Categorical, EqualsCondition

# TODO Update
#training_data_paths = "['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']"
training_data_paths = "['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']"


probability_options = [
    "[0.8, 0.1, 0.1]",
    "[0.7, 0.2, 0.1]",
    "[0.6, 0.2, 0.2]",
    "[0.9, 0.05, 0.05]",
    "[0.85, 0.1, 0.05]"
]
# define data splits

def get_config_space(model_id = "google/t5-efficient-tiny", max_batch_size=32) -> ConfigurationSpace:
    # TODO include batch_size into search space
    # Fixed parameters to be added as constants
    fixed_config = {
        "training_data_paths": [ # TODO Update
            "your/path/to/training_mix.arrow",
            "your/path/tp/kernelsynth.arrow"
            "your/path/to/realdata.arrow"
        ],
        "max_per_device_train_batch_size": max_batch_size,
        "min_past": 60,
        "save_steps": 200_000,
        "log_steps": 500,
        "num_samples": 20,
        "shuffle_buffer_length": 100_000,
        "model_id": model_id,
        "model_type": "seq2seq",
        "fp16":True,
        "random_init": True,
        "tf32": True,
        "torch_compile": True,
        "dataloader_num_workers": 0,
        "use_eos_token": True,
        #"bolt": True,
    }


    cs = ConfigurationSpace()

    # Add fixed parameters as constants if compatible
    for key, value in fixed_config.items():
        if isinstance(value, (int, float, str, bool)):
            cs.add(Constant(key, value))

    # Add tunable hyperparameters
    cs.add(Float("learning_rate", (0.00005,  0.01), log = True, default=0.001))
    cs.add(Float("warmup_ratio",(1e-7, 0.1), log = True, default=1e-7))
    cs.add(Float("dropout_rate", (1e-7, 0.2), log = True, default=1e-7)) 
    cs.add(Categorical("feed_forward_proj", ["relu", "gated-gelu", "gated-relu"], default="relu"))
    cs.add(Categorical("optim", ["adamw_torch_fused", "adafactor"], default="adamw_torch_fused"))
    cs.add(Float("layer_norm_epsilon", (1e-07, 1e-03), log = True, default=1e-6))
    cs.add(Integer("d_model", (6, 12), log = False, default=9))
    cs.add(Integer("num_layers", (1, 12), log = False, default=6))
    cs.add(Integer("num_heads", (1, 12), log = False, default=8))
    cs.add(Integer("d_kv", (3, 10), log = False, default=6))
    cs.add(Integer("d_ff", (5, 12), log = False, default=11))
    cs.add(Integer("context_length", (4, 12), log = False, default=9))
    cs.add(Integer("prediction_length", (3, 7), log = False, default=6))
    cs.add(Integer("batch_size_expo", (1, 11), log = False, default=5))
    cs.add(Float("max_missing_prop", (0.8, 1.0), log = True, default=0.9))
    #cs.add(Categorical("tokenizer_class", ["MeanScaleUniformBins", "MeanScaleQuantileBins"], default="MeanScaleUniformBins"))
    # TODO Must be as long as there a Corpora cs.add(Categorical("probability", [probability_options]))
    cs.add(Categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant","constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau","cosine_with_min_lr"],default="linear"))
    cs.add(Integer("bolt", (0, 1),default=0))

    # Base Chronos Only
    cs.add(Float("tokenizer_limit", (5.0, 50), log=False, default=15))
    cs.add(Integer("n_tokens", (8, 13), log=False, default=12))
    cs.add(EqualsCondition(cs["tokenizer_limit"], cs["bolt"], 0))
    cs.add(EqualsCondition(cs["n_tokens"], cs["bolt"], 0))

    # Bolt Only Stuff
    cs.add(Integer("patch_size_expo", (0, 6), log = False, default=4))
    cs.add(Integer("patch_stride_expo", (0, 7), log = False, default=4)) #TODO how exxactlly does it work -> Adjust
    cs.add(Integer("use_reg_token", (0, 1),default=1))
    cs.add(EqualsCondition(cs["patch_size_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["patch_stride_expo"], cs["bolt"], 1))
    cs.add(EqualsCondition(cs["use_reg_token"], cs["bolt"], 1))
    # TODO Add Patch FFN HP


    return cs


if __name__ == "__main__":
    cs = get_config_space()
    print(cs)
