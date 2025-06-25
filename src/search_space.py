from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, Constant, UniformIntegerHyperparameter, UniformFloatHyperparameter

# TODO Update
#training_data_paths = "['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']"
training_data_paths = "['/data/horse/ws/jipo020b-aion/AION/data/train/tsm_for_Time_Corpus_Processed_with_k-3_length-128_alpha-1_5.arrow']"


probability_options = [
    "[0.8, 0.1, 0.1]",
    "[0.7, 0.2, 0.1]",
    "[0.6, 0.2, 0.2]",
    "[0.9, 0.05, 0.05]",
    "[0.85, 0.1, 0.05]"
]
# define data splits

def get_config_space(model_id = "google/t5-efficient-tiny", batch_size= 256, gradient_accumulation_steps = 1) -> ConfigurationSpace:
    # TODO include batch_size into search space
    # Fixed parameters to be added as constants
    fixed_config = {
        "training_data_paths": [ # TODO Update
            "your/path/to/training_mix.arrow",
            "your/path/tp/kernelsynth.arrow"
            "your/path/to/realdata.arrow"
        ],
        "per_device_train_batch_size": batch_size,
        "min_past": 60,
        "save_steps": 200_000,
        "log_steps": 500,
        "num_samples": 20,
        "shuffle_buffer_length": 100_000,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model_id": model_id,
        "model_type": "seq2seq",
        "random_init": True,
        "tf32": True,
        "torch_compile": True,
        "dataloader_num_workers": 1,
        "use_eos_token": True,
    }


    cs = ConfigurationSpace()

    # Add fixed parameters as constants if compatible
    for key, value in fixed_config.items():
        if isinstance(value, (int, float, str, bool)):
            cs.add(Constant(key, value))

    # Add tunable hyperparameters
    cs.add(UniformIntegerHyperparameter("n_tokens", lower= 512, upper = 8192, log=False, default_value=4096))
    cs.add(UniformFloatHyperparameter("learning_rate", lower = 0.00005, upper = 0.01, log = True, default_value=0.001))
    cs.add(UniformFloatHyperparameter("warmup_ratio",lower = 1e-7, upper = 0.1, log = True, default_value=1e-7))
    cs.add(UniformFloatHyperparameter("dropout_rate", lower = 1e-7, upper = 0.2, log = True, default_value=1e-7)) 
    cs.add(CategoricalHyperparameter("feed_forward_proj", ["relu", "gated-gelu", "gated-relu"], default_value="relu"))
    cs.add(CategoricalHyperparameter("optim", ["adamw_torch_fused", "adafactor"], default_value="adamw_torch_fused"))
    cs.add(UniformFloatHyperparameter("layer_norm_epsilon", lower = 1e-07, upper = 1e-03, log = True, default_value=1e-6))
    cs.add(UniformIntegerHyperparameter("d_model", lower=64, upper=2048, log = False, default_value=512))
    cs.add(UniformIntegerHyperparameter("num_layers", lower = 1, upper = 12, log = False, default_value=6))
    cs.add(UniformIntegerHyperparameter("num_heads", lower = 1, upper = 12, log = False, default_value=8))
    cs.add(UniformIntegerHyperparameter("d_kv", lower = 3, upper = 10, log = False, default_value=6))
    cs.add(UniformIntegerHyperparameter("d_ff", lower = 5, upper = 13, log = False, default_value=11))
    cs.add(UniformIntegerHyperparameter("context_length", lower = 128, upper = 2048, log = False, default_value=512))
    cs.add(UniformIntegerHyperparameter("prediction_length", lower = 16, upper = 128, log = False, default_value=64))
    cs.add(UniformFloatHyperparameter("max_missing_prop", lower=0.8, upper=1.0, log = True, default_value=0.9))
    #cs.add(CategoricalHyperparameter("tokenizer_class", ["MeanScaleUniformBins", "MeanScaleQuantileBins"], default_value="MeanScaleUniformBins"))
    # TODO Must be as long as there a Corpora cs.add(CategoricalHyperparameter("probability", [probability_options]))
    cs.add(CategoricalHyperparameter("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant","constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau","cosine_with_min_lr", "warmup_stable_decay"],default_value="linear"))
    cs.add(UniformFloatHyperparameter("tokenizer_limit", lower=5.0, upper=50, log=False, default_value=15))

    return cs


if __name__ == "__main__":
    cs = get_config_space()
    print(cs)
