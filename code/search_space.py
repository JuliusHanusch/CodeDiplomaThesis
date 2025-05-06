from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, Constant, UniformIntegerHyperparameter, UniformFloatHyperparameter

probability_options = [
    "[0.8, 0.1, 0.1]",
    "[0.7, 0.2, 0.1]",
    "[0.6, 0.2, 0.2]",
    "[0.9, 0.05, 0.05]",
    "[0.85, 0.1, 0.05]"
]

def get_config_space(model_id = "google/t5-efficient-tiny") -> ConfigurationSpace:
    # Fixed parameters to be added as constants
    fixed_config = {
        "training_data_paths": [ # TODO Update
            "your/path/to/training_mix.arrow",
            "your/path/tp/kernelsynth.arrow"
            "your/path/to/realdata.arrow"
        ],
        "context_length": 512, # TODO Include
        "prediction_length": 64, # TODO Include
        "min_past": 60,
        "max_steps": 200_000, # TODO Cut (We set via Budget)
        "save_steps": 200_000,
        "log_steps": 500,
        "optim": "adamw_torch_fused", # TODO Are there alternatives
        "num_samples": 20, # TODO I dont get what this is
        "shuffle_buffer_length": 100_000,
        "gradient_accumulation_steps": 1, # TODO Define Manually 
        "tokenizer_class": "MeanScaleUniformBins",
        "model_id": model_id, # TODO Three
        "model_type": "seq2seq",
        "random_init": True,
        "tf32": True,
        "torch_compile": True,
        "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0}, # TODO Include
        "dataloader_num_workers": 1,
        "max_missing_prop": 0.9, # TODO Include
        "lr_scheduler_type": "linear", # TODO Include
        "use_eos_token": True,
    }


    cs = ConfigurationSpace()

    # Add fixed parameters as constants if compatible
    for key, value in fixed_config.items():
        if isinstance(value, (int, float, str, bool)):
            cs.add(Constant(key, value))

    # Add tunable hyperparameters
    cs.add(UniformIntegerHyperparameter("n_tokens", lower= 2048, upper = 8192, log=False))
    cs.add(UniformFloatHyperparameter("learning_rate", lower = 0.00001, upper = 0.01, log = True))
    cs.add(UniformFloatHyperparameter("warmup_ratio",lower = 1e-6, upper = 0.1, log = True))
    cs.add(UniformFloatHyperparameter("dropout_rate", lower = 1e-6, upper = 0.2, log = True)) # TODO not valid yet
    cs.add(CategoricalHyperparameter("feed_forward_proj", ["relu", "gated-relu"]))
    cs.add(UniformFloatHyperparameter("layer_norm_epsilon", lower = 1e-07, upper = 1e-05, log = True))
    cs.add(UniformIntegerHyperparameter("d_model", lower=64, upper=512, log = False)) # TODO not valid
    cs.add(UniformIntegerHyperparameter("d_ff", lower = 356, upper = 2048, log = False)) # TODO not valid arg
    cs.add(UniformIntegerHyperparameter("num_layers", lower = 3, upper = 6, log = False))
    cs.add(UniformIntegerHyperparameter("num_heads", lower = 4, upper = 8, log = False))
    cs.add(CategoricalHyperparameter("probability", [probability_options]))
    return cs

if __name__ == "__main__":
    cs = get_config_space()
    print(cs)
