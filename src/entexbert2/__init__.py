"""entexbert2 -- two-stage transfer learning with a DNA transformer for predicting allele-specific activity

Stage 1 trains a binding trunk on reference-genome signal
Stage 2 fine-tunes a twin contrast head on EN-TEx allele-specific read counts

Public names are resolved LAZILY (PEP 562): 
importing `entexbert2` itself pulls in no third-party package,
so `import entexbert2` works in an environment without torch, pyBigWig or pyfaidx
"""

__version__ = "0.1.0"

# public name -> the submodule that defines it
_EXPORTS = {
    # dataset building
    "build_dataset": "build_inputs",
    "WindowSpec": "build_inputs",
    "LabelSpec": "build_inputs",
    "BalanceSpec": "build_inputs",
    "PartitionSpec": "build_inputs",
    "RowSource": "build_inputs",
    "HaplotypeCountRowSource": "build_inputs",
    "MultiTissuePeakRowSource": "build_inputs",
    # config-driven runner
    "run_from_config": "build_experiment",
    "load_config": "build_experiment",
    # personal genomes
    "PersonalGenome": "personal_seq",
    "parse_chain_file": "personal_seq",
    # model + scoring
    "entexBERT2ForSequencePrediction": "model",
    "load_model_and_tokenizer": "model_io",
    "run_inference": "model_io",
    "build_windows": "eval_utils",
    "load_hetsnv": "eval_utils",
}

__all__ = sorted(_EXPORTS) + ["__version__"]

def __getattr__(name):
    """PEP 562 lazy attribute access -- import the submodule only when the name is used"""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)

def __dir__():
    return __all__
