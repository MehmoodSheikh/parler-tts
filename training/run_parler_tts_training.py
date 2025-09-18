#!/usr/bin/env python
# coding=utf-8
""" Train Parler-TTS using 🤗 Accelerate"""

import sys
import os
import logging
from multiprocess import set_start_method
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from huggingface_hub import HfApi
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry

from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, DistributedDataParallelKwargs
from accelerate.utils.memory import release_memory

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
    build_delay_pattern_mask,
)

from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from training.arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from training.data import load_multiple_datasets, DataCollatorParlerTTSWithPadding, DataCollatorEncodecWithPadding
from training.eval import clap_similarity, wer, si_sdr

logger = logging.getLogger(__name__)

def main():
    # At the very beginning of main(), ensure models are registered
    try:
        from parler_tts import register_parler_tts_models
        register_parler_tts_models()
        logger.info("Successfully registered ParlerTTS models")
    except ImportError as e:
        logger.warning(f"Failed to import model registration: {e}")
        logger.info("Attempting to continue without explicit registration")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_parler_tts", model_args, data_args)

    # Handle mixed precision setup
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        autocast_kwargs = AutocastKwargs(enabled=True, cache_enabled=True)
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        autocast_kwargs = AutocastKwargs(enabled=True, cache_enabled=True)
    else:
        mixed_precision = "no"
        autocast_kwargs = AutocastKwargs(enabled=False)

    # Padding validation
    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "When using pad_to_max_length, you must specify max_duration_in_seconds, "
            "max_prompt_token_length, and max_description_token_length"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    ####### A. Preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120)), DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    # Initialize tracker with configuration
    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        },
        init_kwargs={"wandb": {"name": data_args.wandb_run_name}} if data_args.wandb_run_name else {},
    )

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore.read():
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore.read():
                    gitignore.write("epoch_*\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Now we can proceed with model and dataset loading...
    
    # Load configuration
    config = ParlerTTSConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # Enhanced model loading with error handling
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )
        accelerator.print("Model loaded successfully")
    except (KeyError, ValueError) as e:
        if "parler_tts" in str(e):
            accelerator.print(f"Model loading failed due to registration issue: {e}")
            accelerator.print("Re-registering models and retrying...")
            from parler_tts import register_parler_tts_models
            register_parler_tts_models()
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                config=config,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
            )
            accelerator.print("Model loaded successfully after re-registration")
        else:
            raise e

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # Rest of your training code continues here...
    # (dataset loading, training loop, etc.)

if __name__ == "__main__":
    main()
