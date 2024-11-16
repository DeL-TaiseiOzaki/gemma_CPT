import argparse
import os
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

import deepspeed
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocess_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int
) -> Dict[str, List[int]]:
    """データの前処理関数"""
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs["labels"] = inputs.input_ids.copy()
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        "-p",
        type=str,
        default="./configs/train_configs/train_base.yaml",
        help="Training configuration file path",
    )
    parser.add_argument("--local_rank", "-l", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    local_rank = args.local_rank

    # Load config
    config = OmegaConf.load(args.train_config)

    # Initialize distributed training
    deepspeed.init_distributed()

    # Set seed
    seed_everything(config.seed)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model, 
        torch_dtype=torch.float16, 
        use_cache=config.model.use_cache
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer,
        add_eos_token=True,
    )

    # Load dataset
    dataset = load_dataset(
        config.dataset.path, 
        split=config.dataset.split
    )

    # Preprocess dataset
    dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, config.model.max_length
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.train_test_split(test_size=0.2)

    # Initialize trainer
    training_args = TrainingArguments(**config.train)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    # Start training
    with torch.autocast("cuda"):
        trainer.train()

if __name__ == "__main__":
    main()