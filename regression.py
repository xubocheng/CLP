import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Sequence
import torch
import transformers
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=0.05)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).',},)
    flash_attn : Optional[bool] = field(default=False)
    output_dir: str = field(default="output")
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    seed: int = field(default=42)
    learning_rate: float = field(default=1e-4)
    #lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=50)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    checkpointing: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    save_model: bool = field(default=False)
    report_to: Optional[str] = field(default='none')
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Convert input_ids, labels, and attention_mask to tensors
        input_ids = torch.tensor([instance["input_ids"] for instance in instances], dtype=torch.long)
        labels = torch.tensor([round(instance["label"], 5) for instance in instances], dtype=torch.float)
        attention_mask = torch.tensor([instance["attention_mask"] for instance in instances], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
  
    logits_flat = logits.reshape(-1)

    #pearson_coeff, p_value = pearsonr(logits_flat, labels)
    r2 = r2_score(labels, logits_flat)
    rmse = mean_squared_error(labels, logits_flat, squared=False)
    mae = mean_absolute_error(labels, logits_flat)
    mse = mean_squared_error(labels, logits_flat)
    medae = median_absolute_error(labels, logits_flat)
    
    return {
        "r2_score": r2,
        "rmse": rmse,
        "mean_absolute_error": mae,
        "squared_error": mse,
        "median_absolute_error": medae,
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
   
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def train():
    parser = transformers.HfArgumentParser((TrainingArguments))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args = parser.parse_args_into_dataclasses()

    MODEL_NAME_OR_PATH = './esm_model' 

    tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH,padding_side='right',use_fast=True,
                                               model_max_length=training_args.model_max_length,
                                                trust_remote_code=True,)
    
    train_data_path = './reg_data/Staphylococcus aureus_reg_train.csv'
    test_data_path = '.reg_data/Staphylococcus aureus_reg_test.csv'
    # 加载数据集
    train_dataset = load_dataset('csv', data_files=train_data_path)['train']
    test_dataset = load_dataset('csv', data_files=test_data_path)['train']
    
    def tokenize_function(examples):
        return tokenizer(examples['Sequence'], padding='max_length',max_length=40, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH,num_labels=1,trust_remote_code=True)

    print('Creating and saving datasets...')
    
    print(len(train_dataset[0]['input_ids']))
    print(train_dataset[0])
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())

    print(f" base model - Total size={n_params/2**20:.2f}M params")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_trainable_params}")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=test_dataset,
                                   data_collator=data_collator)
    trainer.train()
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()
