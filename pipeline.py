import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import time
import re
import os
import pandas as pd

class SequencesDataset(Dataset):
   
    def __init__(self, sequences, tokenizer, max_length=50):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'sequences': sequence
        }

def load_and_process_model(model_path, dataloader, save_path, is_regression=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if is_regression:
        model_name_or_path = './esm_model/'
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1).to(device)
        new_weights_path = './BS_regression/checkpoint-1000/pytorch_model.bin'#the path of the regression model
        new_state_dict = torch.load(new_weights_path)
        model.load_state_dict(new_state_dict)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2 if not is_regression else 1).to(device)
        new_weights_path = './BS/checkpoint-1000/pytorch_model.bin'#the path of the classification model
        new_state_dict = torch.load(new_weights_path)
        model.load_state_dict(new_state_dict)

    batched_process(model, dataloader, save_path, is_regression=is_regression)
    del model  
    torch.cuda.empty_cache()  


def batched_process(model, dataloader, save_path, is_regression=False):
    
    model.eval()
    device = model.device

    with torch.no_grad():
        
        for batch in tqdm(dataloader, desc="Processing batches"):
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          
            if is_regression:
                preds = outputs.logits.squeeze().tolist()
            else:
                preds = torch.argmax(outputs.logits, dim=1).tolist()

            with open(save_path, 'a') as f:
                for seq, pred in zip(batch['sequences'], preds):
                    if is_regression:
                        if pred < 4:
                            f.write(f"{seq}\t{pred}\n")
                    else:
                        if pred == 1:  # assume positive label is 1
                            f.write(f"{seq}\n")


def main():
    data_path = './rand_gen_seq.csv'
    save_base_path = './filter_result'
    classification_save_path = os.path.join(save_base_path, 'classification/BS_positive_sequences.txt')
    regression_save_path = os.path.join(save_base_path, 'regression/BS_final_sequences_with_labels.txt')

    library_sequences = pd.read_csv(data_path)['Sequence'].tolist()

    start_time = time.time()

    # Classification model
    MODEL_NAME_OR_PATH = './esm_model'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side='right', use_fast=True, model_max_length=30, trust_remote_code=True)
    classification_dataset = SequencesDataset(library_sequences, tokenizer, max_length=30)
    classification_dataloader = DataLoader(classification_dataset, batch_size=64, shuffle=False)
    load_and_process_model(MODEL_NAME_OR_PATH, classification_dataloader, classification_save_path, is_regression=False)
    print(f"Positive sequences saved to {classification_save_path}")

    # Read positive sequences
    positive_sequences = []
    with open(classification_save_path, 'r') as f:
        positive_sequences = [line.strip() for line in f]

    # Regression model
    print('**************************** Starting regression model for broad-spectrum bacteria ****************************')
    regression_dataset = SequencesDataset(positive_sequences, tokenizer, max_length=30)
    regression_dataloader = DataLoader(regression_dataset, batch_size=64, shuffle=False)
    load_and_process_model(MODEL_NAME_OR_PATH, regression_dataloader, regression_save_path, is_regression=True)
    print(f"Non-toxic sequences with labels saved to {regression_save_path}")

    # Calculate and output total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
