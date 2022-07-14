import torch
import textpruner
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, BertForMaskedLM
from textpruner import TransformerPruner, TransformerPruningConfig, GeneralConfig

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def transformer_pruner(model, all_data_loader):
    transformer_pruning_config = TransformerPruningConfig(
        target_ffn_size=2048,
        target_num_of_heads=6,
        head_even_masking=False,
        use_logits=True,
        pruning_method='iterative',
        pruning_order="ffn-first",
        n_iters=16)
    pruner = TransformerPruner(model, transformer_pruning_config=transformer_pruning_config,
                               # general_config=GeneralConfig(use_device='auto', output_dir='./pruned_models')
                               )
    pruner.prune(dataloader=all_data_loader, save_model=True)
    # https://textpruner.readthedocs.io/en/latest/APIs/Configurations.html
    # pruner.save_jit_model(example_inputs=[all_data_loader.dataset["input_ids"][:2].to(device),
    #                                       all_data_loader.dataset["token_type_ids"][:2].to(device),
    #                                       all_data_loader.dataset["attention_mask"][:2].to(device)
    #                                       ], dir_name="d.pt")


def evaluate_predict_result(model_path: str, test_file: str):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    with open(test_file, 'r', encoding='utf-8') as f1:
        data = pd.read_table(f1)
    data_dict = data.to_dict(orient='list')

    all_encodings = tokenizer(data_dict['sentence'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    all_dataset = IMDbDataset(all_encodings, data_dict['label'])
    all_data_loader = DataLoader(dataset=all_dataset, batch_size=32, shuffle=True)
    transformer_pruner(model, all_data_loader)


def view_the_influence_parameters(model_path):
    model = BertForMaskedLM.from_pretrained(model_path)
    print("Model summary:")
    print(textpruner.summary(model, max_level=3))


if __name__ == '__main__':
    evaluate_predict_result(model_path="../../model/emotional_cls_4", test_file="../../data/train.csv")
    # view_the_influence_parameters(model_path="./pruned_models/d.pt/pytorch_model.ts")
