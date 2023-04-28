import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification, Trainer, TrainingArguments
from typing import Any, Dict, List, NewType
import csv
from ast import literal_eval as make_tuple
import argparse
InputDataClass = NewType("InputDataClass", Any)
from evaluation import evaluate

class GeoQuery(Dataset):
    def __init__(self, dataframe, idx, tokenizer):
        all_nls = dataframe.NL.tolist()
        all_mrs = dataframe.MR.tolist()
        all_golds = dataframe.GOLD.tolist()
        self.nls = [all_nls[i] for i in range(len(all_nls)) if i in idx]
        self.mrs = [all_mrs[i] for i in range(len(all_mrs)) if i in idx]
        self.golds = [all_golds[i] for i in range(len(all_golds)) if i in idx]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.nls)

    def __getitem__(self, i):
        item = {}
        item["x"] = self.nls[i]
        item["label"] = self.mrs[i]
        item["tokenizer"] = self.tokenizer
        item["gold"] = self.golds[i]
        return item

class LexicalPredictionsTestDataset(Dataset):
    def __init__(self, lexical_pred_df, dataframe, idx, tokenizer):
        ids = lexical_pred_df.ID.tolist()

        if set(ids) != set(idx):
            raise Exception("Lexical predictions indexes and test set indexes are different")

        preds = lexical_pred_df.PRED.tolist()
        preds = [i.split() for i in preds]

        self.nls = preds

        all_mrs = dataframe.MR.tolist()
        all_golds = dataframe.GOLD.tolist()
        self.mrs = [all_mrs[i] for i in ids]
        self.golds = [all_golds[i] for i in ids]

        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.nls)

    def __getitem__(self, i):
        item = {}
        item["x"] = self.nls[i]
        item["label"] = self.mrs[i]
        item["tokenizer"] = self.tokenizer
        item["gold"] = self.golds[i]
        return item

def _align_labels_with_tokenization(offset_mapping, labels):
    new_labels = []
    for o, l in zip(offset_mapping, labels):
        lab = []
        count = -1
        for i in o:
            if i[0]==0 and i[1]==0:
                lab.append(EPSILON_LABEL)
            elif i[0] == 0:
                count += 1
                lab.append(l[count])
            else:
                lab.append(EPSILON_LABEL)
        new_labels.append(lab)
    return new_labels

def bert_classifier_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    tokenizer = features[0]['tokenizer']
    x = [i["x"] for i in features]
    a = tokenizer(x, return_offsets_mapping=True, is_split_into_words=True, padding=True, return_tensors = 'pt')
    offset_mapping = a['offset_mapping']
    if 'label' in features[0]:
        labels = [i["label"] for i in features]
        labels = _align_labels_with_tokenization(offset_mapping, labels)
        labels = torch.tensor(labels)
        a['labels'] = labels
    del a['offset_mapping']
    return a

def preprocess_MR(sentence):
    sequence = sentence.replace('(', ' ( ').replace(')', ' ) ').split()
    s = []
    stop_at_j = -1

    for j in range(len(sequence)):
        if stop_at_j > 0:
            stop_at_j = stop_at_j -1
        elif sequence[j] in ('(', ')'):
            continue
        elif sequence[j] in ('stateid', 'riverid', 'cityid', 'countryid', 'placeid'):
            i = j
            while sequence[i] != ')':
                i += 1
            x = ' '.join(sequence[j:i+1])
            x = x.replace(' ( ', '(').replace(' )', ')')
            s.append(x)
            stop_at_j = i-j
        elif sequence[j] == 'all':
            s[-1] = s[-1] + '(all)'
        else:
            s.append(sequence[j])
    
    s =[j for j in s if j != ',']

    return s

def preprocess_MR_ALIGNMENT(text):
    s = make_tuple(text)
    s = [j[1] for j in s if j[1]!='ε']
    return s

EPSILON_LABEL = 0
def create_label_vocabulary(data, idx):
    seq = data.MR.tolist()
    seq = [seq[i] for i in range(len(seq)) if i in idx]

    labeltoid = {}
    idtolabel = {}
    count = 0

    labeltoid['ε'] = 0
    idtolabel[0] = 'ε'
    count = 1

    for i in seq:
        for j in i:
            if j not in labeltoid:
                labeltoid[j] = count
                idtolabel[count] = j
                count += 1
    return labeltoid, idtolabel

def remove_epsilons(x):
    s = make_tuple(x)
    s = [j[1] for j in s if j[1]!='ε']
    return s

def convert_MR_to_id(data, labeltoid):
    mr = data.MR.tolist()
    mr = [[labeltoid[j] if j in labeltoid else EPSILON_LABEL for j in i] for i in mr]
    data.MR = mr

def preprocess_data(data, test_idx, val_idx, train_idx):
    data.MR = data["MR"].apply(preprocess_MR)
    data.NL = data["ALIGNMENT"].apply(preprocess_MR_ALIGNMENT)

    labeltoid, idtolabel = create_label_vocabulary(data, train_idx+val_idx)

    data["GOLD"] = data["MR"]

    convert_MR_to_id(data, labeltoid)

    a = data.MR.tolist()
    b = data.NL.tolist()

    return labeltoid, idtolabel

def run(args):

    data = pd.read_csv(args.dataset)

    test_idx = [int(line.strip()) for line in open(args.test_ids)]
    val_idx = [int(line.strip()) for line in open(args.val_ids)]
    train_idx = [i for i in data.ID if i not in test_idx + val_idx]

    test_idx.sort()
    val_idx.sort()

    labeltoid, idtolabel = preprocess_data(data, test_idx, val_idx, train_idx)

    if args.language == 'it':
        MODEL_NAME = "dbmdz/bert-base-italian-uncased"
    elif args.language == 'de':
        MODEL_NAME = "dbmdz/bert-base-german-uncased"
    elif args.language == 'en':
        MODEL_NAME = 'bert-base-uncased'

    config=AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(labeltoid)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, config=config)

    lexical_predictions_data= pd.read_csv(args.lexical_predictions)
    test_data = LexicalPredictionsTestDataset(lexical_predictions_data, data, test_idx, tokenizer)
    
    args_t = TrainingArguments(
        output_dir = 'models/',
        num_train_epochs = 25,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy = 'epoch',
        learning_rate = 1e-4,
        weight_decay=0.01,
        save_total_limit=3,
        push_to_hub=False,
    )

    trainer = Trainer(
        data_collator=bert_classifier_data_collator,
        model=model,                         
        args=args_t,                 
        train_dataset=GeoQuery(data,train_idx,tokenizer),     
        eval_dataset=GeoQuery(data,val_idx,tokenizer),
    )    
   
    trainer.train()    

    preds = []
    golds = []
    for i in range(len(test_data)):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      with torch.no_grad():
        logits = model(tokenizer(test_data[i]["x"], is_split_into_words=True, return_tensors = 'pt').to(device)["input_ids"]).logits
        pred = logits.argmax(-1)

      gold = test_data[i]["gold"]
      pred = [idtolabel[i] for i in pred.cpu()[0].tolist()]
      pred = [j for j in pred if j != 'ε']

      preds.append(' '.join(pred))
      golds.append(' '.join(gold))


    monotonic = data.MONOTONIC.tolist()
    monotonic = [monotonic[i] for i in range(len(monotonic)) if i in test_idx]

    nls = data.NL.tolist()
    test_nls = [nls[i] for i in test_idx]

    new_df = pd.DataFrame({"ID":test_idx, "NL":test_nls, "MR": golds, "PRED": preds, "MONOTONIC": monotonic})
    new_df.to_csv(args.out_file)

    stats = evaluate(preds, golds, monotonic, verbose=True)

    if args.results_file is not None:
        with open(args.results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['exact', 'exact_mn', 'exact_nmn', 'tokens', 'tokens_mn', 'tokens_nmn', 'span', 'span_mn', 'span_nmn'])
            writer.writerow([stats['exact_match']['acc'], stats['exact_match']['mn_acc'], stats['exact_match']['nmn_acc'],
                            stats['no_correct_tokens']['acc'], stats['no_correct_tokens']['mn_acc'], stats['no_correct_tokens']['nmn_acc'],
                            stats['max_correct_span']['acc'], stats['max_correct_span']['mn_acc'], stats['max_correct_span']['nmn_acc'],])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='input language (en, it, de)', required=True)
    parser.add_argument('--dataset', type=str, help='dataset path', required=True)
    parser.add_argument('--test-ids', type=str, help='test ids dataset path', required=True)
    parser.add_argument('--val-ids', type=str, help='val ids dataset path', required=True)
    parser.add_argument('--out-file', type=str, help='out file path', required=True)
    parser.add_argument('--lexical-predictions', type=str, help='lexical predictions dataset path', required=True)
    parser.add_argument('--results-file', type=str, help='file path with results')
    args = parser.parse_args()

    run(args)


