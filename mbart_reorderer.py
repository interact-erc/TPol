import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import  DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MBartForConditionalGeneration, MBart50TokenizerFast
import csv
from ast import literal_eval as make_tuple
import argparse
from evaluation import evaluate

class GeoQuery(Dataset):
    def __init__(self, dataframe, idx, tokenizer):
        all_nls = dataframe.NL.tolist()
        all_mrs = dataframe.MR.tolist()
        self.nls = [all_nls[i] for i in range(len(all_nls)) if i in idx]
        self.mrs = [all_mrs[i] for i in range(len(all_mrs)) if i in idx]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.nls)

    def __getitem__(self, i):
        item = {}
        item["input_ids"] = self.tokenizer(self.nls[i])["input_ids"]
        with self.tokenizer.as_target_tokenizer():
            item["labels"] = self.tokenizer(self.mrs[i])["input_ids"]
        return item

class LexicalPredictionsTestDataset(Dataset):
    def __init__(self, lexical_pred_df, dataframe, idx, tokenizer):
        ids = lexical_pred_df.ID.tolist()

        if set(ids) != set(idx):
            raise Exception("Lexical predictions indexes and test set indexes are different")

        preds = lexical_pred_df.PRED.tolist()
        self.nls = preds   
        all_mrs = dataframe.MR.tolist()
        self.mrs = [all_mrs[i] for i in ids]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.nls)

    def __getitem__(self, i):
        item = {}
        item["input_ids"] = self.tokenizer(self.nls[i])["input_ids"]
        with self.tokenizer.as_target_tokenizer():
            item["labels"] = self.tokenizer(self.mrs[i])["input_ids"]
        return item


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

    return ' '.join(s)


def preprocess_MR_ALIGNMENT(text):
    s = make_tuple(text)
    s = [j[1] for j in s if j[1]!='ε']
    return ' '.join(s)


def run(args):

    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    if args.language == 'it':
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang="it_IT", tgt_lang="en_XX")
    elif args.language == 'de':
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang="de_DE", tgt_lang="en_XX")
    elif args.language == 'en':
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang="en_XX", tgt_lang="en_XX")

    args_t = Seq2SeqTrainingArguments(
        output_dir = 'models/',
        evaluation_strategy = "epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=8,
        num_train_epochs=30,
        predict_with_generate=True,
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    data = pd.read_csv(args.dataset)

    test_idx = [int(line.strip()) for line in open(args.test_ids)]
    val_idx = [int(line.strip()) for line in open(args.val_ids)]
    train_idx = [i for i in data.ID if i not in test_idx + val_idx]

    test_idx.sort()
    val_idx.sort()

    data.MR = data["MR"].apply(preprocess_MR)
    if args.silver_predictions_path is not None:
        silver_predictions_data = pd.read_csv(args.silver_predictions_path)
        data.NL = silver_predictions_data.PRED_ALIGNMENT
    else:
        data.NL = data['ALIGNMENT'].apply(preprocess_MR_ALIGNMENT)

    lexical_predictions_data= pd.read_csv(args.lexical_predictions)
    test_data = LexicalPredictionsTestDataset(lexical_predictions_data, data, test_idx, tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args_t,
        train_dataset=GeoQuery(data,train_idx,tokenizer),
        eval_dataset=GeoQuery(data,val_idx,tokenizer),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    preds = []
    golds = []
    for i in range(len(test_data)):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      pred = model.generate(input_ids = torch.tensor(test_data[i]["input_ids"]).to(device).view(1,-1), forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
      #pred = model.generate(input_ids = torch.tensor(test_data[i]["input_ids"]).to(device).view(1,-1))
      pred = tokenizer.decode(np.array(pred.cpu()[0])).replace('<s>','').replace('</s>','')
      p = pred.split()
      if p[0] in ('en_XX', 'it_IT', 'de_DE'):
        p = p[1:]
      p = ' '.join(p)
      preds.append(p)
      gold = tokenizer.decode(np.array(test_data[i]["labels"])).replace('<s>','').replace('</s>','')
      g = gold.split()
      if g[0] in ('en_XX', 'it_IT', 'de_DE'):
        g = g[1:]
      g = ' '.join(g)
      golds.append(g)

    monotonic = data.MONOTONIC.tolist()
    monotonic = [monotonic[i] for i in range(len(monotonic)) if i in test_idx]


    nls = data.NL.tolist()
    test_nls = [nls[i] for i in test_idx]

    new_df = pd.DataFrame({"ID":test_idx, "NL":test_nls, "MR": golds, "PRED": preds, "MONOTONIC": monotonic})
    new_df.to_csv(args.out_file)

    stats = evaluate(preds, golds, monotonic, True)

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
    parser.add_argument('--silver-predictions-path', type=str, help='path to all predictions on train data to train reorderer in silver mode')
    parser.add_argument('--results-file', type=str, help='file path with results')
    args = parser.parse_args()

    run(args)


