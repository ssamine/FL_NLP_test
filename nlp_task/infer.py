from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import sys
from pathlib import Path
import csv
import os
import logging
import argparse
import random
from tqdm import trange, tqdm
import matplotlib as mpl
mpl.use('Agg')
from scipy import interp
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from funcsigs import signature
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from utils import *
from network import NLPClinicalBert


def infer(model_weights_file_path):

    max_seq_length = 512
    eval_batch_size = 8
    local_rank = -1
    output_dir = '/output/'
    readmission_mode = 'early'

    m = nn.Sigmoid()
    test_file_path = '/input/cohort_data.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device : {}'.format(device))
    processors = {"readmission": readmissionProcessor}
    task_name = 'readmission'
    processor = processors[task_name]()
    eval_examples = processor.get_test_examples(test_file_path)
    print('Testing Examples : {}'.format(len(eval_examples)))

    # Demo (only 10% of data is selected)         
    demo = False
    if demo:
        eval_examples = eval_examples[:20]
        print('Test Examples : {}'.format(len(eval_examples)))
    
    label_list = processor.get_labels()     
    tokenizer = BertTokenizer.from_pretrained('/home/localuser/custom/token/bert-base-uncased-vocab.txt')
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)    
    eval_sampler = SequentialSampler(eval_data) if local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    bert_model = '/home/localuser/custom/model/pretraining'
    model = NLPClinicalBert(model_state_dict=bert_model, num_labels=1)
    model.load_state_dict(torch.load(model_weights_file_path)["model"])
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_labels=[]
    pred_labels=[]
    logits_history=[]
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        model.to(device)
        with torch.no_grad():
            tmp_eval_loss, temp_logits = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids,segment_ids,input_mask)
        
        logits = torch.squeeze(m(logits)).detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
        tmp_eval_accuracy=np.sum(outputs == label_ids)
        
        true_labels = true_labels + label_ids.flatten().tolist()
        pred_labels = pred_labels + outputs.flatten().tolist()
        logits_history = logits_history + logits.flatten().tolist()

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
        
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    print('Result : Loss : {}, Accuracy : {}'.format(eval_loss, eval_accuracy))

    # df = pd.DataFrame({'logits':logits_history, 'pred_label': pred_labels, 'label':true_labels})

    print('Logits History : {}'.format(logits_history))
    print('Preds Labels : {}'.format(pred_labels))
    print('True Labels : {}'.format(true_labels))

    print(len(logits_history), len(pred_labels), len(true_labels))

    df_test = pd.read_csv(test_file_path)
    print(df_test.shape)
    # df_test = df_test[:len(logits_history)]

    fpr, tpr, df_out = vote_score(df_test, logits_history, output_dir)

    rp80 = vote_pr_curve(df_test, logits_history, output_dir)

    result = {
        'eval_loss': '{:.2f}'.format(eval_loss),
        'eval_accuracy': '{:.2f}'.format(eval_accuracy),
        'RP80': rp80
        }

    print('Result : {}'.format(result))
    
    # Output File
    df_test['logits_chunks'] = logits_history
    df_test['pred_label_chunks'] = pred_labels
    df_test['label_chunks'] = true_labels
    df_test['logits_readmission'] = df_out['logits'].to_numpy()
    df_test['ID_readmission'] = df_out['ID'].to_numpy()
    df_test.to_csv(os.path.join(output_dir, 'cohort_data.csv'), index=False)
    
    # output_eval_file = os.path.join(output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    args = sys.argv[1:]
    (model_weights_file_path,) = args

    ## Check -- Sahil
    print('Start the prediction ....')
    print('Args : ', args)
    print(model_weights_file_path)
    outpt = os.getcwd()
    print(outpt)
    print('Files : ', os.listdir(outpt))
    
    infer(model_weights_file_path)
    sys.exit(0)
