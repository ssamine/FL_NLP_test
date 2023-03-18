# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import time
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
#important
from network import NLPClinicalBert

from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
from utils import *


class NLPTrainer(Executor):
    def __init__(self, bert_model='/home/localuser/custom/model/pretraining', 
                 readmission_mode='early', 
                 task_name='readmission', 
                 output_dir='/tmp/result', 
                 max_seq_length=128, 
                 train_batch_size=32, 
                 lr=5e-5, epochs=1, 
                 warmup_proportion=0.1, 
                 no_cuda=False, 
                 local_rank=-1, 
                 seed=42, 
                 gradient_accumulation_steps=1, 
                 optimize_on_cpu=False, 
                 fp16=False, 
                 loss_scale=128, 
                 train_task_name=AppConstants.TASK_TRAIN, 
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, 
                 exclude_vars=None):
        """Class :: NLPTrainer
        This class is used for federated training for the NLP task
        
        Args:
            bert_model (_type_): _description_
            readmission_mode (_type_): _description_
            task_name (_type_): _description_
            output_dir (_type_): _description_
            max_seq_length (int, optional): _description_. Defaults to 128.
            train_batch_size (int, optional): _description_. Defaults to 32.
            lr (_type_, optional): _description_. Defaults to 5e-5.
            epochs (int, optional): _description_. Defaults to 1.
            warmup_proportion (float, optional): _description_. Defaults to 0.1.
            no_cuda (bool, optional): _description_. Defaults to False.
            local_rank (int, optional): _description_. Defaults to -1.
            seed (int, optional): _description_. Defaults to 42.
            gradient_accumulation_steps (int, optional): _description_. Defaults to 1.
            optimize_on_cpu (bool, optional): _description_. Defaults to False.
            fp16 (bool, optional): _description_. Defaults to False.
            loss_scale (int, optional): _description_. Defaults to 128.
            train_task_name (_type_, optional): _description_. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task_name (_type_, optional): _description_. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            exclude_vars (_type_, optional): _description_. Defaults to None.
        """

        super().__init__()
        self.st_time = time.time()
        self.bert_model = bert_model
        self.readmission_mode = readmission_mode
        self.task_name = task_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length 
        self.train_batch_size=train_batch_size 
        self.warmup_proportion=warmup_proportion 
        self.no_cuda=no_cuda
        self.local_rank=local_rank 
        self.seed=seed 
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.optimize_on_cpu=optimize_on_cpu
        self.fp16=fp16
        self.loss_scale=loss_scale
        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.global_step = 0
        self.train_loss=100000
        self.number_training_steps=1
        self.global_step_check=0
        self.train_loss_history=[]
        self.model = NLPClinicalBert(model_state_dict=self.bert_model, num_labels=1)
        self.run_allied_functions()
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(data=self.model.state_dict(), default_train_conf=self._default_train_conf)
        print('\nApp Constants : Train : {} -- Submit : {} \n'.format(AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL))


    def check_device(self):
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
            if self.fp16:
                logger.info("16-bits training currently not supported in distributed training")
                self.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
        
        print('Device : {}'.format(self.device))
        logger.info("Device %s GPU %d distributed training %r", self.device, self.n_gpu, bool(self.local_rank != -1))
        

    def random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)


    def get_data(self):
        
        processors = {"readmission": readmissionProcessor}
        
        if os.path.exists(os.getcwd()):
            print(os.getcwd())
            print(os.listdir(os.getcwd()))
            print(os.listdir('/'))
            print(os.listdir('/input/'))
            # print(os.listdir('/output/'))
            # print(os.listdir('/input/cohorts/'))
            # cohort_uid = next(os.walk('/input/cohorts'))[1][0]
            # print(os.listdir('/input/cohorts/' + cohort_uid + '/'))
            
        # if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
        #     raise ValueError("Output directory ({}) already exists and is not empty.".format(self.output_dir))
        
        print(self.output_dir)
        # os.mkdir(self.output_dir)
        # print(os.listdir(self.output_dir))

        task_name = self.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                                self.gradient_accumulation_steps))
        
        processor = processors[task_name]()
        self.label_list = processor.get_labels()

        self.tokenizer = BertTokenizer.from_pretrained('/home/localuser/custom/token/bert-base-uncased-vocab.txt')
        # print(tokenizer)

        self.train_examples = None
        self.num_train_steps = None
        
        cohort_uid = next(os.walk('/input/cohorts'))[1][0]
        print(cohort_uid)
        train_file_path = '/input/cohorts/' + cohort_uid + '/cohort_data.csv'
        print(train_file_path)

        self.train_batch_size = int(self.train_batch_size / self.gradient_accumulation_steps)

        # Calculate training examples and training steps
        self.train_examples = processor.get_train_examples(train_file_path)
        print('Train Examples : {}'.format(len(self.train_examples)))
        
        # Demo (only 10% of data is selected)         
        self.demo = False
        if self.demo:
            self.train_examples = self.train_examples[:50]
            print('Train Examples : {}'.format(len(self.train_examples)))
            
        self.num_train_steps = int(len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps * self._epochs)
        
    
    def get_dataloaders(self):
        train_features = convert_examples_to_features(self.train_examples, self.label_list, self.max_seq_length, self.tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_examples))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
        # Select the data sampler
        train_sampler = RandomSampler(train_data) if self.local_rank == -1 else DistributedSampler(train_data)
        
        # Train Dataloader
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        self._n_iterations = len(self.train_dataloader)
        print('\nIterations : {} \n'.format(self._n_iterations))


    def prepare_model(self):

        if self.fp16:
            self.model.half()
        self.model.to(self.device)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Prepare optimizer
        if self.fp16:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) for n, param in self.model.named_parameters()]
        elif self.optimize_on_cpu:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) for n, param in self.model.named_parameters()]
        else:
            param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=self._lr, warmup=self.warmup_proportion, t_total=self.num_train_steps)
    
    
    def run_allied_functions(self):
        
        # Chcek :: Devices (GPU or CPU)
        print('Step:1 --Check Device--')
        self.check_device()
        
        # Generate :: Random Seed
        print('Step:2 --Seed Generation--')
        self.random_seed()
        
        # Calling :: Get data function
        print('Step:3 --Get Data--')
        self.get_data()
        
        # Calling :: Get data loaders
        print('\nStep:4 --Get Data Loader--\n')
        self.get_dataloaders()
        
        # Preparing :: Model and Model Configuration
        print('\nStep:5 --Prepare Model--\n')
        self.prepare_model()
    
    
    def local_train(self, fl_ctx, weights, abort_signal):
        
        # Set the model weights
        print('\nStep:6 --Start Model Training-- ....\n ')
        self.model.load_state_dict(state_dict=weights)
        self.model.train()
        
        for epo in trange(int(self._epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.fp16 and self.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * self.loss_scale
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                self.train_loss_history.append(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16 or self.optimize_on_cpu:
                        if self.fp16 and self.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / self.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, self.model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            self.loss_scale = self.loss_scale / 2
                            self.model.zero_grad()
                            continue
                        self.optimizer.step()
                        copy_optimizer_params_to_model(self.model.named_parameters(), param_optimizer)
                    else:
                        self.optimizer.step()
                    self.model.zero_grad()
                    self.global_step += 1
                
                if (step+1) % 200 == 0:
                    string = 'step '+str(step+1)
                    print (string)
                
                string_ = 'Epoch: {}/{}, Steps: {}, Loss: {}'.format(epo, self._epochs, step, loss.item()/1.0)
                self.log_info(fl_ctx, string_)
                            
            self.train_loss=tr_loss
            self.global_step_check=self.global_step
            self.number_training_steps=nb_tr_steps
            print('End of Epoch :: {}/{} : Loss : {}'.format(epo, self._epochs, tr_loss))
            print('Elapsed Time : {:.2f} seconds'.format(time.time() - self.st_time))


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        print('\n -- Execute -- \n')
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except Exception:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_files kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)


    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
                