# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import copy


import numpy as np
import torch
import torch.nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForMaskedLM, BertForQuestionAnswering, \
    AlbertTokenizer, AlbertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from src.bert_feature_extractor import convert_nodes_to_examples, convert_examples_to_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.get_vocab()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        # sample = self.sample_to_doc[item]
        # if sample["line"]% 2 != 0:
        #     return

        t1, t2, t3, t4, is_next_label = self.random_sent(item)

        # tokenize
        q_and_a = (t3 + " " + t4).replace(t4, "[MASK]")
        context = (t1 + " " + t2).replace(t4, "[ANS] " + t4 + " [/ANS]")
        # context = context.replace(t5, "[SUB] " + t4 + " [/SUB]")
        # context.replace(t4, "")
        # q_and_a.replace(t4, "[MASK]")
        # context.replace(t4, "[ANS] " + t4 + " [/ANS]")

        tokens_a = self.tokenizer.tokenize(t3 + t4)   # question在前
        tokens_b = self.tokenizer.tokenize(t1 + t2)     # target + context
        answer_ids = self.tokenizer.encode(t4, add_special_tokens=False)
        tokens_a_context = self.tokenizer.tokenize(q_and_a)
        tokens_b_context = self.tokenizer.tokenize(context)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label,
                                   answer=answer_ids, tokens_a_context = tokens_a_context,
                                   tokens_b_context = tokens_b_context)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_ids_withoutmask),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.input_mask_withoutmask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.segment_ids_withoutmask),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next),
                       torch.tensor(cur_features.start_positions),
                       torch.tensor(cur_features.end_positions))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2, t3, t4= self.get_corpus_line(index)

        # if random.random() > 0.5:
        #     label = 0
        # else:
        #     t2 = self.get_random_line()
        #     label = 1
        label = 0

        assert len(t1) > 0
        # assert len(t2) > 0
        return t1, t2, t3, t4, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        t3 = ""
        t4 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            token = self.all_docs[sample["doc_id"]][sample["line"]].split("\t")
            t1 = token[0]       # target_triple
            t2 = token[1]       # context
            t3 = token[2]       # question
            t4 = token[3]       # answer
            # t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            # t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]

            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2, t3, t4
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    t3 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                t3 = next(self.file).strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    t3 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        assert t1 != ""
        assert t2 != ""
        assert t3 != ""
        assert t4 != ""
        return t1, t2, t3, t4

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None, answer=None, tokens_a_context=None,
                 tokens_b_context=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.answer = answer
        self.tokens_a_context = tokens_a_context
        self.tokens_b_context = tokens_b_context


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_ids_withoutmask, input_mask, input_mask_withoutmask, segment_ids,
                 segment_ids_withoutmask, is_next, lm_label_ids, start_positions, end_positions):
        self.input_ids = input_ids
        self.input_ids_withoutmask = input_ids_withoutmask
        self.input_mask = input_mask
        self.input_mask_withoutmask = input_mask_withoutmask
        self.segment_ids = segment_ids
        self.segment_ids_withoutmask = segment_ids_withoutmask
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.start_positions = start_positions
        self.end_positions = end_positions


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.get_vocab().items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.get_vocab()[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.get_vocab()["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    # tokens_a_withoutmask = copy.deepcopy(example.tokens_a)
    tokens_a_withoutmask = example.tokens_a_context
    tokens_a_mask = example.tokens_a
    # tokens_b_withoutmask = copy.deepcopy(example.tokens_b)
    tokens_b_withoutmask = example.tokens_b_context
    tokens_b_mask = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a_withoutmask, tokens_b_withoutmask, max_seq_length - 3)
    _truncate_seq_pair(tokens_a_mask, tokens_b_mask, max_seq_length - 3)

    # context prompt

    # mask
    tokens_a_mask, t1_label = random_word(tokens_a_mask, tokenizer)      # question_mask
    tokens_b_mask, t2_label = random_word(tokens_b_mask, tokenizer)      # context_mask
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-100] + t1_label + [-100] + t2_label + [-100])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens_mask = []
    segment_ids = []
    segment_ids_withoutmask = []
    tokens_mask.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a_mask:
        tokens_mask.append(token)
        segment_ids.append(0)
    tokens_mask.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b_mask) > 0
    for token in tokens_b_mask:
        tokens_mask.append(token)
        segment_ids.append(1)
    tokens_mask.append("[SEP]")
    segment_ids.append(1)

    tokens_withoutmask = []
    tokens_withoutmask.append("[CLS]")
    segment_ids_withoutmask.append(0)
    for token in tokens_a_withoutmask:
        tokens_withoutmask.append(token)
        segment_ids_withoutmask.append(0)
    tokens_withoutmask.append("[SEP]")
    segment_ids_withoutmask.append(0)

    assert len(tokens_b_withoutmask) > 0
    for token in tokens_b_withoutmask:
        tokens_withoutmask.append(token)
        segment_ids_withoutmask.append(1)
    tokens_withoutmask.append("[SEP]")
    segment_ids_withoutmask.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens_mask)
    input_ids_withoutmask = tokenizer.convert_tokens_to_ids(tokens_withoutmask)
    first_sep = input_ids_withoutmask.index(tokenizer.sep_token_id)

    answer_pos_start = input_ids_withoutmask.index(example.answer[0], first_sep)
    answer_pos_end = input_ids_withoutmask.index(example.answer[-1], first_sep)

    # print(tokens_withoutmask, example.answer)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    input_mask_withoutmask = [1] * len(input_ids_withoutmask)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-100)

    while len(input_ids_withoutmask) < max_seq_length:
        input_ids_withoutmask.append(0)
        input_mask_withoutmask.append(0)
        segment_ids_withoutmask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_ids_withoutmask) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(input_mask_withoutmask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(segment_ids_withoutmask) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # if example.guid < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #             [str(x) for x in tokens_mask]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("LM label: %s " % (lm_label_ids))
    #     logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_ids_withoutmask = input_ids_withoutmask,
                             input_mask=input_mask,
                             input_mask_withoutmask = input_mask_withoutmask,
                             segment_ids=segment_ids,
                             segment_ids_withoutmask=segment_ids_withoutmask,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             start_positions=answer_pos_start,
                             end_positions=answer_pos_end)
    return features

def save_bert_model(args, model, node_list):
    # -----------------------------加载.bin 模型-------------------------------------
    max_seq_length = args.max_seq_length_node
    eval_batch_size = args.eval_batch_size_node
    # bert_model = "bert-large-uncased"

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)  # 加载Tokenizer
    if args.pretrained_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer

    elif args.pretrained_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer

    elif args.pretrained_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer

    # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    # print("Loading model from %s" % args.output_dir)
    # bert_model = torch.load(output_model_file, map_location='cpu')  # 加载预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # # self.bert_model = BertLayer.load_state_dict(torch.load(output_model_file))
    # # ------------------------------------------------------------------------------
    # bert_model.to(device)
    # model_state_dict = torch.load(output_model_file, map_location=lambda storage, loc: storage.cuda(2))
    # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=2)
    model.to(device)

    print("Computing BERT embeddings..")
    model.eval()

    eval_examples = convert_nodes_to_examples(node_list)
    eval_features = convert_examples_to_features(
        eval_examples, max_seq_length=max_seq_length, tokenizer=tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    sequence_outputs = []
    nums = len(eval_dataloader)

    idx = 0
    i = 0
    for input_ids, input_mask, segment_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            if args.pretrained_model == "bert":
                sequence_output_1= model.bert(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

            elif args.pretrained_model == "albert":
                sequence_output_1= model.albert(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

            elif args.pretrained_model == "roberta":
                sequence_output_1= model.roberta(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            # sequence_output_1= model.bert(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            sequence_output = sequence_output_1[0]
            temp = sequence_output[:, 0]
        sequence_outputs.append(sequence_output[:, 0])

        i += 1
        print("\r" + str(i) + '/' + str(nums), end="")

        # if len(sequence_outputs) == 800:
        #     save_to_disk(args, torch.cat(sequence_outputs, dim=0), idx)
        #     sequence_outputs = []
        #     idx += 1

    save_to_disk(args, torch.cat(sequence_outputs, dim=0), idx)

def save_to_disk(args, tensor, idx):
    filename = os.path.join(args.output_dir, args.dataset+ str(idx) + "_" + args.pretrained_model +"_embeddings_path.pt")
    torch.save(tensor, filename)


def attention_calculate(embeddings, mask_embedding, input_id, file):
    emb_sim = torch.mm(embeddings, mask_embedding.unsqueeze(dim=-1))
    softmax = torch.nn.Softmax(dim=0)
    emb_attention = softmax(emb_sim)

    file.write(str(input_id)+'\n')
    file.write(str(emb_attention)+'\n')
    # emb_attention = 1-softmax(emb_sim)
    # attention_emb = torch.mm(emb_attention.t(), embeddings)
    # return emb_attention

def loss_prompt(loss_func, input_ids_withoutmask, sequence_output, head_id, tail_id, mask_id, args, epoch, file):
    if len(input_ids_withoutmask) == len(sequence_output):
        loss_batch = 0
        for i in range(args.train_batch_size):
            # temp = enumerate(input_ids_withoutmask[i])
            # for index, id in temp:
            #     print(index)
            head_index = [index for index, id in enumerate(input_ids_withoutmask[i]) if id.item() == head_id]
            tail_index = [index for index, id in enumerate(input_ids_withoutmask[i]) if id.item() == tail_id]
            mask_index = input_ids_withoutmask[i].tolist().index(mask_id)
            ans_indices = torch.tensor(head_index).to(device=args.gpu)
            ans_embeddings = torch.index_select(sequence_output[i], 0, ans_indices)
            ans_embedding = torch.mean(ans_embeddings, dim=0, keepdim=True)
            mask_embedding = sequence_output[i][mask_index]
            dis = torch.nn.PairwiseDistance(p=2)
            distance_pos = dis(mask_embedding, ans_embedding)
            neg_index = random.randint(mask_index + 1, len(sequence_output[i])-1)
            neg_embedding = sequence_output[i][neg_index].unsqueeze(0)
            distance_neg = dis(mask_embedding, neg_embedding)
            # loss = loss_func(ans_embedding, mask_embedding, torch.ones(len(ans_embedding)).to(device=args.gpu))
            loss = loss_func(distance_pos, distance_neg, torch.ones(1).to(device=args.gpu))
            loss_batch += loss

            # if epoch == (args.num_train_epochs - 1):
            #     attention_calculate(ans_embeddings, mask_embedding, input_ids_withoutmask[i], file)
        return loss_batch

def train(args, node_list):
    if args.local_rank == -1 or args.no_cuda:  # 未指定gpu，多个gpu/cpu
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # torch.cuda.set_device(args.local_rank)
        # device = torch.device("cuda", args.local_rank)      # args.local_rank指定gpu
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)  # args.local_rank指定gpu
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        os.makedirs(args.output_dir)

    if args.pretrained_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    elif args.pretrained_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer
        model = AlbertForMaskedLM.from_pretrained(args.bert_model)
    elif args.pretrained_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)  # 加载Tokenizer
        model = RobertaForMaskedLM.from_pretrained(args.bert_model)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    characters = {'additional_special_tokens':["[ANS]", "[/ANS]", "[UNK]"]}
    tokenizer.add_special_tokens(characters)


    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_corpus)
        train_dataset = BERTDataset(args.train_corpus, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory)
        # train_dataset.to(device)
        a = len(train_dataset)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.dataset == "conceptnet":
        output_dir = "../bert_model_embeddings/nodes-lm-conceptnet/lm_pytorch_model.bin"
    else:
        output_dir = "../bert_model_embeddings/nodes-lm-atomic/lm_pytorch_model.bin"
    # output_dir = "../pretrained/log_class/pytorch_model.bin"

    # model = BertForMaskedLM.from_pretrained(args.bert_model)

    model_dict = model.state_dict()
    pretrained_model = torch.load(output_dir, map_location=lambda storage, loc: storage.cuda(args.gpu)).state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # model = BertForMaskedLM.from_pretrained(args.bert_model)
    # model = BertForQuestionAnswering.from_pretrained(args.bert_model)
    # print("without pretrained")
    model.resize_token_embeddings(len(tokenizer))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            model.to(device)
        except ImportError:
            raise ImportError(
                "GPU does not exist")
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            # optimizer = FusedAdam(optimizer_grouped_parameters,
            #                       lr=args.learning_rate,
            #                       bias_correction=False,
            #                       max_grad_norm=1.0)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)

    global_step = 0
    # preset_margin = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

        model.train()
        model_params = list(model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        f = open('path_attention.txt', 'w')
        f.write("attention")
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_ids_withoutmask, input_mask, input_mask_wioutmask, segment_ids, segment_ids_withoutmask, lm_label_ids, \
                is_next, start_positions, end_positions = batch
                # temp = input_ids_withoutmask.cpu().numpy()
                if args.pretrained_model == "bert":
                    sequence_output = model.bert(input_ids=input_ids_withoutmask,
                                                 token_type_ids=segment_ids_withoutmask,
                                                 attention_mask=input_mask_wioutmask).last_hidden_state

                elif args.pretrained_model == "albert":
                    sequence_output = model.albert(input_ids=input_ids_withoutmask,
                                                   token_type_ids=segment_ids_withoutmask,
                                                   attention_mask=input_mask_wioutmask).last_hidden_state

                elif args.pretrained_model == "roberta":
                    sequence_output = model.roberta(input_ids=input_ids_withoutmask,
                                                    token_type_ids=segment_ids_withoutmask,
                                                    attention_mask=input_mask_wioutmask).last_hidden_state
                # sequence_output = model.bert(input_ids=input_ids_withoutmask, token_type_ids=segment_ids_withoutmask,
                #                              attention_mask=input_mask_wioutmask).last_hidden_state
                # sequence_output = model.bert(input_ids=input_ids_withoutmask).last_hidden_state
                loss_func = torch.nn.MarginRankingLoss(margin=args.margin, reduction='mean')
                head_id = tokenizer.encode("[ANS]", add_special_tokens=False)[0]
                tail_id = tokenizer.encode("[/ANS]", add_special_tokens=False)[0]
                mask_id = tokenizer.encode("[MASK]", add_special_tokens=False)[0]


                loss_path = loss_prompt(loss_func, input_ids_withoutmask, sequence_output, head_id, tail_id, mask_id, args,
                                        epoch, f)

                # outputs = model(input_ids = input_ids_withoutmask, token_type_ids=segment_ids,
                #                 start_positions = start_positions, end_positions = end_positions)
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=lm_label_ids)
                # outputs = model(input_ids, segment_ids, input_mask)
                loss_mask = outputs.loss
                loss = args.lambda_path * loss_path + args.lambda_mask * loss_mask
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids_withoutmask.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
        save_bert_model(args, model, node_list)
        # Save a trained model
        if args.do_train:
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    train()
