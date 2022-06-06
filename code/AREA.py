"""Run QA model to extract arguments."""

# from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert_self.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert_self.modeling_features import BertForQuestionAnswering, BertForQuestionAnswering_withIfTriggerEmbedding_golden_args, \
    BertForQuestionAnswering_withIfTriggerEmbedding_golden_args_distill
from pytorch_pretrained_bert_self.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert_self.tokenization import BertTokenizer
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

max_seq_length = 180


class AceExample(object):
    """
    A single training/test example for the ace dataset.
    """

    def __init__(self, sentence, events, s_start):
        self.sentence = sentence
        self.events = events
        self.s_start = s_start

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "event sentence: %s" % (" ".join(self.sentence))
        event_triggers = []
        for event in self.events:
            if event:
                event_triggers.append(self.sentence[event[0][0] - self.s_start])
                event_triggers.append(event[0][1])
                event_triggers.append(str(event[0][0] - self.s_start))
                event_triggers.append("|")
        s += " ||| event triggers: %s" % (" ".join(event_triggers))
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens, token_to_orig_map, input_ids, input_mask, segment_ids, if_trigger_ids,
                 #
                 event_type, argument_type, fea_trigger_offset,
                 #
                 start_position=None, end_position=None, seg_idxs=None, golden_argument=None):
        self.example_id = example_id
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.if_trigger_ids = if_trigger_ids

        self.event_type = event_type
        self.argument_type = argument_type
        self.fea_trigger_offset = fea_trigger_offset

        self.start_position = start_position
        self.end_position = end_position
        self.seg_idxs = seg_idxs
        self.golden_argument = golden_argument


def read_ace_examples(input_file, is_training):
    """Read a ACE json file into a list of AceExample."""
    examples = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            sentence, events, s_start = example["sentence"], example["event"], example["s_start"]
            example = AceExample(sentence=sentence, events=events, s_start=s_start)
            examples.append(example)

    return examples

def preprocess_input(tokenizer, example, teach_query):
    # teacher
    query_tokens = tokenizer.tokenize(teach_query)
    context_tokens = example.sentence
    d_tokens = []
    for i, word in enumerate(context_tokens):
        sub_tokens = tokenizer.tokenize(word)
        # TODO：遇到了奇怪的字符如，就不管
        # TODO：只要trigger第一个词，要改！！
        if sub_tokens:
            d_tokens.append(sub_tokens[0])
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    if len(d_tokens) >= max_tokens_for_doc:
        d_tokens = d_tokens[0:max_tokens_for_doc]

    # prepare [CLS] query [SEP] sentence [SEP]
    seg_idxs = []
    tokens = []
    segment_ids = []
    token_to_orig_map = {}
    # add [CLS]
    tokens.append("[CLS]")
    segment_ids.append(0)
    # add query
    query_tokens = tokenizer.tokenize(teach_query)
    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
    # add [SEP]
    tokens.append("[SEP]")
    segment_ids.append(0)
    seg_idxs.append(len(segment_ids))

    # add sentence
    for (i, token) in enumerate(d_tokens):
        token_to_orig_map[len(tokens)] = i
        tokens.append(token)
        segment_ids.append(1)

    seg_idxs.append(len(segment_ids))
    # add [SEP]
    tokens.append("[SEP]")
    segment_ids.append(1)
    # transform to input_ids ...
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    if len(input_ids) != 512:
        print(example.sentence)
        print(len(input_ids))
        print(input_ids)
    assert len(input_ids) == 512
    assert len(input_mask) == 512
    assert len(segment_ids) == 512
    return input_ids, input_mask, segment_ids, query_tokens, tokens, token_to_orig_map, seg_idxs

def convert_examples_to_features_t_s(examples, tokenizer, query_templates, t_nth_query, s_nth_query, is_training, args,
                                     student_external=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    max_seq_length = args.max_seq_length

    for (example_id, example) in enumerate(examples):
        for event in example.events:
            trigger_offset = event[0][0] - example.s_start
            event_type = event[0][1]
            trigger_token = example.sentence[trigger_offset]
            arguments = event[1:]
            for argument_type in query_templates[event_type]:

                query = query_templates[event_type][argument_type][s_nth_query]
                query = query.replace("[trigger]", trigger_token)

                teach_query = query_templates[event_type][argument_type][t_nth_query]
                teach_query = teach_query.replace("[trigger]", trigger_token)

                # query增加其他所有信息
                for arg in arguments:
                    gold_arg_type = arg[2]
                    if gold_arg_type == argument_type:
                        # 排除自己,选择其他所有
                        rest_args = [j for j in arguments if j!=arg]
                        if rest_args:
                            rest_types = [j[2] for j in rest_args]
                            starts = [j[0] for j in rest_args]
                            ends = [j[1] for j in rest_args]
                            context_tokens = example.sentence
                            word_list = []
                            for start, end in zip(starts, ends):
                                word_list.append(context_tokens[start:end+1])
                            sent_list = []
                            for word, type in zip(word_list, rest_types):
                                sent_list.append(' '.join(word) + ' as ' + type)
                            sent_out = ' and '.join(sent_list)
                            teach_query = teach_query.replace('[args]', sent_out)
                            if student_external:
                                query = query.replace('[args]', sent_out)
                            else:
                                query = query.replace('[args]', 'None')
                        else:
                            teach_query = teach_query.replace('[args]', 'None')
                            query = query.replace('[args]', 'None')

                # 当模板里的argument不在ground truth里时
                if '[args]' in teach_query:
                    teach_query = teach_query.replace('[args]', 'None')
                if '[args]' in query:
                    query = query.replace('[args]', 'None')

                # print('query', query)
                # print('teach_query', teach_query)

                # teacher
                t_input_ids, t_input_mask, t_segment_ids, \
                t_query_tokens, t_tokens, t_token_to_orig_map, t_seg_idxs = preprocess_input(tokenizer, example, teach_query)
                # student
                s_input_ids, s_input_mask, s_segment_ids, \
                s_query_tokens, s_tokens, s_token_to_orig_map, s_seg_idxs = preprocess_input(tokenizer, example, query)

                # teacher
                # start & end position
                t_start_position, t_end_position = None, None

                sentence_start = example.s_start
                t_sentence_offset = len(t_query_tokens) + 2
                t_fea_trigger_offset = trigger_offset + t_sentence_offset

                t_if_trigger_ids = [0] * len(t_segment_ids)
                t_if_trigger_ids[t_fea_trigger_offset] = 1

                # the golden argument positions in teacher
                t_golden_arguments = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
                for arg_i, arg in enumerate(arguments):
                    t_golden_arguments[arg_i] = [arg[0] + t_sentence_offset, arg[1] + t_sentence_offset]

                # student
                # start & end position
                s_start_position, s_end_position = None, None

                # sentence_start = example.s_start
                s_sentence_offset = len(s_query_tokens) + 2
                s_fea_trigger_offset = trigger_offset + s_sentence_offset

                s_if_trigger_ids = [0] * len(s_segment_ids)
                s_if_trigger_ids[s_fea_trigger_offset] = 1

                # the golden argument positions in student
                s_golden_arguments = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
                if student_external:
                    for arg_i, arg in enumerate(arguments):
                        s_golden_arguments[arg_i] = [arg[0] + s_sentence_offset, arg[1] + s_sentence_offset]

                if is_training:
                    no_answer = True
                    for argument in arguments:
                        gold_argument_type = argument[2]
                        if gold_argument_type == argument_type:
                            no_answer = False
                            answer_start, answer_end = argument[0], argument[1]

                            t_start_position = answer_start - sentence_start + t_sentence_offset
                            t_end_position = answer_end - sentence_start + t_sentence_offset

                            s_start_position = answer_start - sentence_start + s_sentence_offset
                            s_end_position = answer_end - sentence_start + s_sentence_offset
                            features.append(
                                [InputFeatures(example_id=example_id, tokens=t_tokens, token_to_orig_map=t_token_to_orig_map,
                                              input_ids=t_input_ids, input_mask=t_input_mask, segment_ids=t_segment_ids,
                                              if_trigger_ids=t_if_trigger_ids,
                                              event_type=event_type, argument_type=argument_type,
                                              fea_trigger_offset=t_fea_trigger_offset,
                                              start_position=t_start_position, end_position=t_end_position,
                                              seg_idxs=t_seg_idxs, golden_argument=t_golden_arguments),
                                 InputFeatures(example_id=example_id, tokens=s_tokens,
                                               token_to_orig_map=s_token_to_orig_map,
                                               input_ids=s_input_ids, input_mask=s_input_mask,
                                               segment_ids=s_segment_ids,
                                               if_trigger_ids=s_if_trigger_ids,
                                               event_type=event_type, argument_type=argument_type,
                                               fea_trigger_offset=s_fea_trigger_offset,
                                               start_position=s_start_position, end_position=s_end_position,
                                               seg_idxs=s_seg_idxs, golden_argument=s_golden_arguments)
                                ]
                            )
                    if no_answer:
                        t_start_position, t_end_position = 0, 0
                        s_start_position, s_end_position = 0, 0
                        features.append(
                            [InputFeatures(example_id=example_id, tokens=t_tokens, token_to_orig_map=t_token_to_orig_map,
                                              input_ids=t_input_ids, input_mask=t_input_mask, segment_ids=t_segment_ids,
                                              if_trigger_ids=t_if_trigger_ids,
                                              event_type=event_type, argument_type=argument_type,
                                              fea_trigger_offset=t_fea_trigger_offset,
                                              start_position=t_start_position, end_position=t_end_position,
                                           seg_idxs=t_seg_idxs, golden_argument=t_golden_arguments),
                                 InputFeatures(example_id=example_id, tokens=s_tokens,
                                               token_to_orig_map=s_token_to_orig_map,
                                               input_ids=s_input_ids, input_mask=s_input_mask,
                                               segment_ids=s_segment_ids,
                                               if_trigger_ids=s_if_trigger_ids,
                                               event_type=event_type, argument_type=argument_type,
                                               fea_trigger_offset=s_fea_trigger_offset,
                                               start_position=s_start_position, end_position=s_end_position,
                                               seg_idxs=s_seg_idxs, golden_argument=s_golden_arguments)
                        ])
                else:
                    features.append(
                        [InputFeatures(example_id=example_id, tokens=t_tokens,
                                       token_to_orig_map=t_token_to_orig_map,
                                       input_ids=t_input_ids, input_mask=t_input_mask,
                                       segment_ids=t_segment_ids,
                                       if_trigger_ids=t_if_trigger_ids,
                                       event_type=event_type, argument_type=argument_type,
                                       fea_trigger_offset=t_fea_trigger_offset,
                                       start_position=t_start_position, end_position=t_end_position,
                                       seg_idxs=t_seg_idxs, golden_argument=t_golden_arguments),
                         InputFeatures(example_id=example_id, tokens=s_tokens,
                                       token_to_orig_map=s_token_to_orig_map,
                                       input_ids=s_input_ids, input_mask=s_input_mask,
                                       segment_ids=s_segment_ids,
                                       if_trigger_ids=s_if_trigger_ids,
                                       event_type=event_type, argument_type=argument_type,
                                       fea_trigger_offset=s_fea_trigger_offset,
                                       start_position=s_start_position, end_position=s_end_position,
                                       seg_idxs=s_seg_idxs, golden_argument=s_golden_arguments)
                        ]
                    )

    return features

def convert_examples_to_features(examples, tokenizer, query_templates, nth_query, is_training, args):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    max_seq_length = args.max_seq_length

    for (example_id, example) in enumerate(examples):
        for event in example.events:
            trigger_offset = event[0][0] - example.s_start
            event_type = event[0][1]
            trigger_token = example.sentence[trigger_offset]
            arguments = event[1:]
            for argument_type in query_templates[event_type]:

                query = query_templates[event_type][argument_type][nth_query]
                query = query.replace("[trigger]", trigger_token)

                query_tokens = tokenizer.tokenize(query)
                context_tokens = example.sentence
                d_tokens = []
                for i, word in enumerate(context_tokens):
                    sub_tokens = tokenizer.tokenize(word)
                    if sub_tokens:
                        d_tokens.append(sub_tokens[0])
                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
                if len(d_tokens) >= max_tokens_for_doc:
                    d_tokens = d_tokens[0:max_tokens_for_doc]

                # prepare [CLS] query [SEP] sentence [SEP]
                tokens = []
                segment_ids = []
                token_to_orig_map = {}
                # add [CLS]
                tokens.append("[CLS]")
                segment_ids.append(0)
                # add query
                query_tokens = tokenizer.tokenize(query)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                # add [SEP]
                tokens.append("[SEP]")
                segment_ids.append(0)

                # add sentence
                for (i, token) in enumerate(d_tokens):
                    token_to_orig_map[len(tokens)] = i
                    tokens.append(token)
                    segment_ids.append(1)

                # add [SEP]
                tokens.append("[SEP]")
                segment_ids.append(1)
                # transform to input_ids ...
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                if len(input_ids) != 512:
                    print(example.sentence)
                    print(len(input_ids))
                    print(input_ids)
                assert len(input_ids) == 512
                assert len(input_mask) == 512
                assert len(segment_ids) == 512

                # start & end position
                start_position, end_position = None, None

                sentence_start = example.s_start
                sentence_offset = len(query_tokens) + 2
                fea_trigger_offset = trigger_offset + sentence_offset

                if_trigger_ids = [0] * len(segment_ids)
                if_trigger_ids[fea_trigger_offset] = 1

                if is_training:
                    no_answer = True
                    for argument in arguments:
                        gold_argument_type = argument[2]
                        if gold_argument_type == argument_type:
                            no_answer = False
                            answer_start, answer_end = argument[0], argument[1]

                            start_position = answer_start - sentence_start + sentence_offset
                            end_position = answer_end - sentence_start + sentence_offset
                            features.append(
                                InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map,
                                              input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                              if_trigger_ids=if_trigger_ids,
                                              event_type=event_type, argument_type=argument_type,
                                              fea_trigger_offset=fea_trigger_offset,
                                              start_position=start_position, end_position=end_position))
                    if no_answer:
                        start_position, end_position = 0, 0
                        features.append(
                            InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map,
                                          input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                          if_trigger_ids=if_trigger_ids,
                                          event_type=event_type, argument_type=argument_type,
                                          fea_trigger_offset=fea_trigger_offset,
                                          start_position=start_position, end_position=end_position))
                else:
                    features.append(
                        InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map,
                                      input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      if_trigger_ids=if_trigger_ids,
                                      event_type=event_type, argument_type=argument_type,
                                      fea_trigger_offset=fea_trigger_offset,
                                      start_position=start_position, end_position=end_position))
    return features

def read_query_templates(normal_file, des_file):
    """Load query templates"""
    query_templates = dict()
    with open(normal_file, "r", encoding='utf-8') as f:
        for line in f:
            # event_arg, query = line.strip().split(",")
            event_arg = line.strip()
            event_type, arg_name = event_arg.split("_")
            event_words = event_type.split('.')

            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if arg_name not in query_templates[event_type]:
                query_templates[event_type][arg_name] = list()

            # 0 template arg_name
            query_templates[event_type][arg_name].append(arg_name)
            # 1 template event_type + arg_name + external knowledge + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(
                event_type + ' ' + arg_name + ' with external knowledge [args]' + " in [trigger]")
            # query_templates[event_type][arg_name].append('event type : ' +' '.join(event_words) + ' and argument name : ' + arg_name + ' , with external knowledge [args]' + " in [trigger]")

            # 2 template event_type + arg_name + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(
                event_type + ' ' + arg_name + " in [trigger]")

    #         # 2 template arg_query
    #         query_templates[event_type][arg_name].append(query)
    #         # 3 arg_query + trigger (replace [trigger] when forming the instance)
    #         query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")
    #
    # with open(des_file, "r", encoding='utf-8') as f:
    #     for line in f:
    #         event_arg, query = line.strip().split(",")
    #         event_type, arg_name = event_arg.split("_")
    #         # 4 template des_query
    #         query_templates[event_type][arg_name].append(query)
    #         # 5 template des_query + trigger (replace [trigger] when forming the instance)
    #         query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")
    #
    # for event_type in query_templates:
    #     for arg_name in query_templates[event_type]:
    #         assert len(query_templates[event_type][arg_name]) == 6

    return query_templates


RawResult = collections.namedtuple("RawResult",
                                   ["example_id", "event_type_offset_argument_type", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, larger_than_cls):
    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_id].append(feature)
    example_id_to_results = collections.defaultdict(list)
    for result in all_results:
        example_id_to_results[result.example_id].append(result)
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    final_all_predictions = collections.OrderedDict()
    # all_nbest_json = collections.OrderedDict()
    # scores_diff_json = collections.OrderedDict()

    for (example_id, example) in enumerate(all_examples):
        features = example_id_to_features[example_id]
        results = example_id_to_results[example_id]
        all_predictions[example_id] = collections.OrderedDict()
        final_all_predictions[example_id] = []
        for (feature_index, feature) in enumerate(features):
            event_type_argument_type = "_".join([feature.event_type, feature.argument_type])
            event_type_offset_argument_type = "_".join(
                [feature.event_type, str(feature.token_to_orig_map[feature.fea_trigger_offset]), feature.argument_type])

            start_indexes, end_indexes = None, None
            prelim_predictions = []
            for result in results:
                if result.event_type_offset_argument_type == event_type_offset_argument_type:
                    start_indexes = _get_best_indexes(result.start_logits, n_best_size, larger_than_cls,
                                                      result.start_logits[0])
                    end_indexes = _get_best_indexes(result.end_logits, n_best_size, larger_than_cls,
                                                    result.end_logits[0])
                    # add span preds
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):
                                continue
                            if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > max_answer_length:
                                continue
                            prelim_predictions.append(
                                _PrelimPrediction(start_index=start_index, end_index=end_index,
                                                  start_logit=result.start_logits[start_index],
                                                  end_logit=result.end_logits[end_index]))

                    ## add null pred
                    if not larger_than_cls:
                        feature_null_score = result.start_logits[0] + result.end_logits[0]
                        prelim_predictions.append(
                            _PrelimPrediction(start_index=0, end_index=0,
                                              start_logit=result.start_logits[0], end_logit=result.end_logits[0]))

                    ## sort
                    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit),
                                                reverse=True)

                    # all_predictions[example_id][event_type_offset_argument_type] = prelim_predictions

                    ## get final pred in format: [event_type_offset_argument_type, [start_offset, end_offset]]
                    max_num_pred_per_arg = 1
                    for idx, pred in enumerate(prelim_predictions):
                        if (idx + 1) > max_num_pred_per_arg: break
                        if pred.start_index == 0 and pred.end_index == 0: break
                        orig_sent_start, orig_sent_end = feature.token_to_orig_map[pred.start_index], \
                                                         feature.token_to_orig_map[pred.end_index]
                        final_all_predictions[example_id].append(
                            [event_type_argument_type, [orig_sent_start, orig_sent_end]])

    return final_all_predictions


def _get_best_indexes(logits, n_best_size=1, larger_than_cls=False, cls_logit=None):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        if larger_than_cls:
            if index_and_score[i][1] < cls_logit:
                break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def find_best_thresh(new_preds, new_all_gold):
    best_score = 0
    best_na_thresh = 0
    gold_arg_n, pred_arg_n = len(new_all_gold), 0

    candidate_preds = []
    for argument in new_preds:
        candidate_preds.append(argument[:-2] + argument[-1:])
        pred_arg_n += 1

        pred_in_gold_n, gold_in_pred_n = 0, 0
        # pred_in_gold_n
        for argu in candidate_preds:
            if argu in new_all_gold:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argu in new_all_gold:
            if argu in candidate_preds:
                gold_in_pred_n += 1

        prec_c, recall_c, f1_c = 0, 0, 0
        if pred_arg_n != 0: prec_c = 100.0 * pred_in_gold_n / pred_arg_n
        else: prec_c = 0
        if gold_arg_n != 0: recall_c = 100.0 * gold_in_pred_n / gold_arg_n
        else: recall_c = 0
        if prec_c or recall_c: f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
        else: f1_c = 0

        if f1_c > best_score:
            best_score = f1_c
            best_na_thresh = argument[-2]

    # import ipdb; ipdb.set_trace()
    return best_na_thresh + 1e-10

def evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features, na_prob_thresh=1.0,
             pred_only=False):
    all_results = []
    model.eval()
    for idx, (input_ids, input_mask, segment_ids, if_trigger_ids, example_indices) in enumerate(eval_dataloader):
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        if_trigger_ids = if_trigger_ids.to(device)
        with torch.no_grad():
            if not args.add_if_trigger_embedding:
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            else:
                # batch_start_logits, batch_end_logits = model(input_ids, segment_ids, if_trigger_ids, input_mask)
                batch_start_logits, batch_end_logits = model(input_ids=input_ids, if_trigger_ids=if_trigger_ids,
                                                             token_type_ids=segment_ids, attention_mask=input_mask,
                                                             )
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            example_id = eval_feature.example_id
            event_type_offset_argument_type = "_".join(
                [eval_feature.event_type, str(eval_feature.token_to_orig_map[eval_feature.fea_trigger_offset]),
                 eval_feature.argument_type])
            all_results.append(
                RawResult(example_id=example_id, event_type_offset_argument_type=event_type_offset_argument_type,
                          start_logits=start_logits, end_logits=end_logits))

    # preds, nbest_preds, na_probs = \
    preds = make_predictions(eval_examples, eval_features, all_results,
                             args.n_best_size, args.max_answer_length, args.larger_than_cls)
    preds_init = copy.deepcopy(preds)

    # get all_gold in format: [event_type_argument_type, [start_offset, end_offset]]
    all_gold = collections.OrderedDict()
    for (example_id, example) in enumerate(gold_examples):
        all_gold[example_id] = []
        for event in example.events:
            # if not event: continue
            trigger_offset = event[0][0] - example.s_start
            event_type = event[0][1]
            for argument in event[1:]:
                argument_start, argument_end, argument_type = argument[0] - example.s_start, argument[
                    1] - example.s_start, argument[2]
                # event_type_offset_argument_type = "_".join([event_type, str(trigger_offset), argument_type])
                event_type_argument_type = "_".join([event_type, argument_type])
                all_gold[example_id].append([event_type_argument_type, [argument_start, argument_end]])

    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for (example_id, _) in enumerate(gold_examples):
        pred_arg = preds[example_id]
        gold_arg = all_gold[example_id]
        # pred_arg_n
        for argument in pred_arg: pred_arg_n += 1
        # gold_arg_n
        for argument in gold_arg: gold_arg_n += 1
        # pred_in_gold_n
        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_arg_n != 0:
        prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_c = 0
    if gold_arg_n != 0:
        recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0

    # get results (identification)
    for (example_id, _) in enumerate(gold_examples):
        for argument in preds[example_id]:
            argument[0] = argument[0].split("_")[0]  # only event_type
        for argument in all_gold[example_id]:
            argument[0] = argument[0].split("_")[0]  # only event_type

    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for (example_id, _) in enumerate(gold_examples):
        pred_arg = preds[example_id]
        gold_arg = all_gold[example_id]
        # pred_arg_n
        for argument in pred_arg: pred_arg_n += 1
        # gold_arg_n
        for argument in gold_arg: gold_arg_n += 1
        # pred_in_gold_n
        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    prec_i, recall_i, f1_i = 0, 0, 0
    if pred_arg_n != 0:
        prec_i = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_i = 0
    if gold_arg_n != 0:
        recall_i = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_i = 0
    if prec_i or recall_i:
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else:
        f1_i = 0

    # # logging for DEBUG results
    # for (example_id, example) in enumerate(gold_examples):
    #     if example_id > 20: break
    #     if preds[example_id] or all_gold[example_id]:
    #         token_idx = []
    #         for idx, token in enumerate(example.sentence): token_idx.append(" ".join([token, str(idx)]))
    #         logger.info("sent: {}".format(" | ".join(token_idx)))

    #         gold_str_list = []
    #         for gold in all_gold[example_id]: gold_str_list.append(" ".join([gold[0], str(gold[1][0]), str(gold[1][1])]))
    #         logger.info("gold: {}".format(" | ".join(gold_str_list)))

    #         pred_str_list = []
    #         for pred in preds[example_id]: pred_str_list.append(" ".join([pred[0], str(pred[1][0]), str(pred[1][1])]))
    #         logger.info("pred: {}".format(" | ".join(pred_str_list)))

    #         logger.info("\n")

    result = collections.OrderedDict()
    result = collections.OrderedDict(
        [('prec_c', prec_c), ('recall_c', recall_c), ('f1_c', f1_c), ('prec_i', prec_i), ('recall_i', recall_i),
         ('f1_i', f1_i)])
    return result, preds_init


def init_config(args,output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(output_dir, "eval.log"), 'w'))
    logger.info(args)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return n_gpu, device


def softmax_mask(s,e, seg_mask, T):
    s_1 = F.softmax(s / T, dim=1)
    e_1 = F.softmax(e / T, dim=1)
    p_s = s_1 * seg_mask
    p_e = e_1 * seg_mask

    p_s_new = torch.zeros_like(p_s)
    p_e_new = torch.zeros_like(p_e)
    for i in range(p_s.shape[0]):
        p_s_new[i] = p_s[i] / torch.sum(p_s, dim=1)[i]
        p_e_new[i] = p_e[i] / torch.sum(p_e, dim=1)[i]

    return p_s_new, p_e_new


def log_softmax_mask(s,e, seg_mask, T):
    s_1 = F.softmax(s / T, dim=1)
    e_1 = F.softmax(e / T, dim=1)
    p_s = s_1 * seg_mask
    p_e = e_1 * seg_mask

    zero = torch.zeros_like(p_s)
    one = torch.ones_like(p_s)

    p_s_new = torch.zeros_like(p_s)
    p_e_new = torch.zeros_like(p_e)
    for i in range(p_s.shape[0]):
        p_s_new[i] = p_s[i] / torch.sum(p_s, dim=1)[i]
        p_e_new[i] = p_e[i] / torch.sum(p_e, dim=1)[i]
    p_s_new = torch.where(p_s_new == 0, one, p_s_new)
    p_e_new = torch.where(p_e_new == 0, one, p_e_new)
    # p_s_new, p_e_new = softmax_mask(s,e, seg_mask, T)

    p_s_new1 = torch.log(p_s_new)
    p_e_new1 = torch.log(p_e_new)

    return p_s_new1, p_e_new1


def get_distance_loss(student_embedding, teacher_embedding, t_arg_mask_metrix, s_arg_mask_metrix):
    s_arg_mask_metrix = s_arg_mask_metrix.bool()

    mean_t = teacher_embedding[teacher_embedding>0].mean()
    teacher_embedding = teacher_embedding / mean_t
    ## TODO 因为教师网络要和学生网络蒸馏的知识一致
    teacher_embedding = torch.masked_select(teacher_embedding, s_arg_mask_metrix)

    if student_embedding != torch.Size([]):
        mean_s = student_embedding[student_embedding > 0].mean()
        student_embedding = student_embedding / mean_s
        student_embedding = torch.masked_select(student_embedding, s_arg_mask_metrix)

        loss = F.smooth_l1_loss(student_embedding, teacher_embedding, reduction='mean')
        return loss

    else:
        return torch.zeros(1).cuda()


def get_relation_distill_loss(t_start_dist1, t_end_dist1,
                              s_start_dist, s_end_dist, t_arg_mask_metrix, s_arg_mask_metrix):
    start_relation_loss1 = get_distance_loss(s_start_dist, t_start_dist1, t_arg_mask_metrix, s_arg_mask_metrix)
    end_relation_loss1 = get_distance_loss(s_end_dist, t_end_dist1, t_arg_mask_metrix, s_arg_mask_metrix)
    relation_dis_loss1 = (start_relation_loss1 + end_relation_loss1) / 2

    return relation_dis_loss1



def get_angle_loss(student_embedding, teacher_embedding, t_arg_mask_metrix, s_arg_mask_metrix):
    s_arg_mask_metrix = s_arg_mask_metrix.bool()

    teacher_embedding = torch.masked_select(teacher_embedding, s_arg_mask_metrix)
    student_embedding = torch.masked_select(student_embedding, s_arg_mask_metrix)

    if student_embedding != torch.Size([]):
        loss = F.smooth_l1_loss(student_embedding, teacher_embedding, reduction='mean')
        return loss

    else:
        return torch.zeros(1).cuda()


def get_relation_distill_loss_angle(t_start_dist1, t_end_dist1,
                                    s_start_dist, s_end_dist, t_arg_mask_metrix, s_arg_mask_metrix):
    start_relation_loss1 = get_angle_loss(s_start_dist, t_start_dist1, t_arg_mask_metrix, s_arg_mask_metrix)
    end_relation_loss1 = get_angle_loss(s_end_dist, t_end_dist1, t_arg_mask_metrix, s_arg_mask_metrix)
    relation_dis_loss1 = (start_relation_loss1 + end_relation_loss1) / 2

    return relation_dis_loss1


def get_MMD_loss(t_arg_features, s_arg_features, s_arg_mask_metrix):
    # batch,5,768
    t_arg_features_mean = torch.mean(t_arg_features, dim=-1)  # batch,5
    t_arg_features_metrix = torch.abs(t_arg_features_mean.unsqueeze(2) - t_arg_features_mean.unsqueeze(1)) # batch,5, 5
    t_arg_features_metrix = t_arg_features_metrix * s_arg_mask_metrix.bool()
    t_arg_features_metrix = t_arg_features_metrix.view(t_arg_features_metrix.shape[0],-1)

    s_arg_features_mean = torch.mean(s_arg_features, dim=-1)  # batch,5
    s_arg_features_metrix = torch.abs(s_arg_features_mean.unsqueeze(2) - s_arg_features_mean.unsqueeze(1))  # batch,5, 5
    s_arg_features_metrix = s_arg_features_metrix * s_arg_mask_metrix.bool()
    s_arg_features_metrix = s_arg_features_metrix.view(s_arg_features_metrix.shape[0],-1)

    T = args.temperature
    log_s = F.log_softmax(s_arg_features_metrix, dim=-1)
    p_t = F.softmax(t_arg_features_metrix, dim=-1)
    MMD_kl_s = F.kl_div(log_s, p_t, reduction='batchmean') * (T ** 2) / t_arg_features.shape[0]

    return MMD_kl_s


def get_relation_distill_loss_MMD(t_arg_features1,
                                  s_arg_features,
                                  s_arg_mask_metrix
                                  ):
    start_MMD_loss1 = get_MMD_loss(t_arg_features1, s_arg_features, s_arg_mask_metrix)
    return start_MMD_loss1



def get_bilinear_loss(t_arg_features1, s_arg_features, s_arg_mask_metrix):

    t_bilinear_matrix = torch.bmm(t_arg_features1, t_arg_features1.permute(0,2,1)).view(t_arg_features1.shape[0],-1)
    # mask掉不需要蒸馏的地方
    t_bilinear_matrix = t_bilinear_matrix * s_arg_mask_metrix.view(t_arg_features1.shape[0],-1).bool()
    s_bilinear_matrix = torch.bmm(s_arg_features, s_arg_features.permute(0,2,1)).view(s_arg_features.shape[0],-1)  # batch,25
    s_bilinear_matrix = s_bilinear_matrix * s_arg_mask_metrix.view(t_arg_features1.shape[0],-1).bool()

    T = args.temperature
    log_s = F.log_softmax(s_bilinear_matrix, dim=-1)
    p_t = F.softmax(t_bilinear_matrix, dim=-1)
    bilinear_kl_s = F.kl_div(log_s, p_t, reduction='batchmean') * (T ** 2) / t_bilinear_matrix.shape[0]
    return bilinear_kl_s


def get_relation_distill_loss_bilinear(t_arg_features1,
                                       s_arg_features,s_arg_mask_metrix):
    start_bilinear_loss1 = get_bilinear_loss(t_arg_features1, s_arg_features,s_arg_mask_metrix)
    return start_bilinear_loss1


def get_RBF_loss(t_arg_features1, s_arg_features, s_arg_mask_metrix):
    # batch,5,768
    segma = torch.Tensor([0.4]).cuda()
    t_RBF_matrix = torch.bmm(t_arg_features1, t_arg_features1.permute(0, 2, 1))  # batch,5,5
    t_distance = torch.exp(-2 *segma) * (1 + (2*segma) * t_RBF_matrix + 2* segma.pow(2) * t_RBF_matrix.pow(2))  # batch,5,5
    t_distance = t_distance * s_arg_mask_metrix.bool()
    t_distance = t_distance.view(t_distance.shape[0],-1)

    s_RBF_matrix = torch.bmm(s_arg_features, s_arg_features.permute(0, 2, 1))  # batch,5,5
    s_distance = torch.exp(-2 *segma) * (1 + (2*segma) * s_RBF_matrix + 2* segma.pow(2) * s_RBF_matrix.pow(2))  # batch,5,5
    s_distance = s_distance * s_arg_mask_metrix.bool()
    s_distance = s_distance.view(s_distance.shape[0],-1)

    T = args.temperature
    log_s = F.log_softmax(s_distance, dim=-1)
    p_t = F.softmax(t_distance, dim=-1)
    RBF_kl_s = F.kl_div(log_s, p_t, reduction='batchmean') * (T ** 2) / t_RBF_matrix.shape[0]
    return RBF_kl_s


def get_relation_distill_loss_RBF(t_arg_features1,
                                  s_arg_features, s_arg_mask_metrix):
    start_RBF_loss1 = get_RBF_loss(t_arg_features1, s_arg_features, s_arg_mask_metrix)
    return start_RBF_loss1


def distillation(y_s, y_e, y_s_masked, y_e_masked, teacher_scores_s1,teacher_scores_e1,
                labels_s, labels_e, args, t_seg_mask_new, s_seg_mask_new,
                t_start_dist1, t_end_dist1,
                 s_start_dist, s_end_dist, t_arg_mask_metrix, s_arg_mask_metrix,
                t_arg_features1,
                 s_arg_features,
                distance_metric
                 ):

    T = args.temperature
    alpha = args.alpha

    p_s, p_e = log_softmax_mask(y_s_masked, y_e_masked, s_seg_mask_new, T)
    q_s1, q_e1 = softmax_mask(teacher_scores_s1, teacher_scores_e1, t_seg_mask_new, T)

    l_kl_s1 = F.kl_div(p_s, q_s1, reduction='batchmean') * (T**2) / y_s.shape[0]
    l_kl_e1 = F.kl_div(p_e, q_e1, reduction='batchmean') * (T**2) / y_e.shape[0]

    l_ce_s = F.cross_entropy(y_s, labels_s)
    l_ce_e = F.cross_entropy(y_e, labels_e)

    if distance_metric=='distance':
        relation_distill_loss = get_relation_distill_loss(t_start_dist1, t_end_dist1,
                                                          s_start_dist, s_end_dist,
                                                          t_arg_mask_metrix, s_arg_mask_metrix
                                                          )
    elif distance_metric=='angle':
        relation_distill_loss = get_relation_distill_loss_angle(t_start_dist1, t_end_dist1,
                                                                s_start_dist, s_end_dist,
                                                          t_arg_mask_metrix, s_arg_mask_metrix
                                                          )

    elif distance_metric=='MMD':
        relation_distill_loss = get_relation_distill_loss_MMD(t_arg_features1,
                                                              s_arg_features,
                                                              s_arg_mask_metrix
                                                                )

    elif distance_metric=='bilinear':
        relation_distill_loss = get_relation_distill_loss_bilinear(t_arg_features1,
                                                                   s_arg_features,
                                                                   s_arg_mask_metrix
                                                                )
    elif distance_metric=='RBF':
        relation_distill_loss = get_relation_distill_loss_RBF(t_arg_features1,
                                                              s_arg_features,
                                                              s_arg_mask_metrix
                                                                )

    # 当teacher不包括oracle knowledge的时候，则不进行RKD
    if args.arg_proportion == 0:
        relation_distill_loss = 0

    return ((l_kl_s1 + l_kl_e1)/2.0 * args.model1_alpha ) * alpha \
           + (l_ce_s + l_ce_e)/2.0 * (1.0 - alpha) + args.beta * relation_distill_loss


def gen_train_example_external(train_examples, tokenizer, query_templates, args):
    '''
    generate dataloader
    :param train_examples:
    :param tokenizer:
    :param query_templates:
    :return:
    '''
    train_features = convert_examples_to_features_t_s(
        examples=train_examples,
        tokenizer=tokenizer,
        query_templates=query_templates,
        t_nth_query=args.teacher_nth_query,
        s_nth_query=args.teacher_nth_query,
        is_training=True,
        args=args, student_external=True)

    # teacher
    all_input_ids = torch.tensor([f[0].input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[0].input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[0].segment_ids for f in train_features], dtype=torch.long)
    all_if_trigger_ids = torch.tensor([f[0].if_trigger_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f[0].start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f[0].end_position for f in train_features], dtype=torch.long)
    all_seg_idxs = torch.tensor([f[0].seg_idxs for f in train_features], dtype=torch.long)
    all_golden_args = torch.tensor([f[0].golden_argument for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                               all_start_positions, all_end_positions, all_seg_idxs, all_golden_args)
    teacher_train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
    teacher_train_batches = [batch for batch in teacher_train_dataloader]

    # student
    all_input_ids = torch.tensor([f[1].input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1].input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[1].segment_ids for f in train_features], dtype=torch.long)
    all_if_trigger_ids = torch.tensor([f[1].if_trigger_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f[1].start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f[1].end_position for f in train_features], dtype=torch.long)
    all_seg_idxs = torch.tensor([f[1].seg_idxs for f in train_features], dtype=torch.long)
    all_golden_args = torch.tensor([f[1].golden_argument for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                               all_start_positions, all_end_positions, all_seg_idxs, all_golden_args)
    student_train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
    student_train_batches = [batch for batch in student_train_dataloader]

    return teacher_train_batches, student_train_batches


def gen_train_example(train_examples, tokenizer, query_templates, args):
    '''
    generate dataloader
    :param train_examples:
    :param tokenizer:
    :param query_templates:
    :return:
    '''
    train_features = convert_examples_to_features_t_s(
        examples=train_examples,
        tokenizer=tokenizer,
        query_templates=query_templates,
        t_nth_query=args.teacher_nth_query,
        s_nth_query=args.student_nth_query,
        is_training=True,
        args=args, student_external=False)

    all_input_ids = torch.tensor([f[0].input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[0].input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[0].segment_ids for f in train_features], dtype=torch.long)
    all_if_trigger_ids = torch.tensor([f[0].if_trigger_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f[0].start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f[0].end_position for f in train_features], dtype=torch.long)
    all_seg_idxs = torch.tensor([f[0].seg_idxs for f in train_features], dtype=torch.long)
    all_golden_args = torch.tensor([f[0].golden_argument for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                               all_start_positions, all_end_positions, all_seg_idxs, all_golden_args)
    teacher_train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
    teacher_train_batches = [batch for batch in teacher_train_dataloader]

    # student
    all_input_ids = torch.tensor([f[1].input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1].input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[1].segment_ids for f in train_features], dtype=torch.long)
    all_if_trigger_ids = torch.tensor([f[1].if_trigger_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f[1].start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f[1].end_position for f in train_features], dtype=torch.long)
    all_seg_idxs = torch.tensor([f[1].seg_idxs for f in train_features], dtype=torch.long)
    all_golden_args = torch.tensor([f[1].golden_argument for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                               all_start_positions, all_end_positions, all_seg_idxs, all_golden_args)
    student_train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
    student_train_batches = [batch for batch in student_train_dataloader]

    return teacher_train_batches, student_train_batches


def mask_query(batch_start_logits, batch_end_logits, t_seg_idxs):
    batch_start_logits_new = torch.zeros_like(batch_start_logits)
    batch_end_logits_new = torch.zeros_like(batch_end_logits)
    seg_mask_new = torch.zeros_like(batch_end_logits)
    for i in range(batch_start_logits.shape[0]):
        batch_start_logits_new[i][0:t_seg_idxs[i][1]-t_seg_idxs[i][0]] = batch_start_logits[i][t_seg_idxs[i][0]:t_seg_idxs[i][1]]
        batch_end_logits_new[i][0:t_seg_idxs[i][1]-t_seg_idxs[i][0]] = batch_end_logits[i][t_seg_idxs[i][0]:t_seg_idxs[i][1]]
        seg_mask_new[i][0:t_seg_idxs[i][1]-t_seg_idxs[i][0]] = torch.Tensor([1]*(t_seg_idxs[i][1].numpy()-t_seg_idxs[i][0].numpy()))

    return batch_start_logits_new, batch_end_logits_new, seg_mask_new

def main(args):

    # curriculum learning proportion
    arg_prop_list = [1.0, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0]
    # dir_temp = str(arg_prop_list)+str(alpha_list)
    dir_temp = str(args.beta)
    output_dir = os.path.join(args.output_dir, dir_temp)
    # args.output_dir = os.path.join(args.output_dir, str(arg_prop_list))
    n_gpu, device = init_config(args, output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    # read query templates
    query_templates = read_query_templates(normal_file=args.normal_file, des_file=args.des_file)

    if args.do_train or (not args.eval_test):
        eval_examples = read_ace_examples(input_file=args.dev_file, is_training=False)
        gold_examples = read_ace_examples(input_file=args.gold_file, is_training=False)

        # student dev dataloader
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            query_templates=query_templates,
            nth_query=args.student_nth_query,
            is_training=False,
            args=args)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
        student_eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        best_result = None
        lr = args.learning_rate
        if not args.add_if_trigger_embedding:
            teacher_model = BertForQuestionAnswering.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
        else:
            teacher_model1 = BertForQuestionAnswering_withIfTriggerEmbedding_golden_args.from_pretrained(args.model1)
            teacher_model1.eval()
            student_model = BertForQuestionAnswering_withIfTriggerEmbedding_golden_args_distill.from_pretrained(args.student_model,
                                                                                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

        if args.fp16:
            student_model.half()
            student_model.to(device)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
        param_optimizer = list(student_model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        global_step = 0
        start_time = time.time()
        epoch_i = 0  # epoch number

        for arg_proportion in arg_prop_list:
            epoch_i += 1
            args.arg_proportion = arg_proportion
            train_examples = read_ace_examples(input_file=args.train_file, is_training=True)

            # 随机选哪些样本要有额外知识
            train_examples_len = len(train_examples)
            external_train_examples = random.sample(train_examples, int(train_examples_len*arg_proportion))
            normal_train_examples = list(set(train_examples).difference(set(external_train_examples)))

            ex_teacher_train_batches, ex_student_train_batches = gen_train_example_external(external_train_examples, tokenizer, query_templates, args)
            nor_teacher_train_batches, nor_student_train_batches = gen_train_example(normal_train_examples, tokenizer, query_templates, args)

            student_train_batches = ex_student_train_batches + nor_student_train_batches
            teacher_train_batches = ex_teacher_train_batches + nor_teacher_train_batches

            num_train_optimization_steps = \
                len(teacher_train_batches) // args.gradient_accumulation_steps * args.num_train_epochs

            logger.info("*****teacher Train *****")
            logger.info("  Num orig examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            eval_step = max(1, len(teacher_train_batches) // args.eval_per_epoch)

            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

            for epoch in range(int(args.num_train_epochs)):

                student_model.train()
                teacher_model1.to(device)
                student_model.to(device)

                logger.info("Start epoch #{} (lr = {})...".format(epoch_i, lr))
                all_train_batches = []
                for i in range(len(student_train_batches)):
                    all_train_batches.append([teacher_train_batches[i], student_train_batches[i]])
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(all_train_batches)
                for step, batch in enumerate(all_train_batches):
                    t_input_ids, t_input_mask, t_segment_ids, \
                    t_if_trigger_ids, t_start_positions, t_end_positions, t_seg_idxs, t_gold_args = batch[0]
                    s_input_ids, s_input_mask, s_segment_ids, \
                    s_if_trigger_ids, s_start_positions, s_end_positions, s_seg_idxs, s_gold_args = batch[1]

                    t_input_ids = t_input_ids.to(device)
                    t_input_mask = t_input_mask.to(device)
                    t_segment_ids = t_segment_ids.to(device)
                    t_if_trigger_ids = t_if_trigger_ids.to(device)
                    t_gold_args = t_gold_args.to(device)

                    s_input_ids = s_input_ids.to(device)
                    s_input_mask = s_input_mask.to(device)
                    s_segment_ids = s_segment_ids.to(device)
                    s_if_trigger_ids = s_if_trigger_ids.to(device)
                    s_start_positions = s_start_positions.to(device)
                    s_end_positions = s_end_positions.to(device)
                    s_gold_args = s_gold_args.to(device)

                    with torch.no_grad():
                        if not args.add_if_trigger_embedding:
                            t_batch_start_logits, t_batch_end_logits = teacher_model1(t_input_ids, t_segment_ids, t_input_mask)

                        else:
                            t_batch_start_logits1, t_batch_end_logits1, t_start_dist1, \
                            t_end_dist1, t_arg_mask_metrix, t_arg_features1 = \
                                teacher_model1(input_ids=t_input_ids, if_trigger_ids=t_if_trigger_ids,
                                                                                     token_type_ids=t_segment_ids,
                                                                                     attention_mask=t_input_mask, golden_args=t_gold_args,
                                                                                     distance_metric=args.distance_metric)

                    # mask teacher query part
                    t_batch_start_logits1_masked, t_batch_end_logits1_masked, t_seg_mask_new = mask_query(t_batch_start_logits1, t_batch_end_logits1, t_seg_idxs)

                    s_start_logits, s_start_positions_clap, s_end_logits, s_end_positions_clap,\
                        s_start_dist, s_end_dist, s_arg_mask_metrix, s_arg_features = \
                        student_model(
                                    input_ids=s_input_ids,
                                    if_trigger_ids=s_if_trigger_ids, token_type_ids = s_segment_ids,
                                    attention_mask = s_input_mask,
                                    start_positions = s_start_positions,end_positions = s_end_positions,
                                    golden_args=s_gold_args, distance_metric=args.distance_metric
                                      )

                    # mask student query part
                    s_start_logits_masked, s_end_logits_masked, s_seg_mask_new = mask_query(s_start_logits, s_end_logits,
                                                                            s_seg_idxs)

                    loss = distillation(s_start_logits, s_end_logits, s_start_logits_masked, s_end_logits_masked,
                                        t_batch_start_logits1_masked, t_batch_end_logits1_masked,
                                        s_start_positions_clap, s_end_positions_clap, args,
                                        t_seg_mask_new, s_seg_mask_new, t_start_dist1, t_end_dist1,
                                        s_start_dist, s_end_dist,
                                        t_arg_mask_metrix, s_arg_mask_metrix,
                                        t_arg_features1,
                                        s_arg_features,
                                        distance_metric=args.distance_metric
                                        )

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss = tr_loss + loss.item()
                    nb_tr_examples += t_input_ids.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0:
                        save_model = False
                        if args.do_eval:
                            result, preds = evaluate(args, student_model, device, student_eval_dataloader, eval_examples, gold_examples,
                                                     eval_features)
                            # import ipdb; ipdb.set_trace()
                            student_model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            # print('f1_c', result[args.eval_metric])
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                    epoch, step + 1, len(all_train_batches), time.time() - start_time,
                                           tr_loss / nb_tr_steps))
                                # logger.info("!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f" %
                                # (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"], result["f1_c"]))
                                logger.info(
                                    "!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, p_i: %.2f, r_i: %.2f, f1_i: %.2f" %
                                    (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"],
                                     result["f1_c"], result["prec_i"], result["recall_i"], result["f1_i"]))
                        else:
                            save_model = True
                        if save_model:
                            if best_result:
                                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                                output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
                                output_config_file = os.path.join(output_dir, "config.json")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                tokenizer.save_vocabulary(output_dir)
                                with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

    # if args.do_eval:
    #     if args.eval_test:
    #         eval_examples = read_ace_examples(input_file=args.dev_file, is_training=False)
    #         gold_examples = read_ace_examples(input_file=args.gold_file, is_training=False)
    #
    #         # student dev dataloader
    #         eval_features = convert_examples_to_features(
    #             examples=eval_examples,
    #             tokenizer=tokenizer,
    #             query_templates=query_templates,
    #             nth_query=args.student_nth_query,
    #             is_training=False,
    #             args=args)
    #         logger.info("***** Dev *****")
    #         logger.info("  Num orig examples = %d", len(eval_examples))
    #         logger.info("  Num split examples = %d", len(eval_features))
    #         logger.info("  Batch size = %d", args.eval_batch_size)
    #
    #         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #         all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
    #         all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
    #                                   all_example_index)
    #         student_eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    #
    #         eval_examples = read_ace_examples(input_file=args.test_file, is_training=False)
    #         gold_examples = read_ace_examples(input_file=args.gold_file, is_training=False)
    #         eval_features = convert_examples_to_features(
    #             examples=eval_examples,
    #             tokenizer=tokenizer,
    #             query_templates=query_templates,
    #             nth_query=args.student_nth_query,
    #             is_training=False,
    #             args=args)
    #         logger.info("***** Test *****")
    #         logger.info("  Num orig examples = %d", len(eval_examples))
    #         logger.info("  Num split examples = %d", len(eval_features))
    #         logger.info("  Batch size = %d", args.eval_batch_size)
    #         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #         all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
    #         all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
    #                                   all_example_index)
    #         eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    #
    #     if not args.add_if_trigger_embedding:
    #         student_model = BertForQuestionAnswering_withIfTriggerEmbedding_distill.from_pretrained(output_dir)
    #     else:
    #         student_model = BertForQuestionAnswering_withIfTriggerEmbedding_distill.from_pretrained(output_dir)
    #     if args.fp16:
    #         student_model.half()
    #     student_model.to(device)
    #
    #     result, preds = evaluate(args, student_model, device, eval_dataloader, eval_examples, gold_examples, eval_features)
    #
    #     with open(os.path.join(output_dir, "test_results_convert.txt"), "w") as writer:
    #         for key in result:
    #             writer.write("%s = %s\n" % (key, str(result[key])))
    #     with open(os.path.join(output_dir, "stu_arg_predictions.txt"), "w") as writer:
    #         for key in preds:
    #             writer.write(json.dumps(preds[key], default=int) + "\n")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", default='./RAMS_teacher_uncased_fenci', type=str)

    parser.add_argument("--student_model", default='./bert-base-uncased', type=str)
    parser.add_argument("--temperature", default=1, type=int)
    parser.add_argument("--output_dir", default='./AREA_output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file",
                        default='../proc/data/RAMS-event/processed-data/RAMS_out_fenci/train_data_fenci.json',
                        type=str)
    parser.add_argument("--dev_file",
                        default='../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json', type=str)
    parser.add_argument("--test_file",
                        default='../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json', type=str)
    parser.add_argument("--gold_file",
                        default='../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json', type=str)

    parser.add_argument("--eval_per_epoch", default=20, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true',default=True, help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="number of each curriculum training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1_c', type=str)
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=5, type=int,
                        help="The maximum length of an answer that can be generated. "
                             "This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument("--teacher_nth_query", default=1, type=int, help="use n-th template query")
    parser.add_argument("--student_nth_query", default=2, type=int, help="use n-th template query")
    parser.add_argument("--normal_file", default='../question_templates/RAMS_query_TCP_final.csv', type=str)
    parser.add_argument("--des_file", default='../question_templates/description_queries.csv', type=str)
    parser.add_argument("--larger_than_cls", action='store_true', help="when indexing s and e")
    parser.add_argument("--add_if_trigger_embedding", default=True,
                        help="add the if_trigger_embedding")
    parser.add_argument('--alpha', default=0.5, type=float, help="the alpha coefficient of CKD loss")
    parser.add_argument('--model1_alpha', default=1, type=float)
    parser.add_argument('--beta', default=0.003, type=float, help="the beta coefficient of RKD loss")
    parser.add_argument('--distance_metric', default='angle', type=str, choices=['distance', 'angle', 'MMD', 'bilinear', 'RBF'])

    args = parser.parse_args()

    if max_seq_length != args.max_seq_length: max_seq_length = args.max_seq_length

    main(args)
