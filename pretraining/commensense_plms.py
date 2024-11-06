#######################################################
# Author: Yudai Pan
#######################################################
import logging
import argparse
import torch
import os
import sys
sys.path.insert(0, '../')
import sys
import warnings
from collections import Counter
# from pretrained.QA_bert import train
# from pretrained.qa_classifier import train
# from pretrained.lm_finetuning_test import train
# from pretrained.finetuning_context import train
# from pretrained.prompt_tuning import train
# from pretrained.prompt_tuning_cor import train
# from pretrained.prompt_tuning_plms import train
from pretrained.prompt_tuning_roberta import train
# from pretrained.prompt_tuning_bi_atten import train
# from pretrained.prompt_tuning_bi import train
import src.reader_utils as reader_utils
from src.reader import AtomicTSVReader, ConceptNetTSVReader, FB15kReader
from pretrained.sentence_generate import question_generate_dict, question_generate, question_generate_context, \
    question_generate_paths, question_generate_paths_bi
from src.bert_feature_extractor import convert_nodes_to_examples, convert_examples_to_features
from src.utils import get_adj_and_degrees
# from cluster_test.cluster_test import save_bert_model_embedding

warnings.filterwarnings('ignore')
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, \
    RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn as nn


def load_data(dataset, reader_cls, data_dir, sim_relations):
    train_network = reader_cls(dataset)
    dev_network = reader_cls(dataset)
    test_network = reader_cls(dataset)

    train_network.read_network(data_dir=data_dir, split="train")
    train_network.print_summary()
    node_list_old = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list_old]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
    test_network.read_network(data_dir=data_dir, split="test", train_network=train_network)

    word_vocab = train_network.graph.node2id

    # Add sim nodes
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)

    node_list = train_network.graph.iter_nodes()
    return train_data, valid_data, test_data, valid_labels, test_labels, train_network, node_list


def data_pretreated(args, dataset_cls, data_dir):
    # -----------------------------------------------数据加载处理-----------------------------------------------------
    # Store entity-wise dicts for filtered metrics
    train_data, valid_data, test_data, valid_labels, test_labels, train_network, node_list = load_data(args.dataset,
                                                                                            dataset_cls,
                                                                                            data_dir,
                                                                                            args.sim_relations)
    # train_network是graph类别的对象

    num_nodes = len(train_network.graph.nodes)
    num_rels = len(train_network.graph.relations)
    all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()  # 所有tuples

    # for filtered ranking
    # 一个头节点对应的所有尾节点；一个尾节点对应的所有头节点
    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, num_rels, args.sim_relations)

    # for training
    # 一个头节点对应的训练集尾节点；一个尾节点对应的训练集头节点
    train_e1_to_multi_e2, train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels,
                                                                                  args.sim_relations)
    # the below dicts `include` sim relations
    # 加相似（-1）的关系
    sim_train_e1_to_multi_e2, sim_train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels)

    # -------------------------------------------------------------------------------------
    return train_network, train_data, valid_data, num_nodes, all_e1_to_multi_e2, all_e2_to_multi_e1, train_e1_to_multi_e2, \
           train_e2_to_multi_e1, sim_train_e1_to_multi_e2, sim_train_e2_to_multi_e1, node_list


def save_bert_model(args, node_list):

    # -----------------------------加载.bin 模型-------------------------------------
    max_seq_length = args.max_seq_length_node
    eval_batch_size = args.eval_batch_size
    # bert_model = "bert-large-uncased"
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)  # 加载Tokenizer
    output_dir = "../"+ args.bert_model + "/" + args.dataset +"/pytorch_model.bin"
    print("Loading model from %s" % args.output_dir)
    # bert_model = torch.load(output_model_file, map_location='cpu')  # 加载预训练模型
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    # # self.bert_model = BertLayer.load_state_dict(torch.load(output_model_file))
    # # ------------------------------------------------------------------------------
    # bert_model.to(device)
    # model_state_dict = torch.load(output_model_file, map_location=lambda storage, loc: storage.cuda(2))
    #
    # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=2)

    if args.pretrained_model == "Bert":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)  # 加载Tokenizer
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    elif args.pretrained_model == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)  # 加载Tokenizer
        model = AlbertForMaskedLM.from_pretrained(args.bert_model)
    elif args.pretrained_model == "Roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)  # 加载Tokenizer
        model = RobertaForMaskedLM.from_pretrained(args.bert_model)
    # model = BertForMaskedLM.from_pretrained(args.bert_model)
    model_dict = model.state_dict()
    pretrained_model = torch.load(output_dir, map_location=lambda storage, loc: storage.cuda(2)).state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

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

    idx = 0
    for input_ids, input_mask, segment_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            sequence_output= model.bert(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)[0]
        sequence_outputs.append(sequence_output[:, 0])

        # if len(sequence_outputs) == 800:
        #     save_to_disk(args.dataset, torch.cat(sequence_outputs, dim=0), idx)
        #     sequence_outputs = []
        #     idx += 1

    save_to_disk(args.dataset, torch.cat(sequence_outputs, dim=0), idx)


def save_to_disk(dataset, tensor, idx):
    filename = os.path.join(args.output_dir, args.dataset + str(idx) + "_bert_embeddings_base.pt")
    torch.save(tensor, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    # General
    parser.add_argument("-d", "--dataset", type=str, default='conceptnet',
                        help="dataset to use")
    parser.add_argument("--sample_num", type=int, default=3,
                        help="how many samples for pre-trained")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--sim_relations", action='store_true', default=False,
                        help="add similarity edges when constructing graph")
    parser.add_argument("--output_dir", type=str, default="plm_roberta_large_margin_2",
                        help="output directory to store metrics and model file")
    parser.add_argument("--gpu", type=int, default=5,
                        help="gpu")
    parser.add_argument("--random_sample", type=bool, default=False,
                        help="random split the train and valid data")
    # parser.add_argument("--epochs", type=int, default=4,
    #                     help="number of minimum training epochs")
    ## Required parameters
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default='comm',
                        type=str,
                        help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,      #32
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,     #32
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true', default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--demo_flag",
                        default=False,
                        type=bool,
                        help="Do we use demo.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--train_corpus",
                        default="",
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--on_memory",
                        action='store_true',
                        default=True,
                        help="Whether to load train samples into memory or use disk")
    # parser.add_argument("--margin_sub",
    #                     default=3,
    #                     type=float,
    #                     help="Margin in marginloss.")
    parser.add_argument("--margin",
                        default=0.2,
                        type=float,
                        help="Margin in marginloss.")
    parser.add_argument("--lambda_path",
                        default=0.2,
                        type=float,
                        help="lambda for path loss.")
    parser.add_argument("--lambda_mask",
                        default=0.8,
                        type=float,
                        help="lambda for mask loss.")
    parser.add_argument("--max_seq_length_node",
                        default=32,
                        type=int,
                        help="max length of node")
    parser.add_argument("--eval_batch_size_node",
                        default=32,
                        type=int,
                        help="evaluation batch size of node embedding saving process")
    parser.add_argument("--pretrained_model",
                        default="roberta",
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default="../pretrained/plms/roberta-large", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")


    args = parser.parse_args()
    print(args)

    # load graph data
    if args.dataset == "FB15K-237":
        dataset_cls = FB15kReader
        data_dir = "../data/FB15k-237/"
    elif args.dataset == "atomic":
        dataset_cls = AtomicTSVReader
        data_dir = "../data/atomic/"
    elif args.dataset == "conceptnet":
        dataset_cls = ConceptNetTSVReader
        data_dir = "../data/ConceptNet/"
    elif args.dataset == "conceptdemo":
        dataset_cls = ConceptNetTSVReader
        data_dir = "../data/conceptdemo/"
    else:
        raise ValueError("Invalid option for dataset.")

    graph_network, train_data, valid_data, num_nodes, all_sub_to_obj, _, train_sub_to_obj, train_obj_to_sub, \
        sim_train_sub_to_obj, _, node_list = data_pretreated(args, dataset_cls, data_dir)
    num_rels = len(graph_network.graph.relations)

    adj_list, degrees, sparse_adj_matrix, rel = get_adj_and_degrees(num_nodes, num_rels, train_data)

    # if args.demo_flag:
    # question_answer_train = question_generate_dict(graph_network, train_sub_to_obj, args)           # question_answer
    # question_answer_train = question_generate_context(graph_network, train_sub_to_obj, train_obj_to_sub, args)          # question_answer
    # question_answer_train = question_generate_paths_prompt(graph_network, train_sub_to_obj, train_obj_to_sub, args)
    # question_answer_train = question_generate_paths(graph_network, train_sub_to_obj, train_obj_to_sub, args)
    # question_answer_train = question_generate_paths_atomic(graph_network, train_sub_to_obj, train_obj_to_sub, args)
    # question_answer_val = question_generate(graph_network, valid_data, all_sub_to_obj, args)
    # question_answer = question_generate(graph_network, train_data, train_sub_to_obj, args)

    file_handler = logging.FileHandler(os.path.join(args.output_dir, f"log_test.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info('============ Initialized logger ============')
    logger.info(args)

    try:
        # train(args, question_answer_train, question_answer_val)
        # train_t = (("What would I do before get up?", "open eyes", 1),
        #            ("What would I do before get up?", "close eyes", 0))
        # valid_t = (("What would I do before sleep?", "close eyes", 1),
        #            ("What would I do before sleep?", "open eyes", 0))
        # train_t = question_answer_train
        # valid_t = question_answer_val
        # with open('corpus_path.txt', 'w') as f:
        #     for question, answer, label in train_t:
        #         # print(question + "," + answer + " ")
        #         if label == 1:
        #             f.writelines(question + "\t" + answer)
        #             f.writelines("\n")
        # train(args, train_t, valid_t, node_list)
        # save_bert_model(args, node_list)
        print("finish")
        train(args, node_list)
    except KeyboardInterrupt:
        print('Interrupted')
