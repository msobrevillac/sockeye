# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import gzip
import logging
import math
import pickle
import random
from collections import OrderedDict
from typing import Any, Dict, Iterator, Iterable, List, NamedTuple, Optional, Tuple

import math
import mxnet as mx
import numpy as np

from sockeye.utils import check_condition
from . import config
from . import constants as C

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step=10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.
    :return: List of bucket sizes.
    """
    buckets = [bucket_len for bucket_len in range(step, max_seq_len + step, step)]
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param length_ratio: Length ratio of data (target/source).
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if length_ratio >= 1.0:
        # target side is longer -> scale source
        source_step_size = max(1, int(bucket_width / length_ratio))
    else:
        # source side is longer, -> scale target
        target_step_size = max(1, int(bucket_width * length_ratio))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # Extra buckets
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    # deduplicate for return
    return list(OrderedDict.fromkeys(parallel_buckets))


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket.
    """
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


def read_parallel_corpus(data_source: str,
                         data_target: str,
                         data_source_graphs: str,
                         vocab_source: Dict[str, int],
                         vocab_target: Dict[str, int],
                         vocab_edges: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Loads source and target data, making sure they have the same length.
    # TODO: fix return type

    :param data_source: Path to source training data.
    :param data_target: Path to target training data.
    :param data_source_graphs: Path to graphs for source training data.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param vocab_edges: Graph edges vocabulary.
    :return: Tuple of (source sentences, target sentences).
    """
    source_sentences = read_sentences(data_source, vocab_source, add_bos=False)
    source_graphs = read_graphs(data_source_graphs, vocab_edges)
    target_sentences = read_sentences(data_target, vocab_target, add_bos=True)
    check_condition(len(source_sentences) == len(target_sentences),
                    "Number of source sentences does not match number of target sentences")
    check_condition(len(source_sentences) == len(source_graphs),
                    "Number of source sentences does not match number of source graphs")
    return source_sentences, target_sentences, source_graphs


def length_statistics(source_sentences: List[List[Any]], target_sentences: List[List[int]]) -> Tuple[float, float]:
    """
    Returns mean and standard deviation of target-to-source length ratios of parallel corpus.

    :param source_sentences: Source sentences.
    :param target_sentences: Target sentences.
    :return: Mean and standard deviation of length ratios.
    """
    #length_ratios = []
    #for ts, ss in zip(target_sentences, source_sentences):
    #    length_ratios += [len(t)/float(len(s)) for t, s in zip(ts, ss)]

    length_ratios = np.array([len(t)/float(len(s)) for ts, ss in zip(target_sentences, source_sentences) 
                                                   for t, s in zip(ts, ss)])
    mean = np.asscalar(np.mean(length_ratios))
    std = np.asscalar(np.std(length_ratios))
    return mean, std


def get_training_data_iters(source: List[str], target: List[str], source_graphs: List[str],
                            validation_source: str, validation_target: str,
                            val_source_graphs: str,
                            vocab_source: Dict[str, int],
                            vocab_target: Dict[str, int],
                            vocab_edge: Dict[str, int],
                            vocab_source_path: Optional[str],
                            vocab_target_path: Optional[str],
                            vocab_edge_path: Optional[str],
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            fill_up: str,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            temperature: float) -> Tuple['ParallelBucketSentenceIter',
                                                        'ParallelBucketSentenceIter',
                                                        'DataConfig']:
    """
    Returns data iterators for training and validation data.

    :param source: Path to source training data.
    :param target: Path to target training data.
    :param source_graphs: Path to source training graphs.
    :param validation_source: Path to source validation data.
    :param validation_target: Path to target validation data.
    :param val_source_graphs: Path to source validation graphs.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param vocab_edge: Graph edges vocabulary.
    :param vocab_source_path: Path to source vocabulary.
    :param vocab_target_path: Path to target vocabulary.
    :param vocab_edge_path: Path to metadata vocabulary.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("Creating train data iterator")
    train_source_sentences = []
    train_target_sentences = []
    train_source_graphs = []
    for src, tgt, src_graphs in zip(source, target, source_graphs):

        (train_src_sentences,
         train_tgt_sentences,
         train_src_graphs) = read_parallel_corpus(src,
                                                     tgt,
                                                     src_graphs,
                                                     vocab_source,
                                                     vocab_target,
                                                     vocab_edge)
        train_source_sentences.append(train_src_sentences)
        train_target_sentences.append(train_tgt_sentences)
        train_source_graphs.append(train_src_graphs)

    
    max_observed_source_len = max((len(s) for train_src_stn in train_source_sentences for s in train_src_stn 
                                          if len(s) <= max_seq_len_source), default=0)
    max_observed_target_len = max((len(t) for train_tgt_stn in train_target_sentences for t in train_tgt_stn 
                                          if len(t) <= max_seq_len_target), default=0)

    lr_mean, lr_std = length_statistics(train_source_sentences, train_target_sentences)
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)", lr_mean, lr_std)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source,
                                      max_seq_len_target,
                                      bucket_width,
                                      lr_mean) if bucketing else [
        (max_seq_len_source, max_seq_len_target)]

    train_iter = ParallelBucketSentenceIter(train_source_sentences,
                                            train_target_sentences,
                                            train_source_graphs,
                                            buckets,
                                            batch_size,
                                            batch_by_words,
                                            batch_num_devices,
                                            vocab_target[C.EOS_SYMBOL],
                                            C.PAD_ID,
                                            vocab_target[C.UNK_SYMBOL],
                                            vocab_edge['d'],
                                            bucket_batch_sizes=None,
                                            fill_up=fill_up,
                                            temperature= temperature)

    logger.info("Creating validation data iterator")
    val_iter = None

    (val_source_sentences,
     val_target_sentences,
     val_src_graphs) = read_parallel_corpus(validation_source,
                                               validation_target,
                                               val_source_graphs,
                                               vocab_source,
                                               vocab_target,
                                               vocab_edge)
    val_iter = ParallelBucketSentenceIter([val_source_sentences],
                                          [val_target_sentences],
                                          [val_src_graphs],
                                          buckets,
                                          batch_size,
                                          batch_by_words,
                                          batch_num_devices,
                                          vocab_target[C.EOS_SYMBOL],
                                          C.PAD_ID,
                                          vocab_target[C.UNK_SYMBOL],
                                          vocab_edge['d'],
                                          bucket_batch_sizes=train_iter.bucket_batch_sizes,
                                          fill_up=fill_up,
                                          temperature=temperature)

    config_data = DataConfig(source, target, source_graphs,
                             validation_source, validation_target,
                             val_source_graphs,
                             vocab_source_path, vocab_target_path,
                             vocab_edge_path,
                             lr_mean, lr_std, max_observed_source_len, max_observed_target_len)
    return train_iter, val_iter, config_data

#train_iter = get_data_iter(source, target, source_graph, vocab_source, vocab_target, batch_size, fill_up,
#                               max_seq_len, bucketing, bucket_width=bucket_width)
#    logger.info("Creating validation data iterator")
#    eval_iter = get_data_iter(validation_source, validation_target, val_source_graph, vocab_source, vocab_target, batch_size, fill_up,
#                              max_seq_len, bucketing, bucket_width=bucket_width)
#    return train_iter, eval_iter



class DataConfig(config.Config):
    """
    Stores data paths from training.
    """
    def __init__(self,
                 source: str,
                 target: str,
                 source_graphs: str,
                 validation_source: str,
                 validation_target: str,
                 val_source_graphs: str,
                 vocab_source: Optional[str],
                 vocab_target: Optional[str],
                 vocab_edge: Optional[str],
                 length_ratio_mean: float = C.TARGET_MAX_LENGTH_FACTOR,
                 length_ratio_std: float = 0.0,
                 max_observed_source_seq_len: Optional[int] = None,
                 max_observed_target_seq_len: Optional[int] = None) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.source_graphs = source_graphs
        self.validation_source = validation_source
        self.validation_target = validation_target
        self.val_source_graphs = val_source_graphs
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.vocab_edge = vocab_edge
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.max_observed_source_seq_len = max_observed_source_seq_len
        self.max_observed_target_seq_len = max_observed_target_seq_len


def smart_open(filename: str, mode="rt", ftype="auto", errors='replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open
    :param errors: Encoding error handling during reading. Defaults to 'replace'
    :return: File descriptor
    """
    if ftype == 'gzip' or ftype == 'gz' or (ftype == 'auto' and filename.endswith(".gz")):
        return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)


def read_content(path: str, limit=None) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of words.
    """
    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            yield list(get_tokens(line))


def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token


def tokens2ids(tokens: Iterable[str], vocab: Dict[str, int]) -> List[int]:
    """
    Returns sequence of ids given a sequence of tokens and vocab.

    :param tokens: List of tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :return: List of word ids.
    """
    return [vocab.get(w, vocab[C.UNK_SYMBOL]) for w in tokens]


def read_sentences(path: str, vocab: Dict[str, int], add_bos=False, limit=None) -> List[List[int]]:
    """
    Reads sentences from path and creates word id sentences.

    :param path: Path to read data from.
    :param vocab: Vocabulary mapping.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    :return: List of integer sequences.
    """
    assert C.UNK_SYMBOL in vocab
    assert C.UNK_SYMBOL in vocab
    assert vocab[C.PAD_SYMBOL] == C.PAD_ID
    assert C.BOS_SYMBOL in vocab
    assert C.EOS_SYMBOL in vocab
    sentences = []
    for sentence_tokens in read_content(path, limit):
        sentence = tokens2ids(sentence_tokens, vocab)
        check_condition(bool(sentence), "Empty sentence in file %s" % path)
        if add_bos:
            sentence.insert(0, vocab[C.BOS_SYMBOL])
        sentences.append(sentence)
    logger.info("%d sentences loaded from '%s'", len(sentences), path)
    return sentences


def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)


def get_parallel_bucket(buckets: List[Tuple[int, int]],
                        length_source: int,
                        length_target: int) -> Optional[Tuple[int, Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None  # type: Tuple[int, Tuple[int, int]]
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            bucket = j, (source_bkt, target_bkt)
            break
    return bucket


BucketBatchSize = NamedTuple("BucketBatchSize", [
    ("batch_size", int),
    ("average_words_per_batch", float)
])
"""
:param batch_size: Number of sentences in each batch.
:param average_words_per_batch: Approximate number of non-padding tokens in each batch.
"""


def read_graphs(path: str, vocab: Dict[str, int], limit=None): #TODO: add return type
    """
    Reads graphs from path, creating a list of tuples for each sentence.
    We assume the format for graphs uses whitespace as separator.
    This allows us to reuse the reading methods for the sentences.

    :param path: Path to read data from.
    :return: List of sequences of integer tuples with the edge label.
    """
    graphs = []
    for graph_tokens in read_content(path, limit):
        graph = process_edges(graph_tokens, vocab)
        assert len(graph) > 0, "Empty graph in file %s" % path
        graphs.append(graph)
    logger.info("%d graphs loaded from '%s'", len(graphs), path)
    return graphs


def process_edges(graph_tokens: Iterable[str], vocab: Dict[str, int]): #TODO: add typing
    """
    Returns sequence of int tuples given a sequence of graph edges.
    
    TODO: this can be more efficient...

    :param graph_tokens: List of tokens containing graph edges.
    :return: List of (int, int) tuples
    """
    adj_list = [(int(tok[1:-1].split(',')[0]),
                 int(tok[1:-1].split(',')[1]),
                 vocab[tok[1:-1].split(',')[2]]) for tok in graph_tokens]
    return adj_list
    

# TODO: consider more memory-efficient data reading (load from disk on demand)
# TODO: consider using HDF5 format for language data
class ParallelBucketSentenceIter(mx.io.DataIter):
    """
    A Bucket sentence iterator for parallel data. Randomly shuffles the data after every call to reset().
    Data is stored in NDArrays for each epoch for fast indexing during iteration.

    :param source_sentences: List of source sentences (integer-coded).
    :param target_sentences: List of target sentences (integer-coded).
    :param source_graphs: List of source graphs (tuples of index pairs).
    :param buckets: List of buckets.
    :param batch_size: Batch_size of generated data batches.
           Incomplete batches are discarded if fill_up == None, or filled up according to the fill_up strategy.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param md_vocab_size: Size of metadata vocabulary, needed for the adjacency tensors.
    :param fill_up: If not None, fill up bucket data to a multiple of batch_size to avoid discarding incomplete batches.
           for each bucket. If set to 'replicate', sample examples from the bucket and use them to fill up.
    :param eos_id: Word id for end-of-sentence.
    :param pad_id: Word id for padding symbols.
    :param unk_id: Word id for unknown symbols.
    :param forward_id: Word id for forward edge symbol (used to get graph positions).
    :param bucket_batch_sizes: Pre-computed bucket batch sizes (used to keep iterators consistent for train/validation).
    :param dtype: Data type of generated NDArrays.
    """

    def __init__(self,
                 source_sentences: List[List[int]],
                 target_sentences: List[List[int]],
                 source_graphs: List[Tuple[int, int, str]],
                 buckets: List[Tuple[int, int]],
                 batch_size: int,
                 batch_by_words: bool,
                 batch_num_devices: int,
                 #edge_vocab_size: int,
                 eos_id: int,
                 pad_id: int,
                 unk_id: int,
                 forward_id: int,
                 bucket_batch_sizes: Optional[List[BucketBatchSize]] = None,
                 fill_up: Optional[str] = None,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 src_graphs_name=C.SOURCE_GRAPHS_NAME,
                 src_positions_name=C.SOURCE_POSITIONS_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 temperature=1.0,
                 dtype='float32') -> None:
        super(ParallelBucketSentenceIter, self).__init__()

        self.buckets = list(buckets)
        self.buckets.sort()
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.batch_size = batch_size
        self.batch_by_words = batch_by_words
        self.batch_num_devices = batch_num_devices
        #self.edge_vocab_size = edge_vocab_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.forward_id = forward_id
        self.dtype = dtype
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.src_graphs_name = src_graphs_name
        self.src_positions_name = src_positions_name
        self.label_name = label_name
        self.fill_up = fill_up
        self.number_languages = len(source_sentences) #number of datasets in the training
        self.temperature = temperature

        self.data_source = [] # type: ignore
        self.data_target = []  # type: ignore
        self.data_label = []  # type: ignore
        self.data_label_average_len = []
        self.data_src_graphs = []
        self.data_src_positions = []

        self.batch_samples = []
        self.majority_index = 0

        logger.info("Buckets: %s", self.buckets)

        # TODO: consider avoiding explicitly creating label arrays to save host memory
        for i in range(self.number_languages): #To handle several datasets (multilingual NLG)
            self.data_source.append([[] for _ in self.buckets])  # type: ignore
            self.data_target.append([[] for _ in self.buckets])  # type: ignore
            self.data_label.append([[] for _ in self.buckets])  # type: ignore
            self.data_label_average_len.append([0 for _ in self.buckets])
            self.data_src_graphs.append([[] for _ in self.buckets])
            self.data_src_positions.append([[] for _ in self.buckets])

        logger.info("Data source: %s", self.data_source)

        # Per-bucket batch sizes (num seq, num word)
        # If not None, populated as part of assigning to buckets
        self.bucket_batch_sizes = bucket_batch_sizes
        logger.info("Bucket batch sizes: %s", self.bucket_batch_sizes)

        # assign sentence pairs to buckets
        self._assign_to_buckets(source_sentences, target_sentences, source_graphs)

        # convert to single numpy array for each bucket
        self._convert_to_array()

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        self.provide_data = [
            mx.io.DataDesc(name=source_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[0]),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=target_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=src_graphs_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size,
                                  self.default_bucket_key[0],
                                  self.default_bucket_key[0]),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=src_positions_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size,
                                  self.default_bucket_key[0]),
                           layout=C.BATCH_MAJOR)]

        self.provide_label = [
            mx.io.DataDesc(name=label_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.target_data_name, self.src_graphs_name, self.src_positions_name]
        self.label_names = [self.label_name]
        logger.info(self.data_names)
        logger.info(self.label_names)

        # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
        #Now we are working with one or more idx and curre_idx (multilingual training)
        self.idx = []
        self.curr_idx = []
        for index in range(self.number_languages):
            idxi = []  # type: List[Tuple[int, int]]
            for i, buck in enumerate(self.data_source[index]):
                batch_size_seq = self.bucket_batch_sizes[i].batch_size

                if self.number_languages > 1:
                    batch_size_seq = self.batch_samples[index]

                logger.info("Buck: %d and batch_size_seq: %d/%d", len(buck), batch_size_seq, self.bucket_batch_sizes[i].batch_size)
                rest = len(buck) % batch_size_seq
                if rest > 0:
                    logger.info("Discarding %d samples from bucket %s due to incomplete batch", rest, self.buckets[i])
                idxs = [(i, j) for j in range(0, len(buck) - batch_size_seq + 1, batch_size_seq)]
                idxi.extend(idxs)
            self.idx.append(idxi)
            self.curr_idx.append(0)

        self.indices = [[] for _ in range(self.number_languages)]
        self.nd_source = [[] for _ in range(self.number_languages)]
        self.nd_target = [[] for _ in range(self.number_languages)]
        self.nd_label = [[] for _ in range(self.number_languages)]
        #####
        # GCN
        self.nd_src_graphs = [[] for _ in range(self.number_languages)]
        self.nd_src_positions = [[] for _ in range(self.number_languages)]

        self.reset()

#    @staticmethod
#    def _get_bucket(buckets, length_source, length_target):
#        """
#        Determines bucket given source and target length.
#        """
#        bucket = None, None
#        for j, (source_bkt, target_bkt) in enumerate(buckets):
#            if source_bkt >= length_source and target_bkt >= length_target:
#                bucket = j, (source_bkt, target_bkt)
#                break
#        return bucket

    def _assign_to_buckets(self, source_sentences, target_sentences, source_graphs):
        ndiscard = 0
        tokens_source = 0
        tokens_target = 0
        num_of_unks_source = 0
        num_of_unks_target = 0

        # Bucket sentences as padded np arrays
        for i, (src_sentences, tgt_sentences, src_graphs) in enumerate(zip(source_sentences, target_sentences, source_graphs)):
            for source, target, src_graph in zip(src_sentences, tgt_sentences, src_graphs):
                tokens_source += len(source)
                tokens_target += len(target)
                num_of_unks_source += source.count(self.unk_id)
                num_of_unks_target += target.count(self.unk_id)

                buck_idx, buck = get_parallel_bucket(self.buckets, len(source), len(target))
                #logger.info("Buck idx %s and buck %s for source \" %s \"", buck_idx, buck, source)
                if buck is None:
                    ndiscard += 1
                    continue
                buff_source = np.full((buck[0],), self.pad_id, dtype=self.dtype)
                buff_target = np.full((buck[1],), self.pad_id, dtype=self.dtype)
                buff_label = np.full((buck[1],), self.pad_id, dtype=self.dtype)
                buff_source[:len(source)] = source
                buff_target[:len(target)] = target
                buff_label[:len(target)] = target[1:] + [self.eos_id]
                self.data_source[i][buck_idx].append(buff_source)          
                self.data_target[i][buck_idx].append(buff_target)
                self.data_label[i][buck_idx].append(buff_label)
                self.data_label_average_len[i][buck_idx] += len(target)
                #####
                # GCN
                self.data_src_graphs[i][buck_idx].append(src_graph)
                # just fill empty lists here, these will be updated when
                # converting the data.
                self.data_src_positions[i][buck_idx].append([])
                #####

            #logger.info("Data source %d: %d", i, len(self.data_source[i]))
            #for j, buk in enumerate(self.data_source[i]):
            #    logger.info("Bucket %d: %d", j, len(buk))

            # Average number of non-padding elements in target sequence per bucket
            for buck_idx, buck in enumerate(self.buckets):
                # Case of empty bucket -> use default padded length
                if self.data_label_average_len[i][buck_idx] == 0:
                    self.data_label_average_len[i][buck_idx] = buck[1]
                else:
                    self.data_label_average_len[i][buck_idx] /= len(self.data_label[i][buck_idx])

        # calculating subsamples for each training dataset
        if self.number_languages > 1:
            logger.info("Calculating batch sub-samples with temperature %d for each dataset", self.temperature)
            self.batch_samples, self.majority_index = self._calculate_batch_samples(self.temperature)

        # We now have sufficient information to populate bucket batch sizes
        self._populate_bucket_batch_sizes()

        logger.info("Source words: %d", tokens_source)
        logger.info("Target words: %d", tokens_target)
        logger.info("Vocab coverage source: %.0f%%", (1 - num_of_unks_source / tokens_source) * 100)
        logger.info("Vocab coverage target: %.0f%%", (1 - num_of_unks_target / tokens_target) * 100)
        logger.info("Total: %d samples in %d buckets", sum(len(b) for ds in self.data_source for b in ds), len(self.buckets))

        for i in range(self.number_languages):
            nsamples = 0
            for bkt, buck, batch_size_seq, average_seq_len in zip(self.buckets,
                                                              self.data_source[i],
                                                              (bbs.batch_size for bbs in self.bucket_batch_sizes),
                                                              self.data_label_average_len[i]):

                if self.number_languages > 1:
                    batch_size_seq = self.batch_samples[i]
                
                logger.info("Bucket of %s : %d samples in %d batches of %d, approx %0.1f words/batch",
                        bkt,
                        len(buck),
                        math.ceil(len(buck) / batch_size_seq),
                        batch_size_seq,
                        batch_size_seq * average_seq_len)
                nsamples += len(buck)
            check_condition(nsamples > 0, "0 data points available in the data iterator. "
                                      "%d data points have been discarded because they "
                                      "didn't fit into any bucket. Consider increasing "
                                      "--max-seq-len to fit your data." % ndiscard)
            logger.info("%d sentence pairs out of buckets", ndiscard)
            logger.info("fill up mode: %s", self.fill_up)
            logger.info("")

    def _populate_bucket_batch_sizes(self):
        """
        Compute bucket-specific batch sizes (sentences, average_words) and default bucket batch
        size.

        If sentence-based batching: number of sentences is the same for each batch, determines the
        number of words.

        If word-based batching: number of sentences for each batch is set to the multiple of number
        of devices that produces the number of words closest to the target batch size.  Average
        target sentence length (non-padding symbols) is used for word number calculations.

        Sets: self.bucket_batch_sizes
        """
        # Pre-defined bucket batch sizes
        if self.bucket_batch_sizes is not None:
            return
        # Otherwise compute here
        self.bucket_batch_sizes = [None for _ in self.buckets]
        largest_total_batch_size = 0
        for buck_idx, bucket_shape in enumerate(self.buckets):
            # Target/label length with padding
            padded_seq_len = bucket_shape[1]
            # Average target/label length excluding padding

            # This considers the mean of all datasets
            data_label_mean_len = sum([data_label_average_len_i[buck_idx] 
                     for data_label_average_len_i in self.data_label_average_len]) / self.number_languages
            average_seq_len = data_label_mean_len

            #average_seq_len = self.data_label_average_len[0][buck_idx]

            # Word-based: num words determines num sentences
            # Sentence-based: num sentences determines num words
            if self.batch_by_words:
                check_condition(padded_seq_len <= self.batch_size, "Word batch size must cover sequence lengths for all"
                                " buckets: (%d > %d)" % (padded_seq_len, self.batch_size))
                # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
                # average length
                batch_size_seq = self.batch_num_devices * round((self.batch_size / average_seq_len)
                                                                / self.batch_num_devices)
                batch_size_word = batch_size_seq * average_seq_len
            else:
                batch_size_seq = self.batch_size
                batch_size_word = batch_size_seq * average_seq_len
            self.bucket_batch_sizes[buck_idx] = BucketBatchSize(batch_size_seq, batch_size_word)
            # Track largest batch size by total elements
            largest_total_batch_size = max(largest_total_batch_size, batch_size_seq * max(*bucket_shape))


        # Final step: guarantee that largest bucket by sequence length also has largest total batch size.
        # When batching by sentences, this will already be the case.
        #TODO msobrevillac Change this part to handle multi dataset in average_seq_len = self.data_label_average_len[-1]
        if self.batch_by_words:
            padded_seq_len = max(*self.buckets[-1])
            average_seq_len = self.data_label_average_len[-1]
            while self.bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_batch_size:
                self.bucket_batch_sizes[-1] = BucketBatchSize(
                    self.bucket_batch_sizes[-1].batch_size + self.batch_num_devices,
                    self.bucket_batch_sizes[-1].average_words_per_batch + self.batch_num_devices * average_seq_len)

    def _convert_to_array(self):
        from collections import Counter

        for index in range(self.number_languages):
            max_dists = Counter()
            for i in range(len(self.data_source[index])):
                self.data_source[index][i] = np.asarray(self.data_source[index][i], dtype=self.dtype)
                self.data_target[index][i] = np.asarray(self.data_target[index][i], dtype=self.dtype)
                self.data_label[index][i] = np.asarray(self.data_label[index][i], dtype=self.dtype)
                #####
                # GCN
                self.data_src_graphs[index][i] = self._convert_to_adj_matrix(self.buckets[i][0], self.data_src_graphs[index][i])
                self.data_src_positions[index][i] = self._get_graph_positions(self.buckets[i][0], self.data_src_graphs[index][i])
                try:
                    max_dist = np.max(self.data_src_positions[index][i], axis=1)
                    for val in max_dist:
                        max_dists[val] += 1
                except ValueError:
                    max_dist = 0
                #logger.info(max_dist)
                #logger.info("SRC_METADATA SHAPE: " + str(self.data_src_metadata[i].shape))
                #####
                
                n = len(self.data_source[index][i])
                if self.number_languages == 1:
                    batch_size_seq = self.bucket_batch_sizes[i].batch_size
                else:
                    batch_size_seq = self.batch_samples[index]

                if n % batch_size_seq != 0:
                    buck_shape = self.buckets[i]
                    rest = batch_size_seq - n % batch_size_seq
                    if self.fill_up == 'pad':
                        raise NotImplementedError
                    elif self.fill_up == 'replicate':
                        logger.info("Replicating %d random sentences from bucket %s to size it to multiple of %d", rest,
                                buck_shape, batch_size_seq)
                        random_indices = np.random.randint(self.data_source[index][i].shape[0], size=rest)
                        self.data_source[index][i] = np.concatenate((self.data_source[index][i], 
                                                                   self.data_source[index][i][random_indices, :]), axis=0)
                        self.data_target[index][i] = np.concatenate((self.data_target[index][i], 
                                                                   self.data_target[index][i][random_indices, :]), axis=0)
                        self.data_label[index][i] = np.concatenate((self.data_label[index][i], 
                                                                   self.data_label[index][i][random_indices, :]), axis=0)
                        ####
                        # GCN: we add an empty list as padding
                        self.data_src_graphs[index][i] = np.concatenate((self.data_src_graphs[index][i], 
                                                                   self.data_src_graphs[index][i][random_indices, :, :]), axis=0)
                        self.data_src_positions[index][i] = np.concatenate((self.data_src_positions[index][i], 
                                                                   self.data_src_positions[index][i][random_indices]), axis=0)
                        ####
                        #logger.info('Shapes after replication')
                        #logger.info(self.data_source[i].shape)
                        #logger.info(self.data_src_metadata[i].shape)
            logger.info(max_dists)


    def _calculate_batch_samples(self, temperature):
        distribution = []
        for index in range(self.number_languages):
            distribution.append(sum([len(bucket) for bucket in self.data_source[index]]))

        prob_distribution = [d ** (1.0/temperature) for d in distribution]
        prob_distribution = [d/sum(prob_distribution) for d in prob_distribution]
        
        batch_samples = [d * self.batch_size for d in prob_distribution]

        sum_batch_size = 0
        _max = -1
        ind_max = -1
        for index in range(len(batch_samples)):
            if batch_samples[index] > _max:
                _max = batch_samples[index]
                ind_max = index
            if batch_samples[index] < 1:
                batch_samples[index] = 1
            if batch_samples[index] - int(batch_samples[index]) > 0.5:
                batch_samples[index] = math.ceil(batch_samples[index])
            else:
                batch_samples[index] = int(batch_samples[index])
            sum_batch_size += batch_samples[index]

        batch_samples[ind_max] = batch_samples[ind_max] - (sum_batch_size - self.batch_size)
        logger.info("batch sub-sampling per dataset %s", batch_samples)
        majority_index = batch_samples.index(max(batch_samples))

        return batch_samples, majority_index


                    
    def _convert_to_adj_matrix(self, bucket_size, data_src_graphs):
        """
        Graph information is in the form of an adjacency list.
        We convert this to an adjacency matrix in NumPy format.
        The matrix contains label ids.
        IMPORTANT: we add one to the label id stored in the matrix.
        This is because 0 is a valid vocab id but we want to use 0's
        to represent lack of edges instead. This means that the GCN code
        should account for this.
        """
        batch_size = len(data_src_graphs)
        new_src_graphs = np.array([np.zeros((bucket_size, bucket_size)) for sent in range(batch_size)])
        for i, graph in enumerate(data_src_graphs):
            for tup in graph:
                new_src_graphs[i][tup[0]][tup[1]] = tup[2] + 1
                # Get the id for self label
                if tup[0] == tup[1]:
                    self_id = tup[2] + 1
            # Populate diagonal, need this because pad symbols need to have a self loop
            for j in range(bucket_size):
                new_src_graphs[i][j][j] = self_id
        return new_src_graphs

    def _get_graph_positions(self, bucket_size, data_src_graphs):
        """
        Precalculate graph positions according to the distance from the root.
        """
        batch_size = data_src_graphs.shape[0]
        positions = np.array([np.ones(bucket_size) * -1000 for sent in range(batch_size)])
        for i, adj in enumerate(data_src_graphs):
            dist = 0
            curr_node = self._find_root(adj)
            positions[i][curr_node] = dist # fill root node pos
            #print(adj)
            #print(self.forward_id + 1)
            forward_mask = np.ones_like(adj) * (self.forward_id + 1)
            forward_adj = (forward_mask == adj)
            # Start recursion over the adj matrix
            
            self._fill_pos(dist+1, curr_node, positions[i], forward_adj)
        return positions

    def _find_root(self, adj):
        """
        Find root node by inspecting the adj matrix.
        """
        adj_t = np.transpose(adj)
        fallback = 0 # cycles...
        for i, row in enumerate(adj_t):
            if self.forward_id + 1 not in row:
                return i
        return fallback

    def _fill_pos(self, dist, curr_node, positions, adj):
        tups = enumerate(adj[curr_node])
        # fill positions first (BFS)
        #print(positions)
        #print(list(tups))
        for i, edge in tups:
            if edge:
                if positions[i] == -1000:
                    #print('UPDATED')
                    # not updated yet
                    positions[i] = dist
        # iterate again for recursion
        tups = enumerate(adj[curr_node])
        for i, edge in tups:
            if edge:
                if positions[i] == dist:
                    self._fill_pos(dist+1, i, positions, adj)   
        
    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_idx = [0 for _ in range(self.number_languages)]


        # shuffle indices
        for i in range(self.number_languages):
            random.shuffle(self.idx[i])

        self.indices = [[] for _ in range(self.number_languages)]
        self.nd_source = [[] for _ in range(self.number_languages)]
        self.nd_target = [[] for _ in range(self.number_languages)]
        self.nd_label = [[] for _ in range(self.number_languages)]
        #####
        # GCN
        self.nd_src_graphs = [[] for _ in range(self.number_languages)]
        self.nd_src_positions = [[] for _ in range(self.number_languages)]
        #####

        for index in range(self.number_languages):
            for i in range(len(self.data_source[index])):
                # shuffle indices within each bucket
                self.indices[index].append(np.random.permutation(len(self.data_source[index][i])))
                #logger.info("self.indices[index] %s", self.indices[index])
                self._append_ndarrays(index, i, self.indices[index][-1])

        #logger.info(len(self.nd_source))
        #logger.info(self.nd_src_graphs)
        #logger.info(self.nd_src_positions)
        #logger.info(self.nd_src_positions[1].asnumpy())
        #logger.info(self.data_src_graphs)
        #logger.info(self.data_src_positions)

    def _append_ndarrays(self, index: int, bucket: int, shuffled_indices: np.array):
        """
        Appends the actual data, selected by the given indices, to the NDArrays
        of the appropriate bucket. Use when reshuffling the data.

        :param bucket: Current bucket.
        :param shuffled_indices: Indices indicating which data to select.
        """
        self.nd_source[index].append(mx.nd.array(self.data_source[index][bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_target[index].append(mx.nd.array(self.data_target[index][bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_label[index].append(mx.nd.array(self.data_label[index][bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_src_graphs[index].append(mx.nd.array(self.data_src_graphs[index][bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_src_positions[index].append(mx.nd.array(self.data_src_positions[index][bucket].take(shuffled_indices, axis=0), dtype=self.dtype))           

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_idx[self.majority_index] != len(self.idx[self.majority_index])

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        source = None        
        target = None
        src_graphs = None
        src_positions = None
        _label = None

        i = 0 #This value will keep fixed if there is no bucketing

        for index in range(self.number_languages):
            # This in in case of balancing datasets
            if self.curr_idx[index] == len(self.idx[index]):
                self.curr_idx[index] = 0

            i, j = self.idx[index][self.curr_idx[index]]
            self.curr_idx[index] += 1

            if self.number_languages > 1:
                batch_size_seq = self.batch_samples[index]
            else:
                batch_size_seq = self.bucket_batch_sizes[i].batch_size

            source = (self.nd_source[index][i][j:j + batch_size_seq] 
                      if source is None else mx.nd.concat(source, self.nd_source[index][i][j:j + batch_size_seq], dim=0))

            target = (self.nd_target[index][i][j:j + batch_size_seq] 
                      if target is None else mx.nd.concat(target, self.nd_target[index][i][j:j + batch_size_seq], dim=0))

            src_graphs = (self.nd_src_graphs[index][i][j:j + batch_size_seq] 
                      if src_graphs is None else mx.nd.concat(src_graphs, self.nd_src_graphs[index][i][j:j + batch_size_seq], dim=0))

            src_positions = (self.nd_src_positions[index][i][j:j + batch_size_seq] 
                      if src_positions is None else mx.nd.concat(src_positions, self.nd_src_positions[index][i][j:j + batch_size_seq], dim=0))

            _label = (self.nd_label[index][i][j:j + batch_size_seq]
                      if _label is None else mx.nd.concat(_label, self.nd_label[index][i][j:j + batch_size_seq], dim=0))

  
        data = [source, target, src_graphs, src_positions]
        label = [_label]


        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        # TODO: num pad examples is not set here if fillup strategy would be padding
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """

        with open(fname, "wb") as fp:
            pickle.dump(self.idx, fp)
            pickle.dump(self.curr_idx, fp)
            np.save(fp, self.indices)

    def load_state(self, fname: str):

        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """
        with open(fname, "rb") as fp:
            self.idx = pickle.load(fp)
            self.curr_idx = pickle.load(fp)
            self.indices = np.load(fp)

        # Because of how checkpointing is done (pre-fetching the next batch in
        # each iteration), curr_idx should be always >= 1
        assert self.curr_idx >= 1
        # Right after loading the iterator state, next() should be called
        self.curr_idx -= 1

        self.nd_source = []
        self.nd_target = []
        self.nd_label = []
        for i in range(len(self.data_source)):
            self._append_ndarrays(i, self.indices[i])
