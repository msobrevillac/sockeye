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
Code for inference/translation
"""
import logging
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import model
from . import utils
from . import vocab

logger = logging.getLogger(__name__)


class InferenceModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param model_folder: Folder to load model from.
    :param context: MXNet context to bind modules to.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in model_folder.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    """

    def __init__(self,
                 model_folder: str,
                 context: mx.context.Context,
                 fused: bool,
                 edge_vocab_size: int,
                 beam_size: int,
                 checkpoint: Optional[int] = None,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH) -> None:
        self.model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", self.model_version)
        utils.check_version(self.model_version)
        config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
        super().__init__(config)

        self.fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)

        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')
        self.tensor_dim = edge_vocab_size
        self.beam_size = beam_size
        self.softmax_temperature = softmax_temperature
        self.encoder_batch_size = 1
        self.context = context

        self._build_model_components(fused)

        self.max_input_length, self.get_max_output_length = get_max_input_output_length([self],
                                                                                        max_output_length_num_stds)

        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_data_shapes_cache = None  # type: Optional[Dict]

    def initialize(self, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_input_length = max_input_length
        logger.info(self.max_input_length)
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

        logger.info("BEFORE ENCODER MODULE")
        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()
        logger.info("AFTER DECODER MODULE")
        
        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
        logger.info("BEFORE ENCODER BIND")
        logger.info(self.encoder_module)
        logger.info(max_encoder_data_shapes)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        logger.info("AFTER ENCODER BIND")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.fname_params)
        self.encoder_module.init_params(arg_params=self.params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, allow_missing=False)
        logger.info("FINISHED INIT")

    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_graphs = mx.sym.Variable(C.SOURCE_GRAPHS_NAME)
            source_positions = mx.sym.Variable(C.SOURCE_POSITIONS_NAME)
            source_length = utils.compute_lengths(source)

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, source_seq_len,
                                                           metadata=(source_graphs, source_positions))
            # TODO(fhieber): Consider standardizing encoders to return batch-major data to avoid this line.
            source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len)

            data_names = [C.SOURCE_NAME, C.SOURCE_GRAPHS_NAME, C.SOURCE_POSITIONS_NAME]
            label_names = []  # type: List[str]
            return mx.sym.Group(decoder_init_states), data_names, label_names

        default_bucket_key = self.max_input_length
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int]]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the source sequence
        and the current length of the target sequence.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int]):
            source_max_len, target_max_len = bucket_key
            source_encoded_seq_len = self.encoder.get_encoded_seq_len(source_max_len)

            self.decoder.reset()
            prev_word_ids = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables()
            state_names = [state.name for state in states]

            logits, attention_probs, states = self.decoder.decode_step(prev_word_ids,
                                                                       target_max_len,
                                                                       source_encoded_seq_len,
                                                                       *states)
            if self.softmax_temperature is not None:
                logits /= self.softmax_temperature

            softmax = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_NAME] + state_names
            label_names = []  # type: List[str]
            return mx.sym.Group([softmax, attention_probs] + states), data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = (self.max_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_encoder_data_shapes(self, bucket_key: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(self.encoder_batch_size, bucket_key),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(name=C.SOURCE_GRAPHS_NAME, 
                               shape=(self.encoder_batch_size, bucket_key, bucket_key),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(name=C.SOURCE_POSITIONS_NAME,
                               shape=(self.encoder_batch_size, bucket_key),
                               layout=C.BATCH_MAJOR)]


    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.
        Caches results for bucket_keys if called iteratively.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        return self.decoder_data_shapes_cache.setdefault(
            bucket_key,
            [mx.io.DataDesc(C.TARGET_NAME, (self.beam_size, target_max_length), layout="NT")] +
            self.decoder.state_shapes(self.beam_size,
                                      self.encoder.get_encoded_seq_len(source_max_length),
                                      self.encoder.get_num_hidden()))

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int,
                    source_graph: mx.nd.NDArray,
                    source_positions: mx.nd.NDArray) -> List[mx.nd.NDArray]:
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens.
        :param source_graph: Graph input.
        :param source_max_length: Bucket key.
        :return: Encoded source, source length, initial decoder hidden state, initial decoder hidden states.
        """
        batch = mx.io.DataBatch(data=[source, source_graph, source_positions],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length))

#        batch = mx.io.DataBatch(data=[source, source_length, source_graph], label=None,
#                                bucket_key=bucket_key,
#                                provide_data=[
#        mx.io.DataDesc(name=C.SOURCE_NAME, shape=(self.encoder_batch_size, bucket_key),
#                       layout=C.BATCH_MAJOR),
#        mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(self.encoder_batch_size,),
#                       layout=C.BATCH_MAJOR),
#        mx.io.DataDesc(name=C.SOURCE_GRAPHS_NAME, shape=(self.encoder_batch_size, self.tensor_dim,
#                                                         bucket_key, bucket_key),
#                       layout=C.BATCH_MAJOR)])

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()
        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.broadcast_axis(s, axis=0, size=self.beam_size) for s in decoder_states]
        return decoder_states

    def run_decoder(self,
                    sequences: mx.nd.NDArray,
                    bucket_key: Tuple[int, int],
                    model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :return: Probability distribution over next word, attention scores, updated model state.
        """
        batch = mx.io.DataBatch(
            data=[sequences.as_in_context(self.context)] + model_state.states,
            label=None,
            bucket_key=bucket_key,
            provide_data=self._get_decoder_data_shapes(bucket_key))
        self.decoder_module.forward(data_batch=batch, is_train=False)
        probs, attention_probs, *model_state.states = self.decoder_module.get_outputs()
        return probs, attention_probs, model_state

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        if self.config.config_data.max_observed_source_seq_len is not None:
            return self.config.config_data.max_observed_source_seq_len
        else:
            return self.config.max_seq_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        if self.config.config_data.max_observed_target_seq_len is not None:
            return self.config.config_data.max_observed_target_seq_len
        else:
            return self.config.max_seq_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.length_ratio_std


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                beam_size: int,
                model_folders: List[str],
                edge_vocab: Dict[str, int],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH) \
        -> Tuple[List[InferenceModel], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param model_folders: List of model folders to load models from.
    :param edge_vocab_size: Size of edge vocabulary.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations to add to mean target-source length ratio to compute maximum output length.

    :return: List of models, source vocabulary, target vocabulary.
    """
    models, source_vocabs, target_vocabs = [], [], []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    for model_folder, checkpoint in zip(model_folders, checkpoints):
        source_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_SRC_NAME)))
        target_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_TRG_NAME)))
        model = InferenceModel(model_folder=model_folder,
                               context=context,
                               fused=False,
                               edge_vocab_size=len(edge_vocab),
                               beam_size=beam_size,
                               softmax_temperature=softmax_temperature,
                               checkpoint=checkpoint)                               
        logger.info("LOADED MODEL")
        models.append(model)

    utils.check_condition(all(set(vocab.items()) == set(source_vocabs[0].items()) for vocab in source_vocabs),
                          "Source vocabulary ids do not match")
    utils.check_condition(all(set(vocab.items()) == set(target_vocabs[0].items()) for vocab in target_vocabs),
                          "Target vocabulary ids do not match")

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = get_max_input_output_length(models,
                                                                       max_output_length_num_stds,
                                                                       max_input_len)
    for model in models:
        model.initialize(max_input_len, get_max_output_length)

    return models, source_vocabs[0], target_vocabs[0], edge_vocab


def get_max_input_output_length(models: List[InferenceModel], num_stds: int,
                                max_input_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param max_input_len: An optional overwrite of the maximum input length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)

    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
    else:
        factor = max_mean + (max_std * num_stds)

    supported_max_seq_len_source = min((model.max_supported_seq_len_source for model in models
                                        if model.max_supported_seq_len_source is not None),
                                       default=None)
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)

    training_max_seq_len_source = min(model.training_max_seq_len_source for model in models)

    if max_input_len is None:
        # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
        # constraint. This is for example the case for learned positional embeddings, which are only defined for the
        # maximum source and target sequence length observed during training.
        if supported_max_seq_len_source is not None and supported_max_seq_len_target is None:
            max_input_len = supported_max_seq_len_source
        elif supported_max_seq_len_source is None and supported_max_seq_len_target is not None:
            if np.ceil(factor * training_max_seq_len_source) > supported_max_seq_len_target:
                max_input_len = int(np.floor(supported_max_seq_len_target / factor))
            else:
                max_input_len = training_max_seq_len_source
        elif supported_max_seq_len_source is not None or supported_max_seq_len_target is not None:
            if np.ceil(factor * supported_max_seq_len_source) > supported_max_seq_len_target:
                max_input_len = int(np.floor(supported_max_seq_len_target / factor))
            else:
                max_input_len = supported_max_seq_len_source
        else:
            # Any source/target length is supported and max_input_len was not manually set, therefore we use the
            # maximum length from training.
            max_input_len = training_max_seq_len_source

    def get_max_output_length(input_length: int):
        return int(np.ceil(factor * input_length))

    return max_input_len, get_max_output_length


TranslatorInput = NamedTuple('TranslatorInput', [
    ('id', int),
    ('sentence', str),
    ('tokens', List[str]),
    ('graph', List[Tuple[int, int, int]])
])
"""
Required input for Translator.

:param id: Sentence id.
:param sentence: Input sentence.
:param tokens: List of input tokens.
"""

TranslatorOutput = NamedTuple('TranslatorOutput', [
    ('id', int),
    ('translation', str),
    ('tokens', List[str]),
    ('attention_matrix', np.ndarray),
    ('score', float),
])
"""
Output structure from Translator.

:param id: Id of input sentence.
:param translation: Translation string without sentence boundary tokens.
:param tokens: List of translated tokens.
:param attention_matrix: Attention matrix. Shape: (target_length, source_length).
:param score: Negative log probability of generated translation.
"""

TokenIds = List[int]
Translation = NamedTuple('Translation', [
    ('target_ids', TokenIds),
    ('attention_matrix', np.ndarray),
    ('score', float)
])


class ModelState:
    """
    A ModelState encapsulates information about the decoder states of an InferenceModel.
    """

    def __init__(self, states: List[mx.nd.NDArray]) -> None:
        self.states = states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        self.states = [mx.nd.take(ds, best_hyp_indices) for ds in self.states]


class LengthPenalty:
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016.

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def __call__(self, lengths: Union[mx.nd.NDArray, int, float]) -> Union[mx.nd.NDArray, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A scalar or a matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        if self.alpha == 0.0:
            if isinstance(lengths, mx.nd.NDArray):
                # no length penalty:
                return mx.nd.ones_like(lengths)
            else:
                return 1.0
        else:
            # note: we avoid unnecessary addition or pow operations
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


def _concat_translations(translations: List[Translation], start_id: int, stop_ids: Set[int],
                         length_penalty: LengthPenalty) -> Translation:
    """
    Combine translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol, attention_matrix), score and length.
    :param start_id: The EOS symbol.
    :param translations: The BOS symbols.
    :return: A concatenation if the translations with a score.
    """
    # Concatenation of all target ids without BOS and EOS
    target_ids = [start_id]
    attention_matrices = []
    for idx, translation in enumerate(translations):
        assert translation.target_ids[0] == start_id
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids[1:])
            attention_matrices.append(translation.attention_matrix[1:, :])
        else:
            if translation.target_ids[-1] in stop_ids:
                target_ids.extend(translation.target_ids[1:-1])
                attention_matrices.append(translation.attention_matrix[1:-1, :])
            else:
                target_ids.extend(translation.target_ids[1:])
                attention_matrices.append(translation.attention_matrix[1:, :])

    # Combine attention matrices:
    attention_shapes = [attention_matrix.shape for attention_matrix in attention_matrices]
    # Adding another row for the empty BOS alignment vector
    bos_align_shape = np.asarray([1, 0])
    attention_matrix_combined = np.zeros(np.sum(np.asarray(attention_shapes), axis=0) + bos_align_shape)

    # We start at position 1 as position 0 is for the BOS, which is kept zero
    pos_t, pos_s = 1, 0
    for attention_matrix, (len_t, len_s) in zip(attention_matrices, attention_shapes):
        attention_matrix_combined[pos_t:pos_t + len_t, pos_s:pos_s + len_s] = attention_matrix
        pos_t += len_t
        pos_s += len_s

    # Unnormalize + sum and renormalize the score:
    score = sum(translation.score * length_penalty(len(translation.target_ids))
                for translation in translations)
    score = score / length_penalty(len(target_ids))
    return Translation(target_ids, attention_matrix_combined, score)


class Translator:
    """
    Translator uses one or several models to translate input.
    It holds references to vocabularies to takes care of encoding input strings as word ids and conversion
    of target ids into a translation string.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param models: List of models.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 bucket_source_width: int,
                 bucket_target_width: int,
                 length_penalty: LengthPenalty,
                 models: List[InferenceModel],
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int],
                 vocab_edge: Dict[str, int]) -> None:
        self.context = context
        self.length_penalty = length_penalty
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.vocab_edge = vocab_edge
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}  # type: Set[int]
        self.models = models
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        # after models are loaded we ensured that they agree on max_input_length and max_output_length
        self.max_input_length = self.models[0].max_input_length
        max_output_length = self.models[0].get_max_output_length(self.max_input_length)
        if bucket_source_width > 0:
            self.buckets_source = data_io.define_buckets(self.max_input_length, step=bucket_source_width)
        else:
            self.buckets_source = [self.max_input_length]
        if bucket_target_width > 0:
            self.buckets_target = data_io.define_buckets(max_output_length, step=bucket_target_width)
        else:
            self.buckets_target = [max_output_length]
        self.pad_dist = mx.nd.full((self.beam_size, len(self.vocab_target)), val=np.inf, ctx=self.context)
        logger.info("Translator (%d model(s) beam_size=%d ensemble_mode=%s "
                    "buckets_source=%s buckets_target=%s)",
                    len(self.models),
                    self.beam_size,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.buckets_source,
                    self.buckets_target)

    @staticmethod
    def _get_interpolation_func(ensemble_mode):
        if ensemble_mode == 'linear':
            return Translator._linear_interpolation
        elif ensemble_mode == 'log_linear':
            return Translator._log_linear_interpolation
        else:
            raise ValueError("unknown interpolation type")

    @staticmethod
    def _linear_interpolation(predictions):
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([mx.nd.log(p) for p in predictions])
        return -mx.nd.log(mx.nd.softmax(log_probs))

    @staticmethod
    def make_input(sentence_id: int, sentence: str, graph, edge_vocab) -> TranslatorInput:
        """
        Returns TranslatorInput from input_string

        :param sentence_id: Input sentence id.
        :param sentence: Input sentence.
        :param graph: Input graph.
        :param edge_vocab: Edge label vocabulary.
        :return: Input for translate method.
        """
        tokens = list(data_io.get_tokens(sentence))
        edge_list = list(data_io.get_tokens(graph))
        edges = data_io.process_edges(edge_list, edge_vocab)

        return TranslatorInput(id=sentence_id, sentence=sentence.rstrip(), tokens=tokens, graph=edges)

    def translate(self, trans_input: TranslatorInput) -> TranslatorOutput:
        """
        Translates a TranslatorInput and returns a TranslatorOutput

        :param trans_input: TranslatorInput as returned by make_input().
        :return: translation result.
        """
        if not trans_input.tokens:
            return TranslatorOutput(id=trans_input.id,
                                    translation="",
                                    tokens=[""],
                                    attention_matrix=np.asarray([[0]]),
                                    score=-np.inf)
        
        if len(trans_input.tokens) > self.max_input_length:
            logger.debug("Input (%d) exceeds max input length (%d). Splitting into chunks of size %d.",
                         len(trans_input.tokens), self.buckets_source[-1], self.max_input_length)
            token_chunks = utils.chunks(trans_input.tokens, self.max_input_length)
            logger.debug("WARNING: graphs will also be split, and there is no guarantee the splits will be consistent!!!")
            graph_chunks = utils.chunks(trans_input.graph, self.max_input_length)
            translations = [self.translate_nd(*self._get_inference_input(tokens, graphs))
                            for tokens, graphs in zip(token_chunks, graph_chunks)]
            translation = self._concat_translations(translations)
            return self._make_result(trans_input, translation)
        else:
            return self._make_result(trans_input, self.translate_nd(*self._get_inference_input(trans_input.tokens, trans_input.graph)))

    def _get_inference_input(self, tokens: List[str], graph) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, int]:
        """
        Returns NDArray of source ids (shape=(1, bucket_key)) and corresponding bucket_key.

        :param tokens: List of input tokens.
        :return NDArray of source ids and bucket key.
        """
        bucket_key = data_io.get_bucket(len(tokens), self.buckets_source)

        utils.check_condition(C.PAD_ID == 0, "pad id should be 0")
        source = mx.nd.zeros((1, bucket_key))
        ids = data_io.tokens2ids(tokens, self.vocab_source)
        for i, wid in enumerate(ids):
            source[0, i] = wid

        ########
        # GCN
        new_graph = mx.nd.zeros((1, bucket_key, bucket_key))
        # gaaaaaah!!!!
        self_id = 3
        for tup in graph:
            if (tup[0] < bucket_key) and (tup[1] < bucket_key):
                # Stripping for graphs as well
                #new_graph[0][tup[2]][tup[0]][tup[1]] = 1.0
                new_graph[0][tup[0]][tup[1]] = tup[2] + 1
                # Get the id for self label
                if tup[0] == tup[1]:
                    self_id = tup[2] + 1
        # Populate diagonal, need this because pad symbols need to have a self loop
        for j in range(bucket_key):
            new_graph[0][j][j] = self_id

        positions = self._get_graph_positions(bucket_key, new_graph.asnumpy())
        positions = mx.nd.array(positions)
        ########
        #logger.info(source)
        #logger.info(bucket_key)
        #logger.info(new_graph)
        #logger.info(new_graph.asnumpy())
        #logger.info(positions.asnumpy())

        return source, bucket_key, new_graph, positions


    def _get_graph_positions(self, bucket_key, graph):
        """
        Precalculate graph positions according to the distance from the root.
        """
        ##############
        # !!!!!!!!!!
        self.forward_id = 1
        ###############
        positions = np.ones((1, bucket_key)) * 1000
        adj = graph[0]
        dist = 0
        curr_node = self._find_root(adj)
        positions[0][curr_node] = dist # fill root node pos
        forward_mask = np.ones_like(adj) * (self.forward_id + 1)
        forward_adj = (forward_mask == adj)
        # Start recursion over the adj matrix
        self._fill_pos(dist+1, curr_node, positions[0], forward_adj)
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
        for i, edge in tups:
            if edge:
                if positions[i] == 1000:
                    # not updated yet
                    positions[i] = dist
        # iterate again for recursion
        tups = enumerate(adj[curr_node])
        for i, edge in tups:
            if edge:
                if positions[i] == dist:
                    self._fill_pos(dist+1, i, positions, adj)  

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        # remove special sentence start symbol (<s>) from the output:
        target_ids = translation.target_ids[1:]
        attention_matrix = translation.attention_matrix[1:, :]

        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]
        target_string = C.TOKEN_SEPARATOR.join(
            target_token for target_id, target_token in zip(target_ids, target_tokens) if
            target_id not in self.stop_ids)
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]

        return TranslatorOutput(id=trans_input.id,
                                translation=target_string,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=translation.score)

    def _concat_translations(self, translations: List[Translation]) -> Translation:
        """
        Combine translations through concatenation.

        :param translations: A list of translations (sequence, attention_matrix), score and length.
        :return: A concatenation if the translations with a score.
        """
        return _concat_translations(translations, self.start_id, self.stop_ids, self.length_penalty)

    def translate_nd(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     source_graph: mx.nd.NDArray,
                     source_positions: mx.nd.NDArray) -> Translation:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (1, bucket_key).
        :param source_length: Bucket key.

        :return: Sequence of translated ids, attention matrix, length-normalized negative log probability.
        """
        return self._get_best_from_beam(*self._beam_search(source, source_length, source_graph, source_positions))

    def _encode(self, source: mx.nd.NDArray,
                source_length: int,
                source_graph: mx.nd.NDArray,
                source_positions: mx.nd.NDArray) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param source: Source ids. Shape: (1, bucket_key).
        :param source_length: Bucket key.
        :param source_graph: Input graph.
        :param source_positions: Input graph positions
        :return: List of ModelStates.
        """
        return [ModelState(states=m.run_encoder(source, source_length, source_graph, source_positions)) for m in self.models]

    def _decode_step(self,
                     sequences: mx.nd.NDArray,
                     t: int,
                     source_length: int,
                     max_output_length: int,
                     states: List[ModelState]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState]]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param sequences: Sequences of current hypotheses. Shape: (beam_size, max_output_length).
        :param t: Beam search iteration.
        :param source_length: Length of the input sequence.
        :param max_output_length: Maximum output length.
        :param: List of model states.
        :return: (probs, attention scores, list of model states)
        """
        bucket_key = (source_length, max_output_length)
        # bucket target max length based on beam_search progress
        if len(self.buckets_target) > 1:
            target_max_length = data_io.get_bucket(t, self.buckets_target)
            if target_max_length < max_output_length:
                bucket_key = (source_length, target_max_length)
                sequences = mx.nd.slice_axis(sequences, axis=1, begin=0, end=target_max_length)

        model_probs, model_attention_probs, model_states = [], [], []
        for model, state in zip(self.models, states):
            probs, attention_probs, state = model.run_decoder(sequences, bucket_key, state)
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states.append(state)
        probs, attention_probs = self._combine_predictions(model_probs, model_attention_probs)
        return probs, attention_probs, model_states

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(probs[0])
        else:
            neg_logprobs = self.interpolation_func(probs)
        return neg_logprobs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     source_graph: mx.nd.NDArray,
                     source_positions: mx.nd.NDArray) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray]:
        """
        Translates a single sentence using beam search.

        :param source: Source ids. Shape: (1, bucket_key).
        :param source_length: Source length.
        :param source_graph: Source graph.
        :return List of lists of word ids, list of attentions, array of accumulated length-normalized
                negative log-probs.
        """
        #logger.info("BEAM SEARCH")
        #logger.info(source.asnumpy())
        #logger.info(source_graph.asnumpy())
        #logger.info(source_positions.asnumpy())
        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)

        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length)

        # sequences: (beam_size, output_length), pre-filled with <s> symbols on index 0
        sequences = mx.nd.array(np.full((self.beam_size, max_output_length), C.PAD_ID), dtype='int32', ctx=self.context)
        sequences[:, 0] = self.start_id

        lengths = mx.nd.ones((self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.beam_size,), dtype='int32', ctx=self.context)

        # attentions: (beam_size, output_length, encoded_source_length)
        attentions = mx.nd.zeros((self.beam_size, max_output_length, encoded_source_length), ctx=self.context)

        # best_hyp_indices: row indices of smallest scores (ascending).
        best_hyp_indices = mx.nd.zeros((self.beam_size,), ctx=self.context)
        # best_word_indices: column indices of smallest scores (ascending).
        best_word_indices = mx.nd.zeros((self.beam_size,), ctx=self.context, dtype='int32')
        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.beam_size, 1), ctx=self.context)

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # (0) encode source sentence
        model_states = self._encode(source, source_length, source_graph, source_positions)

        for t in range(1, max_output_length):

            # (1) obtain next predictions and advance models' state
            # scores: (beam_size, target_vocab_size)
            # attention_scores: (beam_size, bucket_key)
            scores, attention_scores, model_states = self._decode_step(sequences,
                                                                       t,
                                                                       source_length,
                                                                       max_output_length,
                                                                       model_states)

            # (2) compute length-normalized accumulated scores in place
            if t == 1:  # only one hypothesis at t==1
                scores = scores[:1] / self.length_penalty(lengths[:1])
            else:
                # renormalize scores by length ...
                scores = (scores + scores_accumulated * self.length_penalty(lengths - 1)) / self.length_penalty(lengths)
                # ... but not for finished hyps.
                # their predicted distribution is set to their accumulated scores at C.PAD_ID.
                #logger.info("PADDST")
                #logger.info(self.pad_dist[:, C.PAD_ID].asnumpy())
                #scores_accumulated = scores_accumulated.reshape(shape=(-1,))
                #logger.info(scores_accumulated.asnumpy())
                self.pad_dist[:, C.PAD_ID] = scores_accumulated.reshape(shape=(-1,))
                # this is equivalent to doing this in numpy:
                #   self.pad_dist[finished, :] = np.inf
                #   self.pad_dist[finished, C.PAD_ID] = scores_accumulated[finished]
                scores = mx.nd.where(finished, self.pad_dist, scores)

            # (3) get beam_size winning hypotheses
            # TODO(fhieber): once mx.nd.topk is sped-up no numpy conversion necessary anymore.
            (best_hyp_indices[:], best_word_indices_np), scores_accumulated_np = utils.smallest_k(scores.asnumpy(),
                                                                                                  self.beam_size)
            scores_accumulated[:] = np.expand_dims(scores_accumulated_np, axis=1)
            best_word_indices[:] = best_word_indices_np

            # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending)
            sequences = mx.nd.take(sequences, best_hyp_indices)
            lengths = mx.nd.take(lengths, best_hyp_indices)
            finished = mx.nd.take(finished, best_hyp_indices)
            attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
            attentions = mx.nd.take(attentions, best_hyp_indices)

            # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
            #logger.info(sequences.asnumpy())
            #logger.info(best_word_indices.asnumpy())
            sequences[:, t] = best_word_indices
            #sequences[:, t] = mx.nd.expand_dims(best_word_indices, axis=1)
            attentions[:, t, :] = attention_scores
            #attentions[:, t, :] = mx.nd.expand_dims(attention_scores, axis=1)
            lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')

            # (6) determine which hypotheses in the beam are now finished
            finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
            if mx.nd.sum(finished).asscalar() == self.beam_size:  # all finished
                break

            # (7) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices)

        return sequences, attentions, scores_accumulated, lengths

    @staticmethod
    def _get_best_from_beam(sequences: mx.nd.NDArray,
                            attention_lists: mx.nd.NDArray,
                            accumulated_scores: mx.nd.NDArray,
                            lengths: mx.nd.NDArray) -> Translation:
        """
        Return the best (aka top) entry from the n-best list.

        :param sequences: Array of word ids. Shape: (beam_size, bucket_key).
        :param attention_lists: Array of attentions over source words. Shape: (length, bucket_key).
        :param accumulated_scores: Array of length-normalized negative log-probs.
        :return: Top sequence, top attention matrix, top accumulated score (length-normalized negative log-probs)
                 and length.
        """
        # sequences & accumulated scores are in latest 'k-best order', thus 0th element is best
        best = 0
        length = int(lengths[best].asscalar())
        sequence = sequences[best][:length].asnumpy().tolist()
        # attention_matrix: (target_seq_len, source_seq_len)
        attention_matrix = np.stack(attention_lists[best].asnumpy()[:length, :], axis=0)
        score = accumulated_scores[best].asscalar()
        return Translation(sequence, attention_matrix, score)