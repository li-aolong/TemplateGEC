# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from torch import nn
from fairseq import utils

from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)

from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

try:
    from transformers import BertModel, BertTokenizer, BertConfig
    has_hf = True
except ImportError:
    has_hf = False

logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 510


# class HuggingFaceBertLanguageModel(FairseqLanguageModel):
@register_model('hf_bert')
class HuggingFaceBertLanguageModel(FairseqEncoderModel):
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        if not has_hf:
            raise ImportError(
                '\n\nPlease install huggingface/transformers with:'
                '\n\n  pip install transformers'
                '\n\nOr to make local edits, install the submodule:'
                '\n\n  git submodule update --init '
                'fairseq/models/huggingface/transformers'
            )
        self.task = task
        self.classification_heads = {}
        self.args = args
        self.encoder = encoder
        #self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        # todo 先不管这些, 我们后续一个个改成对应的
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')

        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')

        parser.add_argument('--load-hf-bert-from', type=str, default='',
                            help='load huggingface pretrained bert from path')

        parser.add_argument('--load-hf-bert-config-only', action='store_true',
                            help='only load config in the path so we can get a hf model')

        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # print("In build_model !!!")
        default_architecture(args)
        assert args.load_hf_bert_from != ''
        encoder = HuggingFaceBertEncoder(args, task.dictionary)

        return cls(args, encoder, task)

    # copy it from roberta's code
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, return_all_hiddens=False,
                classification_head_name=None, **kwargs):

        src_tokens = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        x, extra = self.encoder(src_tokens, return_all_hiddens)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = HuggingFaceBertClassificationHead(
            self.args.embed_dim,  # self.args.encoder_embed_dim,
            inner_dim or self.args.embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args.quant_noise_pq,
            self.args.quant_noise_pq_block_size,
        )


class HuggingFaceBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        # logging.info("features {}: {}".format(features.shape, features))
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HuggingFaceBertEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        try:
            # Prepend the transformers submodule to the path, so that
            # it's prioritized over other installations. This allows
            # making local changes in the submodule.
            sys.path.insert(
                0, os.path.join(os.path.dirname(__file__), 'transformers', 'src')
            )
            from transformers import BertModel, BertTokenizer, BertConfig
        except ImportError:
            raise ImportError(
                '\n\nPlease install huggingface/transformers with:'
                '\n\n  pip install transformers'
                '\n\nOr to make local edits, install the submodule:'
                '\n\n  git submodule update --init '
                'fairseq/models/huggingface/transformers'
            )

        # logging.info(args)
        # raise NotImplementedError(args.load_hf_bert_from)
        load_hf_bert_from = getattr(args, 'load_hf_bert_from', '')
        assert load_hf_bert_from != ''
        model_path = load_hf_bert_from

        config = BertConfig.from_pretrained(model_path)

        #logging.info("args: {}".format(args))
        if getattr(args, 'load_hf_bert_config_only', False) is True:
            logger.info(
                "now we will init the hf_bert model from config without the weights,"
                " since we will restore the weights later")
            self.model = BertModel(config)
        else:
            logger.info("now we will init the hf_bert model from {} with all the weights".format(model_path))
            self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = dictionary.tokenizer
        self.dictionary = dictionary
        self.args = args
        self.config = config

    def forward(self, src_tokens, return_all_hiddens=False, ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states = self.model(**src_tokens)
        features = inner_states[0].float()
        return features, {'inner_states': inner_states[2] if return_all_hiddens else None}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return min(self.args.max_positions, self.model.config.max_position_embeddings - 2)


@register_model_architecture('hf_bert', 'hf_bert_base')
def default_architecture(args):
    if getattr(args, 'max_target_positions', None) is None:
        args.max_target_positions = getattr(
            args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS
        )
    args.embed_dim = getattr(args, 'embed_dim', 768)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 8)
    args.num_layers = getattr(args, 'num_layers', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.max_positions = getattr(args, 'max-positions', 510)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)