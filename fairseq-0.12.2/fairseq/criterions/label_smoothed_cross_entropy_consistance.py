# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math, logging
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import torch.nn.functional as F
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
)

logger = logging.getLogger(__name__)

@dataclass
class LabelSmoothedCrossEntropyCriterionConsistanceConfig(LabelSmoothedCrossEntropyCriterionConfig):
    alpha: int = field(
        default=1,
        metadata={"help": "alpha hyperparameter for consistance loss"},
    )
    lossname: str = field(
        default='consistance',
        metadata={"help": "alpha hyperparameter for consistance loss"},
    )
    component: str = field(
        default='decoder',
        metadata={"help": "alpha hyperparameter for consistance loss"},
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    # print(loss)
    # print(nll_loss)
    # print(F.nll_loss(
    #         lprobs,
    #         target.squeeze(),
    #         ignore_index=ignore_index,
    #         reduction="mean" if reduce else "none",
    #     ))
    # exit()

    return loss, nll_loss

def mse_loss(tensor_a, tensor_b):
    # (length, bs, hs)
    # encoder_out = net_output_encoder[0]['encoder_out'][0]
    # encoder_pred_out = net_output_encoder[1]['encoder_out'][0]
    mse_loss = F.mse_loss(
        tensor_a.view(-1, tensor_a.size(-1)),
        tensor_b.view(-1, tensor_b.size(-1)),
        reduction='sum',
    )
    return mse_loss

@register_criterion(
    "label_smoothed_cross_entropy_consistance", dataclass=LabelSmoothedCrossEntropyCriterionConsistanceConfig
)
class LabelSmoothedCrossEntropyConsistanceCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        alpha=1,
        lossname=None,
        component=None,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.alpha = alpha
        self.lossname = lossname
        self.component = component
        logger.info("Alpha for consistance Loss set to {} .".format(self.alpha))

    def compute_kl_loss(self, model, net_output, net_output_pred, pad_mask=None, reduce=True):
        # mean ouptut probs for the 2 forward passes
        # mean_net_output = (net_output[0] + net_output_pred[0]) / 2
        # mean_probs = model.get_normalized_probs((mean_net_output,), log_probs=False)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs_pred = model.get_normalized_probs(net_output_pred, log_probs=True)

        probs = model.get_normalized_probs(net_output, log_probs=False)
        probs_pred = model.get_normalized_probs(net_output_pred, log_probs=False)

        # print(lprobs)
        # print(probs)
        # print(mse_loss(lprobs, lprobs_pred), mse_loss(probs, probs_pred), mse_loss(net_output[0], net_output_pred[0]))
        # exit()

        # p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
        # p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

        # og
        # p_loss = torch.nn.functional.kl_div(lprobs, mean_probs, reduction="none")
        # q_loss = torch.nn.functional.kl_div(lprobs_pred, mean_probs, reduction="none")

        p_loss = torch.nn.functional.kl_div(lprobs, probs_pred, reduction="none")
        q_loss = torch.nn.functional.kl_div(lprobs_pred, probs, reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_mse_loss(self, model, net_output, net_output_pred):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        probs_pred = model.get_normalized_probs(net_output_pred, log_probs=False)

        # print(lprobs)
        # print(probs)
        # print(mse_loss(probs_pred, probs), mse_loss(probs, probs_pred))
        # exit()
        loss = F.mse_loss(
            probs.view(-1, probs.size(-1)),
            probs_pred.view(-1, probs_pred.size(-1)),
            reduction='sum',
        )
        return loss

    def forward(self, model, sample, reduce=True, num_updates=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # original outputs
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        
        # pred outputs
        net_output_pred = model(**sample["pred"]["net_input"])
        lprobs_pred = model.get_normalized_probs(net_output_pred, log_probs=True)
        lprobs_pred = lprobs_pred.view(-1, lprobs_pred.size(-1))

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        # print(target)
        # print(self.padding_idx)
        # print(pad_mask)
        # exit()

        # encoder output
        # del sample["net_input"]['prev_output_tokens']
        # del sample["pred"]["net_input"]['prev_output_tokens']
        # encoder_output = model.encoder(**sample["net_input"])['encoder_out']
        # encoder_output_pred = model.encoder(**sample["pred"]["net_input"])['encoder_out']
        
        # nll_loss for original input
        loss_ori, nll_loss_ori = label_smoothed_nll_loss(
            lprobs,
            target.view(-1, 1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        # nll_loss for pred input
        loss_pred, nll_loss_pred = label_smoothed_nll_loss(
            lprobs_pred,
            target.view(-1, 1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )


        if self.component == 'encoder':
            net_output = net_output[2]['encoder_out']
            net_output_pred = net_output_pred[2]['encoder_out']
            pad_mask = None

        if self.lossname == 'consistance':
            loss_consistance = self.compute_kl_loss(model, net_output, net_output_pred, pad_mask=pad_mask)
        elif self.lossname == 'mse':
            loss_consistance = self.compute_mse_loss(model, net_output, net_output_pred)
        else:
            print('self.lossname error')
            exit()

        # 多计算一个encoder
        if self.component == 'encoder-decoder':
            if self.lossname == 'consistance':
                loss_consistance_encoder = self.compute_kl_loss(model, net_output[2]['encoder_out'], net_output_pred[2]['encoder_out'], pad_mask=None)
            elif self.lossname == 'mse':
                loss_consistance_encoder = self.compute_mse_loss(model, net_output[2]['encoder_out'], net_output_pred[2]['encoder_out'])
        
            loss_consistance += loss_consistance_encoder

        # loss = loss_ori + loss_pred + self.alpha * loss_consistance
        loss = 0.5 * loss_ori + 0.5 * loss_pred + 1 * loss_consistance
        # print('loss_ori: {:.3f}, loss_pred: {:.3f}, loss_consistance: {:.3f}'.format(loss_ori.data, loss_pred.data, loss_consistance.data))

        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        sample_size = sample_size * 2
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss_ori.data) if reduce else nll_loss_ori.data,
            "nll_loss_pred": utils.item(nll_loss_pred.data) if reduce else nll_loss_pred.data,
            "loss_ori": utils.item(loss_ori.data) if reduce else loss_ori.data,
            "loss_pred": utils.item(loss_pred.data) if reduce else loss_pred.data,
            "loss_consistance": utils.item(loss_consistance.data) if reduce else loss_consistance.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0) * 2,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        loss_ori = utils.item(sum(log.get("loss_ori", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss_ori",
            loss_ori / sample_size / math.log(2),
            sample_size,
            round=3,
        )

        loss_pred = utils.item(sum(log.get("loss_pred", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss_pred",
            loss_pred / sample_size / math.log(2),
            ntokens,
            round=3,
        )

        loss_consistance = utils.item(sum(log.get("loss_consistance", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss_consistance",
            loss_consistance / sample_size / math.log(2),
            ntokens,
            round=3,
        )

