# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/12_training-token-classification.ipynb.

# %% ../../nbs/12_training-token-classification.ipynb 4
from __future__ import annotations

import ast, gc, importlib, sys, traceback

from accelerate.logging import get_logger
from dataclasses import dataclass
from dotenv import load_dotenv
from fastai.callback.all import *
from fastai.imports import *
from fastai.learner import *
from fastai.torch_core import *
from fastai.torch_imports import *
from seqeval import metrics as seq_metrics
from transformers import logging as hf_logging

from ..data.core import first_blurr_tfm
from ..data.token_classification import TokenClassTextInput, get_token_labels_from_input_ids, get_word_labels_from_token_labels

# %% auto 0
__all__ = ['logger', 'calculate_token_class_metrics', 'TokenClassMetricsCallback', 'show_results']

# %% ../../nbs/12_training-token-classification.ipynb 6
# silence all the HF warnings and load environment variables
warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()
logger = get_logger(__name__)

load_dotenv()

# %% ../../nbs/12_training-token-classification.ipynb 53
def calculate_token_class_metrics(pred_toks, targ_toks, metric_key):
    if metric_key == "accuracy":
        return seq_metrics.accuracy_score(targ_toks, pred_toks)

    if metric_key == "precision":
        return seq_metrics.precision_score(targ_toks, pred_toks)

    if metric_key == "recall":
        return seq_metrics.recall_score(targ_toks, pred_toks)

    if metric_key == "f1":
        return seq_metrics.f1_score(targ_toks, pred_toks)

    if metric_key == "classification_report":
        return seq_metrics.classification_report(targ_toks, pred_toks)

# %% ../../nbs/12_training-token-classification.ipynb 56
class TokenClassMetricsCallback(Callback):
    """
    A fastai friendly callback that includes accuracy, precision, recall, and f1 metrics using the
    `seqeval` library.  Additionally, this metric knows how to *not* include your 'ignore_token' in it's
    calculations.

    See [here](https://github.com/chakki-works/seqeval) for more information on `seqeval`.
    """

    def __init__(self, tok_metrics=["accuracy", "precision", "recall", "f1"], **kwargs):
        self.run_before = Recorder

        store_attr(self=self, names="tok_metrics, kwargs")
        self.custom_metrics_dict = {k: None for k in tok_metrics}

        self.do_setup = True

    def setup(self):
        # one time setup code here.
        if not self.do_setup:
            return

        # grab the hf_tokenizer from the TokenClassBatchTokenizeTransform
        tfm = first_blurr_tfm(self.learn.dls)
        self.hf_tokenizer = tfm.hf_tokenizer
        self.ignore_label_token_id = tfm.ignore_token_id
        self.tok_special_symbols = list(self.hf_tokenizer.special_tokens_map.values())
        self.tok_kwargs = tfm.kwargs

        # grab our label names
        self.trg_labels = tfm.kwargs["label_names"] if ("label_names" in tfm.kwargs) else None
        if self.trg_labels is None and self.learn.dls.vocab is not None:
            self.trg_labels = self.learn.dls.vocab

        # add custom text generation specific metrics
        custom_metric_keys = self.custom_metrics_dict.keys()
        custom_metrics = L([ValueMetric(partial(self.metric_value, metric_key=k), k) for k in custom_metric_keys])
        self.learn.metrics = self.learn.metrics + custom_metrics
        self.learn.token_classification_report = None

        self.do_setup = False

    def before_fit(self):
        self.setup()

    # --- batch begin/after phases ---
    def before_batch(self):
        pass

    def after_batch(self):
        if self.training or self.learn.y is None:
            return

        # do this only for validation set
        preds = self.pred.argmax(dim=-1)
        targs = self.yb[0]  # yb is TensorText tuple, item 0 is the data

        preds_list, targets_list = [], []
        for i in range(targs.shape[0]):
            item_targs, item_preds = [], []

            for j in range(targs.shape[1]):
                if targs[i, j] != self.ignore_label_token_id:
                    item_preds.append(self.trg_labels[preds[i][j].item()])
                    item_targs.append(self.trg_labels[targs[i][j].item()])

            preds_list.append(item_preds)
            targets_list.append(item_targs)

        self.results += [(res[0], res[1]) for res in zip(preds_list, targets_list)]

    # --- validation begin/after phases ---
    def before_validate(self):
        self.results = []

    def after_validate(self):
        if len(self.results) < 1:
            return

        preds, targs = map(list, zip(*self.results))
        for k in self.custom_metrics_dict.keys():
            self.custom_metrics_dict[k] = calculate_token_class_metrics(targs, preds, metric_key=k)

        try:
            self.learn.token_classification_report = calculate_token_class_metrics(targs, preds, "classification_report")
        except ZeroDivisionError as err:
            print(f"Couldn't calcualte classification report: {err}")

    # --- for ValueMetric metrics ---
    def metric_value(self, metric_key):
        return self.custom_metrics_dict[metric_key]

# %% ../../nbs/12_training-token-classification.ipynb 58
@typedispatch
def show_results(
    # This typedispatched `show_results` will be called for `TextInput` typed inputs
    x: TokenClassTextInput,
    # Your targets
    y,
    # Your raw inputs/targets
    samples,
    # The model's predictions
    outs,
    # Your `Learner`. This is required so as to get at the Hugging Face objects for decoding them into
    # something understandable
    learner,
    # Your `show_results` context
    ctxs=None,
    # The maximum number of items to show
    max_n=6,
    # Any truncation your want applied to your decoded inputs
    trunc_at=None,
    # Any other keyword arguments you want applied to `show_results`
    **kwargs,
):
    # grab our tokenizer
    tfm = first_blurr_tfm(learner.dls)
    hf_arch, hf_tokenizer = tfm.hf_arch, tfm.hf_tokenizer

    # if we've included our labels list, we'll use it to look up the value of our target(s)
    trg_labels = tfm.kwargs["label_names"] if ("label_names" in tfm.kwargs) else None
    if trg_labels is None and learner.dls.vocab is not None:
        trg_labels = learner.dls.vocab

    res = L()
    n_inp = learner.dls.n_inp

    n_samples = min(max_n, learner.dls.bs)
    for idx in range(n_samples):
        input_ids = x[idx]
        trgs = y[idx]
        pred = outs[idx]
        sample = samples[idx] if samples is not None else None

        # align "tokens" with labels
        tok_labels = get_token_labels_from_input_ids(hf_tokenizer, input_ids, trgs, trg_labels)
        # align "words" with labels
        word_labels = get_word_labels_from_token_labels(hf_arch, hf_tokenizer, tok_labels)
        # align "words" with "predicted" labels
        if isinstance(pred[0], str):
            pred_labels = ast.literal_eval(pred[0])
        elif torch.is_tensor(pred[0]):
            pred_labels = [trg_labels[label_id] for label_id in list(pred[0].numpy())]

        word_pred_labels = [pred_lbl for lbl_id, pred_lbl in zip(trgs, pred_labels) if lbl_id != -100]
        # stringify list of (word,label) for example
        res.append(
            [
                f"{[ (word_targ[0], word_targ[1], pred_targ) for idx, (word_targ, pred_targ) in enumerate(zip(word_labels, word_pred_labels)) if (trunc_at is None or idx < trunc_at) ]}"
            ]
        )

    display_df(pd.DataFrame(res, columns=["token / target label / predicted label"])[:max_n])
    return ctxs
