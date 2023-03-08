# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/10_data-core.ipynb.

# %% ../../nbs/10_data-core.ipynb 4
from __future__ import annotations

import gc, importlib, sys, traceback

from accelerate.logging import get_logger
from dataclasses import dataclass
from dotenv import load_dotenv
from fastai.imports import *
from fastai.losses import CrossEntropyLossFlat
from fastai.data.block import TransformBlock
from fastai.data.transforms import DataLoaders, ItemTransform, Transform
from fastai.text.data import SortedDL
from fastai.torch_core import *
from fastai.torch_imports import *
from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel, AutoModelForSequenceClassification
from transformers import logging as hf_logging
from transformers.data.data_collator import DataCollatorWithPadding

from ..utils import get_hf_objects

# %% auto 0
__all__ = ['logger', 'get_task_hf_objects', 'multiclass_tokenize_func', 'multilabel_tokenize_func', 'TextCollatorWithPadding',
           'TextInput', 'BatchDecodeTransform', 'get_blurr_tfm', 'first_blurr_tfm', 'show_batch', 'sorted_dl_func',
           'BatchTokenizeTransform', 'ItemTokenizeTransform', 'TextBlock']

# %% ../../nbs/10_data-core.ipynb 6
# silence all the HF warnings and load environment variables
warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()
logger = get_logger(__name__)

load_dotenv()

# %% ../../nbs/10_data-core.ipynb 17
def get_task_hf_objects(pretrained_model_name: str, label_names: list[str] = ["neg", "pos"], verbose: bool = False):
    """A helper function for getting the Hugging Face objects that works out of the box for most sequence classification tasks"""
    model_cls = AutoModelForSequenceClassification
    n_label_names = len(label_names)

    hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(
        pretrained_model_name, model_cls=model_cls, config_kwargs={"num_labels": n_label_names}
    )

    if verbose:
        hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)

        print("=== config ===")
        print(f"# of labels:\t{hf_config.num_labels}")
        print("")
        print("=== tokenizer ===")
        print(f"Vocab size:\t\t{hf_tokenizer.vocab_size}")
        print(f"Max # of tokens:\t{hf_tokenizer.model_max_length}")
        print(f"Attributes expected by model in forward pass:\t{hf_tokenizer.model_input_names}")

    return hf_arch, hf_config, hf_tokenizer, hf_model

# %% ../../nbs/10_data-core.ipynb 20
def multiclass_tokenize_func(
    examples,
    hf_tokenizer: PreTrainedTokenizerBase,
    text_attr: str = "text",
    text_pair_attr: str = None,
    max_length: int = None,
    padding: bool | str = True,
    truncation: bool | str = True,
    tok_kwargs: dict = {},
):
    """A tokenization function that works out of the box for most multiclassification tasks"""
    txts = [examples[text_attr]] if text_pair_attr is None else [examples[text_attr], examples[text_pair_attr]]
    return hf_tokenizer(*txts, max_length=max_length, padding=padding, truncation=truncation, **tok_kwargs)

# %% ../../nbs/10_data-core.ipynb 24
def multilabel_tokenize_func(
    examples,
    hf_tokenizer: PreTrainedTokenizerBase,
    label_attrs: list[str],
    text_attr: str = "text",
    text_pair_attr: str = None,
    max_length: int = None,
    padding: bool | str = True,
    truncation: bool | str = True,
    tok_kwargs: dict = {},
):
    """A tokenization function that works out of the box for most multilabel classification tasks"""
    txts = [examples[text_attr]] if text_pair_attr is None else [examples[text_attr], examples[text_pair_attr]]
    inputs = hf_tokenizer(*txts, max_length=max_length, padding=padding, truncation=truncation, **tok_kwargs)

    label_names = torch.stack([tensor(examples[lbl]) for lbl in label_attrs], dim=-1)
    inputs["label"] = label_names

    return inputs

# %% ../../nbs/10_data-core.ipynb 29
@dataclass
class TextCollatorWithPadding:
    def __init__(
        self,
        # A Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase,
        # The abbreviation/name of your Hugging Face transformer architecture (e.b., bert, bart, etc..)
        hf_arch: str = None,
        # A specific configuration instance you want to use
        hf_config: PretrainedConfig = None,
        # A Hugging Face model
        hf_model: PreTrainedModel = None,
        # The number of inputs expected by your model
        n_inp: int = 1,
        # Defaults to use Hugging Face's DataCollatorWithPadding(tokenizer=hf_tokenizer)
        data_collator_cls: type = DataCollatorWithPadding,
        # kwyargs specific for the instantiation of the `data_collator`
        data_collator_kwargs: dict = {},
    ):
        """A data collation function that can be used across blurr's base, low-level, and mid-level APIs"""
        store_attr()
        self.hf_tokenizer = data_collator_kwargs.pop("tokenizer", self.hf_tokenizer)
        self.data_collator = data_collator_cls(tokenizer=self.hf_tokenizer, **data_collator_kwargs)

    def __call__(self, features):
        features = L(features)
        inputs, labels, targs = [], [], []

        # features contain dictionaries
        if isinstance(features[0], dict):
            feature_keys = list(features[0].keys())
            inputs = [self._build_inputs_d(features, feature_keys)]

            input_labels = self._build_input_labels(inputs[0], features, feature_keys)
            if input_labels is not None:
                labels, targs = [input_labels], [input_labels.clone()]
        # features contains tuples, each of which can contain multiple inputs and/or targets
        elif isinstance(features[0], tuple):
            for f_idx in range(self.n_inp):
                feature_keys = list(features[0][f_idx].keys())
                inputs.append(self._build_inputs_d(features.itemgot(f_idx), feature_keys))

                input_labels = self._build_input_labels(inputs[0], features.itemgot(f_idx), feature_keys)
                labels.append(input_labels if input_labels is not None else [])

            targs = [self._proc_targets(inputs[0], list(features.itemgot(f_idx))) for f_idx in range(self.n_inp, len(features[0]))]

        return self._build_batch(inputs, labels, targs)

    # ----- utility methods -----

    # to build the inputs dictionary
    def _build_inputs_d(self, features, feature_keys):
        return {fwd_arg: list(features.attrgot(fwd_arg)) for fwd_arg in self.hf_tokenizer.model_input_names if fwd_arg in feature_keys}

    # to build the input "labels"
    def _build_input_labels(self, inputs_d, features, feature_keys):
        if "label" in feature_keys:
            labels = list(features.attrgot("label"))
            return self._proc_targets(inputs_d, labels)
        return None

    # used to give the labels/targets the right shape
    def _proc_targets(self, inputs_d, targs):
        if is_listy(targs[0]):
            targs = torch.stack([tensor(lbls) for lbls in targs])
        elif isinstance(targs[0], torch.Tensor) and len(targs[0].size()) > 0:
            targs = torch.stack(targs)
        else:
            targs = torch.tensor(targs)

        return targs

    # will properly assemble are batch given a list of inputs, labels, and targets
    def _build_batch(self, inputs, labels, targs):
        batch = []

        for input, input_labels in zip(inputs, labels):
            input_d = dict(self.data_collator(input))
            if len(input_labels) > 0:
                input_d["labels"] = input_labels

            batch.append(input_d)

        for targ in targs:
            batch.append(targ)

        return tuplify(batch)

# %% ../../nbs/10_data-core.ipynb 74
class TextInput(TensorBase):
    """The base represenation of your inputs; used by the various fastai `show` methods"""

    pass

# %% ../../nbs/10_data-core.ipynb 77
class BatchDecodeTransform(Transform):
    """A class used to cast your inputs as `input_return_type` for fastai `show` methods"""

    def __init__(
        self,
        # A Hugging Face tokenizer (not required if passing in an instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_tokenizer: PreTrainedTokenizerBase,
        # The abbreviation/name of your Hugging Face transformer architecture (not required if passing in an instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_arch: str = None,
        # A Hugging Face configuration object (not required if passing in an instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_config: PretrainedConfig = None,
        # A Hugging Face model (not required if passing in an instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_model: PreTrainedModel = None,
        # Used by typedispatched show methods
        input_return_type: type = TextInput,
        # The token ID that should be ignored when calculating the loss
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
        # Any other keyword arguments
        **kwargs,
    ):
        store_attr()
        self.kwargs = kwargs

    def decodes(self, items: dict):
        """Returns the proper object and data for show related fastai methods"""
        return self.input_return_type(items["input_ids"])

# %% ../../nbs/10_data-core.ipynb 80
def get_blurr_tfm(
    # A list of transforms (e.g., dls.after_batch, dls.before_batch, etc...)
    tfms_list: Pipeline,
    # The transform to find
    tfm_class: Transform = BatchDecodeTransform,
):
    """
    Given a fastai DataLoaders batch transforms, this method can be used to get at a transform
    instance used in your Blurr DataBlock
    """
    return next(filter(lambda el: issubclass(type(el), tfm_class), tfms_list), None)

# %% ../../nbs/10_data-core.ipynb 82
def first_blurr_tfm(
    # Your fast.ai `DataLoaders
    dls: DataLoaders,
    # The Blurr transforms to look for in order
    tfms: list[Transform] = [BatchDecodeTransform],
):
    """
    This convenience method will find the first Blurr transform required for methods such as
    `show_batch` and `show_results`. The returned transform should have everything you need to properly
    decode and 'show' your Hugging Face inputs/targets
    """
    for tfm in tfms:
        found_tfm = get_blurr_tfm(dls.before_batch, tfm_class=tfm)
        if found_tfm:
            return found_tfm

        found_tfm = get_blurr_tfm(dls.after_batch, tfm_class=tfm)
        if found_tfm:
            return found_tfm

# %% ../../nbs/10_data-core.ipynb 85
@typedispatch
def show_batch(
    # This typedispatched `show_batch` will be called for `TextInput` typed inputs
    x: TextInput,
    # Your targets
    y,
    # Your raw inputs/targets
    samples,
    # Your `DataLoaders`. This is required so as to get at the Hugging Face objects for
    # decoding them into something understandable
    dataloaders,
    # Your `show_batch` context
    ctxs=None,
    # The maximum number of items to show
    max_n=6,
    # Any truncation your want applied to your decoded inputs
    trunc_at=None,
    # Any other keyword arguments you want applied to `show_batch`
    **kwargs,
):
    # grab our tokenizer
    tfm = first_blurr_tfm(dataloaders)
    hf_tokenizer = tfm.hf_tokenizer

    # if we've included our label_names list, we'll use it to look up the value of our target(s)
    trg_label_names = tfm.kwargs["label_names"] if ("label_names" in tfm.kwargs) else None
    if trg_label_names is None and dataloaders.vocab is not None:
        trg_label_names = dataloaders.vocab

    res = L()
    n_inp = dataloaders.n_inp

    n_samples = min(max_n, dataloaders.bs)
    for idx in range(n_samples):
        input_ids = x[idx]
        rets = [hf_tokenizer.decode(input_ids, skip_special_tokens=True)[:trunc_at]]

        sample = samples[idx] if samples is not None else None
        for item_idx, item in enumerate(sample[n_inp:]):
            label = y[item_idx] if y is not None else item

            if torch.is_tensor(label):
                label = list(label.numpy()) if len(label.size()) > 0 else label.item()

            if is_listy(label):
                trg = [trg_label_names[int(idx)] for idx, val in enumerate(label) if (val == 1)] if trg_label_names else label
            else:
                trg = trg_label_names[int(item)] if trg_label_names else item

            rets.append(trg)
        res.append(tuplify(rets))

    cols = ["text"] + ["target" if (i == 0) else f"target_{i}" for i in range(len(res[0]) - n_inp)]
    display_df(pd.DataFrame(res, columns=cols)[:max_n])
    return ctxs

# %% ../../nbs/10_data-core.ipynb 87
def sorted_dl_func(
    example,
    # A Hugging Face tokenizer
    hf_tokenizer: PreTrainedTokenizerBase,
    # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. \
    # Set this to 'True' if your inputs are pre-tokenized (not numericalized)
    is_split_into_words: bool = False,
    # Any other keyword arguments you want to include during tokenization
    tok_kwargs: dict = {},
):
    """This method is used by the `SortedDL` to ensure your dataset is sorted *after* tokenization"""
    txt = None
    if isinstance(example[0], dict):
        if "input_ids" in example[0]:
            # if inputs are pretokenized
            return len(example[0]["input_ids"])
        else:
            txt = example[0]["text"]
    else:
        txt = example[0]

    return len(txt) if is_split_into_words else len(hf_tokenizer.tokenize(txt, **tok_kwargs))

# %% ../../nbs/10_data-core.ipynb 114
class BatchTokenizeTransform(Transform):
    """
    Handles everything you need to assemble a mini-batch of inputs and targets, as well as
    decode the dictionary produced as a byproduct of the tokenization process in the `encodes` method.
    """

    def __init__(
        self,
        # The abbreviation/name of your Hugging Face transformer architecture (e.b., bert, bart, etc..)
        hf_arch: str,
        # A specific configuration instance you want to use
        hf_config: PretrainedConfig,
        # A Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase,
        # A Hugging Face model
        hf_model: PreTrainedModel,
        # To control whether the "include_labels" are included in your inputs. If they are, the loss will be calculated in \
        # the model's forward function and you can simply use `PreCalculatedLoss` as your `Learner`'s loss function to use it
        include_labels: bool = True,
        # The token ID that should be ignored when calculating the loss
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
        # To control the length of the padding/truncation. It can be an integer or None, \
        # in which case it will default to the maximum length the model can accept. \
        # If the model has no specific maximum input length, truncation/padding to max_length is deactivated. \
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        max_length: int = None,
        # To control the `padding` applied to your `hf_tokenizer` during tokenization. \
        # If None, will default to 'False' or 'do_not_pad'. \
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        padding: bool | str = True,
        # To control `truncation` applied to your `hf_tokenizer` during tokenization. \
        # If None, will default to 'False' or 'do_not_truncate'. \
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        truncation: bool | str = True,
        # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. \
        # Set this to 'True' if your inputs are pre-tokenized (not numericalized) \
        is_split_into_words: bool = False,
        # Any other keyword arguments you want included when using your `hf_tokenizer` to tokenize your inputs
        tok_kwargs: dict = {},
        # Keyword arguments to apply to `BatchTokenizeTransform`
        **kwargs,
    ):
        store_attr()
        self.kwargs = kwargs

    def encodes(self, samples, return_batch_encoding=False):
        """
        This method peforms on-the-fly, batch-time tokenization of your data. In other words, your raw inputs
        are tokenized as needed for each mini-batch of data rather than requiring pre-tokenization of your full
        dataset ahead of time.
        """
        samples = L(samples)

        # grab inputs
        is_dict = isinstance(samples[0][0], dict)
        test_inp = samples[0][0]["text"] if is_dict else samples[0][0]

        if is_listy(test_inp) and not self.is_split_into_words:
            if is_dict:
                inps = [(item["text"][0], item["text"][1]) for item in samples.itemgot(0).items]
            else:
                inps = list(zip(samples.itemgot(0, 0), samples.itemgot(0, 1)))
        else:
            inps = [item["text"] for item in samples.itemgot(0).items] if is_dict else samples.itemgot(0).items

        inputs = self.hf_tokenizer(
            inps,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            is_split_into_words=self.is_split_into_words,
            return_tensors="pt",
            **self.tok_kwargs,
        )

        d_keys = inputs.keys()

        # update the samples with tokenized inputs (e.g. input_ids, attention_mask, etc...), as well as extra information
        # if the inputs is a dictionary.
        # (< 2.0.0): updated_samples = [(*[{k: inputs[k][idx] for k in d_keys}], *sample[1:]) for idx, sample in enumerate(samples)]
        updated_samples = []
        for idx, sample in enumerate(samples):
            inps = {k: inputs[k][idx] for k in d_keys}
            if is_dict:
                inps = {
                    **inps,
                    **{k: v for k, v in sample[0].items() if k not in ["text"]},
                }

            trgs = sample[1:]
            if self.include_labels and len(trgs) > 0:
                inps["label"] = trgs[0]

            updated_samples.append((*[inps], *trgs))

        if return_batch_encoding:
            return updated_samples, inputs

        return updated_samples

# %% ../../nbs/10_data-core.ipynb 117
class ItemTokenizeTransform(ItemTransform):
    split_idx = None

    def __init__(
        self,
        # A Hugging Face configuration object
        hf_config: PretrainedConfig = None,
        # A Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase = None,
        # Any keyword arguments you want your Hugging Face tokenizer to use during tokenization
        tok_kwargs: dict = {},
        # Any keyword arguments you want applied to `ItemTokenizeTransform`
        **kwargs,
    ) -> None:
        store_attr()

        if tok_kwargs.get("truncation", None) is None:
            tok_kwargs["truncation"] = True
        if tok_kwargs.get("max_length", None) is None:
            tok_kwargs["max_length"] = True

    def encodes(self, txt, **kwargs):
        inputs = self.hf_tokenizer(txt, **self.tok_kwargs)
        return dict(inputs)

# %% ../../nbs/10_data-core.ipynb 120
class TextBlock(TransformBlock):
    """The core `TransformBlock` to prepare your inputs for training in Blurr with fastai's `DataBlock` API"""

    def __init__(
        self,
        # The abbreviation/name of your Hugging Face transformer architecture (not required if passing in an \
        # instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_arch: str = None,
        # A Hugging Face configuration object (not required if passing in an \
        # instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_config: PretrainedConfig = None,
        # A Hugging Face tokenizer (not required if passing in an \
        # instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_tokenizer: PreTrainedTokenizerBase = None,
        # A Hugging Face model (not required if passing in an \
        # instance of `BatchTokenizeTransform` to `before_batch_tfm`)
        hf_model: PreTrainedModel = None,
        # Any transforms to apply when getting an item from a dataset (useufl for item-time tokenization)
        type_tfms: list[ItemTokenizeTransform] = None,
        # The "before_batch" transform you want to use if tokenizing your raw data on the fly (optional)
        tokenize_tfm: Transform = None,
        # The batch_tfm you want to decode your inputs into a type that can be used in the fastai show methods, \
        # (defaults to BatchDecodeTransform)
        batch_decode_tfm: BatchDecodeTransform = None,
        # An instance of `TextCollatorWithPadding` to use when not performing batch-time tokenization, \
        # (defaults to `TextCollatorWithPadding` when using pretokenized or item-time tokenization)
        data_collator: TextCollatorWithPadding = None,
        # To control whether the "include_labels" are included in your inputs. If they are, the loss will be calculated in \
        # the model's forward function and you can simply use `PreCalculatedLoss` as your `Learner`'s loss function to use it
        include_labels: bool = True,
        # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. \
        # Set this to `True` if your inputs are pre-tokenized (not numericalized)
        is_split_into_words: bool = False,
        # The return type your decoded inputs should be cast too (used by methods such as `show_batch`)
        input_return_type: type = TextInput,
        # The type of `DataLoader` you want created (defaults to `SortedDL`)
        dl_type: DataLoader = None,
        # Any keyword arguments you want applied to your `batch_decode_tfm` (will be set as a fastai `batch_tfms`)
        batch_decode_kwargs: dict = {},
        # Any keyword arguments you want your Hugging Face tokenizer to use during tokenization
        tok_kwargs: dict = {},
        # Any keyword arguments you want to have applied with generating text
        text_gen_kwargs: dict = {},
        # Any keyword arguments you want applied to `TextBlock`
        **kwargs,
    ):
        if (not all([hf_arch, hf_config, hf_tokenizer, hf_model])) and tokenize_tfm is None:
            raise ValueError("You must supply an hf_arch, hf_config, hf_tokenizer, hf_model -or- a tokenize_tfm")

        # if we are using a transform to tokenize our inputs, grab the HF objects from it
        if tokenize_tfm is not None:
            hf_arch = getattr(tokenize_tfm, "hf_arch", hf_arch)
            hf_config = getattr(tokenize_tfm, "hf_config", hf_config)
            hf_tokenizer = getattr(tokenize_tfm, "hf_tokenizer", hf_tokenizer)
            hf_model = getattr(tokenize_tfm, "hf_model", hf_model)
            is_split_into_words = getattr(tokenize_tfm, "is_split_into_words", is_split_into_words)
            include_labels = getattr(tokenize_tfm, "include_labels", include_labels)

        # configure our batch decode transform (used by show_batch/results methods)
        if batch_decode_tfm is None:
            batch_decode_tfm = BatchDecodeTransform(
                hf_arch=hf_arch,
                hf_config=hf_config,
                hf_tokenizer=hf_tokenizer,
                hf_model=hf_model,
                input_return_type=input_return_type,
                **batch_decode_kwargs.copy(),
            )

        # default to SortedDL using our custom sort function if no `dl_type` is specified
        if dl_type is None:
            dl_sort_func = partial(
                sorted_dl_func, hf_tokenizer=hf_tokenizer, is_split_into_words=is_split_into_words, tok_kwargs=tok_kwargs.copy()
            )
            dl_type = partial(SortedDL, sort_func=dl_sort_func)

        # build our custom `TransformBlock`
        if tokenize_tfm is None:
            if data_collator is None:
                data_collator = TextCollatorWithPadding(hf_tokenizer)
            dl_kwargs = {"create_batch": data_collator}
        else:
            dl_kwargs = {"before_batch": tokenize_tfm}

        return super().__init__(dl_type=dl_type, dls_kwargs=dl_kwargs, type_tfms=type_tfms, batch_tfms=batch_decode_tfm)
