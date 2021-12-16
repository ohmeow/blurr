# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_data-core.ipynb (unless otherwise specified).

__all__ = ['HF_BaseInput', 'HF_BeforeBatchTransform', 'HF_AfterBatchTransform', 'blurr_sort_func', 'OverflowDL',
           'HF_TextBlock', 'BlurrBatchCreator', 'BlurrBatchTransform', 'BlurrDataLoader', 'get_blurr_tfm',
           'first_blurr_tfm', 'preproc_hf_dataset']

# Cell
import os, inspect
from dataclasses import dataclass
from functools import reduce, partial
from typing import Any, Callable, List, Optional, Union, Type

from fastcore.all import *
from fastai.data.block import TransformBlock
from fastai.data.core import Datasets, DataLoader, DataLoaders, TfmdDL
from fastai.imports import *
from fastai.losses import CrossEntropyLossFlat
from fastai.text.data import SortedDL
from fastai.torch_core import *
from fastai.torch_imports import *
from transformers import DataCollatorWithPadding, logging, PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel

from ..utils import BLURR

logging.set_verbosity_error()


# Cell
class HF_BaseInput(TensorBase):
    """The base represenation of your inputs; used by the various fastai `show` methods"""

    def show(
        self,
        # A Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase,
        # The "context" associated to the current `show_batch/results` call
        ctx=None,
        # Any truncation you want to apply to the decoded tokenized inputs
        trunc_at: int = None,
        # A decoded string of your tokenized inputs (input_ids)
    ) -> str:
        input_ids = self.cpu().numpy()
        decoded_input = str(hf_tokenizer.decode(input_ids, skip_special_tokens=True))[:trunc_at]

        return show_title(decoded_input, ctx=ctx, label="text")



# Cell
class HF_BeforeBatchTransform(Transform):
    """Handles everything you need to assemble a mini-batch of inputs and targets, as well as
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
        # To control the length of the padding/truncation. It can be an integer or None,
        # in which case it will default to the maximum length the model can accept. If the model has no
        # specific maximum input length, truncation/padding to max_length is deactivated.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        max_length: int = None,
        # To control the `padding` applied to your `hf_tokenizer` during tokenization. If None, will default to
        # `False` or `'do_not_pad'.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        padding: Union[bool, str] = True,
        # To control `truncation` applied to your `hf_tokenizer` during tokenization. If None, will default to
        # `False` or `do_not_truncate`.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        truncation: Union[bool, str] = True,
        # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. Set this to `True`
        # if your inputs are pre-tokenized (not numericalized)
        is_split_into_words: bool = False,
        # Any other keyword arguments you want included when using your `hf_tokenizer` to tokenize your inputs
        tok_kwargs: dict = {},
        # Keyword arguments to apply to `HF_BeforeBatchTransform`
        **kwargs
    ):
        store_attr(self=self, names="hf_arch, hf_config, hf_tokenizer, hf_model")
        store_attr(self=self, names="max_length, padding, truncation, is_split_into_words, tok_kwargs")
        store_attr(self=self, names="kwargs")

    def encodes(self, samples, return_batch_encoding = False):  # A subset of data to put into a mini-batch
        """This method peforms on-the-fly, batch-time tokenization of your data. In other words, your raw inputs
        are tokenized as needed for each mini-batch of data rather than requiring pre-tokenization of your full
        dataset ahead of time.
        """
        samples = L(samples)

        # grab inputs
        if is_listy(samples[0][0]) and not self.is_split_into_words:
            inps = list(zip(samples.itemgot(0, 0), samples.itemgot(0, 1)))
        else:
            inps = samples.itemgot(0).items

        # tokenize
        tok_d = self.hf_tokenizer(
            inps,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            is_split_into_words=self.is_split_into_words,
            return_tensors="pt",
            **self.tok_kwargs
        )

        # update the samples with tokenized inputs (e.g. input_ids, attention_mask, etc...), ensureing that if
        # "overflow_to_sample_mapping" = True we include each sample chunk
        d_keys = tok_d.keys()
        updated_samples = []
        if ("overflow_to_sample_mapping" in d_keys):
            for idx, seq_idx in enumerate(tok_d["overflow_to_sample_mapping"]):
                s = (*[{k: tok_d[k][idx] for k in d_keys}], *samples[seq_idx][1:])
                updated_samples.append(s)
        else:
            updated_samples = [(*[{k: tok_d[k][idx] for k in d_keys}], *sample[1:]) for idx, sample in enumerate(samples)]

        if (return_batch_encoding):
            return updated_samples, tok_d

        return updated_samples


# Cell
class HF_AfterBatchTransform(Transform):
    """A class used to cast your inputs into something understandable in fastai `show` methods"""

    def __init__(
        self,
        # A Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase,
        # The return type your decoded inputs should be cast too (used by methods such as `show_batch`)
        input_return_type: Type = HF_BaseInput,
    ):
        store_attr(self=self, names="hf_tokenizer, input_return_type")

    def decodes(
        self,
        # The encoded samples for your batch. `input_ids` will be pulled out of your dictionary of Hugging Face
        # inputs, cast to `self.input_return_type` and returned for methods such as `show_batch`
        encoded_samples: Type,
    ):
        """Returns the proper object and data for show related fastai methods"""
        if isinstance(encoded_samples, dict):
            return self.input_return_type(encoded_samples["input_ids"], hf_tokenizer=self.hf_tokenizer)
        return encoded_samples


# Cell
def blurr_sort_func(
    example,
    # A Hugging Face tokenizer
    hf_tokenizer: PreTrainedTokenizerBase,
    # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. Set this to `True`
    # if your inputs are pre-tokenized (not numericalized)
    is_split_into_words: bool = False,
    # Any other keyword arguments you want to include during tokenization
    tok_kwargs: dict = {},
):
    """This method is used by the `SortedDL` to ensure your dataset is sorted *after* tokenization"""
    if is_split_into_words:
        return len(example[0])
    return len(hf_tokenizer.tokenize(example[0], **tok_kwargs))


# Cell
@delegates(TfmdDL)
class OverflowDL(SortedDL):
    def __init__(self, dataset, sort_func=None, res=None, overflow_map_key="overflow_to_sample_mapping", **kwargs):
        super().__init__(dataset, sort_func=sort_func, res=res, **kwargs)
        self.overflow_map_key = overflow_map_key
        self.batch_items = None

    def create_batches(self, samps):
        if self.dataset is not None:
            self.it = iter(self.dataset)
        res = filter(lambda o: o is not None, map(self.do_item, samps))

        for b in map(self.do_batch, self.chunkify(res)):
            while self._n_batch_items() >= self.bs:
                yield self._get_batch()

    def do_batch(self, b):
        b = super().do_batch(b)
        self._add_batch(b)

    def _add_batch(self, b):
        if not self.batch_items:
            self.batch_items = b
        else:
            for i in range(len(b)):
                if isinstance(b[i], dict):
                    for k in self.batch_items[i].keys():
                        self.batch_items[i][k] = torch.cat([self.batch_items[i][k], b[i][k]])
                else:
                    self.batch_items[i].data = torch.cat([self.batch_items[i], b[i]])

        # update "n" to reflect the additional samples
        overflow_map = b[0][self.overflow_map_key].numpy()
        self.n += np.sum([i - 1 for i in Counter(overflow_map).values()])

    def _get_batch(self):
        chunked_batch = []

        for i in range(len(self.batch_items)):
            if isinstance(self.batch_items[i], dict):
                chunked_d = {}
                for k in self.batch_items[i].keys():
                    chunked_d[k] = self.batch_items[i][k][: self.bs]
                    self.batch_items[i][k] = self.batch_items[i][k][self.bs :]

                chunked_batch.append(chunked_d)
            else:
                chunked_batch.append(self.batch_items[i][: self.bs])
                self.batch_items[i].data = self.batch_items[i][self.bs :]

        return tuplify(chunked_batch)

    def _n_batch_items(self):
        return len(self.batch_items[0][self.overflow_map_key]) if self.batch_items else 0

    def _one_pass(self):
        self.do_batch([self.do_item(0)])
        b = self._get_batch()
        if self.device is not None:
            b = to_device(b, self.device)
        its = self.after_batch(b)
        self._n_inp = 1 if not isinstance(its, (list, tuple)) or len(its) == 1 else len(its) - 1
        self._types = explode_types(its)


# Cell
class HF_TextBlock(TransformBlock):
    """The core `TransformBlock` to prepare your data for training in Blurr with fastai's `DataBlock` API"""

    def __init__(
        self,
        # The abbreviation/name of your Hugging Face transformer architecture (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_arch: str = None,
        # A Hugging Face configuration object (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_config: PretrainedConfig = None,
        # A Hugging Face tokenizer (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_tokenizer: PreTrainedTokenizerBase = None,
        # A Hugging Face model (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_model: PreTrainedModel = None,
        # The before batch transform you want to use to tokenize your raw data on the fly
        # (defaults to an instance of `HF_BeforeBatchTransform` created using the Hugging Face objects defined above)
        before_batch_tfm: HF_BeforeBatchTransform = None,
        # The batch_tfms to apply to the creation of your DataLoaders,
        # (defaults to HF_AfterBatchTransform created using the Hugging Face objects defined above)
        after_batch_tfm: HF_AfterBatchTransform = None,
        # To control the length of the padding/truncation. It can be an integer or None,
        # in which case it will default to the maximum length the model can accept. If the model has no
        # specific maximum input length, truncation/padding to max_length is deactivated.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        max_length: int = None,
        # To control the `padding` applied to your `hf_tokenizer` during tokenization. If None, will default to
        # `False` or `'do_not_pad'.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        padding: Union[bool, str] = True,
        # To control `truncation` applied to your `hf_tokenizer` during tokenization. If None, will default to
        # `False` or `do_not_truncate`.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        truncation: Union[bool, str] = True,
        # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. Set this to `True`
        # if your inputs are pre-tokenized (not numericalized)
        is_split_into_words: bool = False,
        # The return type your decoded inputs should be cast too (used by methods such as `show_batch`)
        input_return_type: Type = HF_BaseInput,
        # The type of `DataLoader` you want created (defaults to `SortedDL`)
        dl_type: DataLoader = None,
        # Any keyword arguments you want applied to your before batch tfm
        before_batch_kwargs: dict = {},
        # Any keyword arguments you want applied to your after batch tfm (or referred to in fastai as `batch_tfms`)
        after_batch_kwargs: dict = {},
        # Any keyword arguments you want your Hugging Face tokenizer to use during tokenization
        tok_kwargs: dict = {},
        # Any keyword arguments you want to have applied with generating text
        text_gen_kwargs: dict = {},
        # Any keyword arguments you want applied to `HF_TextBlock`
        **kwargs
    ):
        if (not all([hf_arch, hf_config, hf_tokenizer, hf_model])) and before_batch_tfm is None:
            raise ValueError(
                """You must supply the Hugging Face architecture, config, tokenizer, and model
                - or - an instances of HF_BeforeBatchTransform"""
            )

        if before_batch_tfm is None:
            # if allowing overflow, if we have to ensure mixed batch items are the same shape
            if ("return_overflowing_tokens" in tok_kwargs):
                padding = 'max_length'

            before_batch_tfm = HF_BeforeBatchTransform(
                hf_arch,
                hf_config,
                hf_tokenizer,
                hf_model,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                is_split_into_words=is_split_into_words,
                tok_kwargs=tok_kwargs.copy(),
                **before_batch_kwargs.copy()
            )

        if after_batch_tfm is None:
            after_batch_tfm = HF_AfterBatchTransform(
                hf_tokenizer=before_batch_tfm.hf_tokenizer, input_return_type=input_return_type, **after_batch_kwargs.copy()
            )

        if dl_type is None:
            dl_sort_func = partial(
                blurr_sort_func,
                hf_tokenizer=before_batch_tfm.hf_tokenizer,
                is_split_into_words=before_batch_tfm.is_split_into_words,
                tok_kwargs=before_batch_tfm.tok_kwargs.copy(),
            )

            # `OverflowDL` is a `DataLoader` that knows how to serve batches of items that are created on the fly as a result
            # of asking the tokenizer to return an input in chunks if the lenght > max_length
            if ("return_overflowing_tokens" in before_batch_tfm.tok_kwargs):
              dl_type = partial(OverflowDL, sort_func=dl_sort_func)
            else:
                partial(SortedDL, sort_func=dl_sort_func)


        # set the TransformBlock's Hugging Face face objects
        self.hf_arch = before_batch_tfm.hf_arch
        self.hf_config = before_batch_tfm.hf_config
        self.hf_tokenizer = before_batch_tfm.hf_tokenizer
        self.hf_model = before_batch_tfm.hf_model

        return super().__init__(dl_type=dl_type, dls_kwargs={"before_batch": before_batch_tfm}, batch_tfms=after_batch_tfm)



# Cell
@dataclass
class BlurrBatchCreator:
    """A class that can be assigned to a `TfmdDL.create_batch` method; used to in Blurr's low-level API
    to create batches that can be used in the Blurr library
    """

    def __init__(
        self,
        # Your Hugging Face tokenizer
        hf_tokenizer: PreTrainedTokenizerBase,
        # Defaults to use Hugging Face's DataCollatorWithPadding(tokenizer=hf_tokenizer)
        data_collator: Type = None,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.data_collator = data_collator if (data_collator) else DataCollatorWithPadding(tokenizer=hf_tokenizer)

    def __call__(self, features):  # A mini-batch (list of examples to run through your model)
        """This method will collate your data using `self.data_collator` and add a target element to the
        returned tuples if `labels` are defined as is the case when most Hugging Face datasets
        """
        batch = self.data_collator(features)
        if isinstance(features[0], dict):
            return dict(batch), batch["labels"] if ("labels" in features[0]) else dict(batch)

        return batch


# Cell
class BlurrBatchTransform(HF_AfterBatchTransform):
    """A class used to cast your inputs into something understandable in fastai `show` methods"""

    def __init__(
        self,
        # The abbreviation/name of your Hugging Face transformer architecture (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_arch: str = None,
        # A Hugging Face configuration object (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_config: PretrainedConfig = None,
        # A Hugging Face tokenizer (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_tokenizer: PreTrainedTokenizerBase = None,
        # A Hugging Face model (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_model: PreTrainedModel = None,
        # The `is_split_into_words` argument applied to your `hf_tokenizer` during tokenization. Set this to `True`
        # if your inputs are pre-tokenized (not numericalized)
        is_split_into_words: bool = False,
        # The token ID to ignore when calculating loss/metrics
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
        # Any other keyword arguments you want included when using your `hf_tokenizer` to tokenize your inputs
        tok_kwargs: dict = {},
        # Any text generation keyword arguments
        text_gen_kwargs: dict = {},
        # The return type your decoded inputs should be cast too (used by methods such as `show_batch`)
        input_return_type: Type = HF_BaseInput,
        # Any other keyword arguments you need to pass to `HF_AfterBatchTransform`
        **kwargs
    ):
        super().__init__(hf_tokenizer=hf_tokenizer, input_return_type=input_return_type)

        store_attr(self=self, names="hf_arch, hf_config, hf_model, tok_kwargs, text_gen_kwargs")
        store_attr(self=self, names="is_split_into_words, ignore_token_id, kwargs")



# Cell
@delegates()
class BlurrDataLoader(TfmdDL):
    """A class that makes creating a fast.ai `DataLoader` that works with Blurr"""

    def __init__(
        self,
        # A standard PyTorch Dataset
        dataset: Union[torch.utils.data.dataset.Dataset, Datasets],
        # The abbreviation/name of your Hugging Face transformer architecture (not required if passing in an
        # instance of `HF_BeforeBatchTransform` to `before_batch_tfm`)
        hf_arch: str,
        # A Hugging Face configuration object (not required if passing in an instance of `HF_BeforeBatchTransform`
        # to `before_batch_tfm`)
        hf_config: PretrainedConfig,
        # A Hugging Face tokenizer (not required if passing in an instance of `HF_BeforeBatchTransform` to
        # `before_batch_tfm`)
        hf_tokenizer: PreTrainedTokenizerBase,
        # A Hugging Face model (not required if passing in an instance of `HF_BeforeBatchTransform` to
        # `before_batch_tfm`)
        hf_model: PreTrainedModel,
        # An instance of `BlurrBatchCreator` or equivalent
        batch_creator: BlurrBatchCreator = None,
        # The batch_tfm used to decode Blurr batches (default: HF_AfterBatchTransform)
        batch_tfm: BlurrBatchTransform = None,
        # (optional) A preprocessing function that will be applied to your dataset
        preproccesing_func: Callable[
            [Union[torch.utils.data.dataset.Dataset, Datasets], PreTrainedTokenizerBase, PreTrainedModel],
            Union[torch.utils.data.dataset.Dataset, Datasets],
        ] = None,
        # (optional) list of corresponding labels names for classes; if included then methods like `show_batch` will
        # show the name corresponding to the label index vs. just the integer index.
        label_names: Optional[list] = None,
        # Keyword arguments to be applied to your `batch_tfm`
        batch_tfm_kwargs: dict = {},
        # Keyword arguments to be applied to `BlurrDataLoader`
        **kwargs
    ):
        if preproccesing_func:
            dataset = preproccesing_func(dataset, hf_tokenizer, hf_model)

        if "create_batch" in kwargs:
            kwargs.pop("create_batch")
        if not batch_creator:
            batch_creator = BlurrBatchCreator(hf_tokenizer=hf_tokenizer)

        if "after_batch" in kwargs:
            kwargs.pop("after_batch")
        if not batch_tfm:
            batch_tfm = BlurrBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, **batch_tfm_kwargs.copy())

        super().__init__(dataset=dataset, create_batch=batch_creator, after_batch=batch_tfm, **kwargs)
        store_attr(self=self, names="hf_arch, hf_config, hf_tokenizer, hf_model, label_names")

    def new(
        self,
        # A standard PyTorch and fastai dataset
        dataset: Union[torch.utils.data.dataset.Dataset, Datasets] = None,
        # The class you want to create an instance of (will be "self" if None)
        cls: Type = None,
        #  Any additional keyword arguments you want to pass to the __init__ method of `cls`
        **kwargs
    ):
        """We have to override the new method in order to add back the Hugging Face objects in this factory
        method (called for example in places like `show_results`). With the exception of the additions to the kwargs
        dictionary, the code below is pulled from the `DataLoaders.new` method as is.
        """
        if dataset is None:
            dataset = self.dataset
        if cls is None:
            cls = type(self)

        cur_kwargs = dict(
            dataset=dataset,
            num_workers=self.fake_l.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            bs=self.bs,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            indexed=self.indexed,
            device=self.device,
        )

        for n in self._methods:
            o = getattr(self, n)
            if not isinstance(o, MethodType):
                cur_kwargs[n] = o

        # we need to add these arguments back in (these, after_batch, and create_batch will go in as kwargs)
        kwargs["hf_arch"] = self.hf_arch
        kwargs["hf_config"] = self.hf_config
        kwargs["hf_tokenizer"] = self.hf_tokenizer
        kwargs["hf_model"] = self.hf_model

        return cls(**merge(cur_kwargs, kwargs))


# Cell
def get_blurr_tfm(
    # A list of transforms (e.g., dls.after_batch, dls.before_batch, etc...)
    tfms_list: Pipeline,
    # The transform to find
    tfm_class: Transform = HF_BeforeBatchTransform,
):
    """Given a fastai DataLoaders batch transforms, this method can be used to get at a transform
    instance used in your Blurr DataBlock
    """
    return next(filter(lambda el: issubclass(type(el), tfm_class), tfms_list), None)


# Cell
def first_blurr_tfm(
    dls: DataLoaders,  # Your fast.ai `DataLoaders
    before_batch_tfm_class: Transform = HF_BeforeBatchTransform,  # The before_batch transform to look for
    blurr_batch_tfm_class: Transform = BlurrBatchTransform,  # The after_batch (or batch_tfm) to look for
):
    """This convenience method will find the first Blurr transform required for methods such as
    `show_batch` and `show_results`. The returned transform should have everything you need to properly
    decode and 'show' your Hugging Face inputs/targets
    """
    # try our befor_batch tfms (this will be used if you're using the mid-level DataBlock API)
    tfm = get_blurr_tfm(dls.before_batch, tfm_class=before_batch_tfm_class)
    if tfm:
        return tfm

    # try our after_batch tfms (this will be used if you're using the low-level Blurr data API)
    return get_blurr_tfm(dls.after_batch, tfm_class=blurr_batch_tfm_class)


# Cell
@typedispatch
def show_batch(
    # This typedispatched `show_batch` will be called for `HF_BaseInput` typed inputs
    x: HF_BaseInput,
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

    trg_labels = None
    if hasattr(dataloaders, "label_names"):
        trg_labels = dataloaders.label_names

    res = L()
    n_inp = dataloaders.n_inp

    for idx, (input_ids, label, sample) in enumerate(zip(x, y, samples)):
        if idx >= max_n:
            break

        rets = [hf_tokenizer.decode(input_ids, skip_special_tokens=True)[:trunc_at]]
        for item in sample[n_inp:]:
            if not torch.is_tensor(item):
                trg = item
            elif is_listy(item.tolist()):
                trg = [trg_labels[idx] for idx, val in enumerate(label.numpy().tolist()) if (val == 1)] if (trg_labels) else label.item()
            else:
                trg = trg_labels[label.item()] if (trg_labels) else label.item()

            rets.append(trg)
        res.append(tuplify(rets))

    cols = ["text"] + ["target" if (i == 0) else f"target_{i}" for i in range(len(res[0]) - n_inp)]
    display_df(pd.DataFrame(res, columns=cols)[:max_n])
    return ctxs


# Cell
def preproc_hf_dataset(
    # A standard PyTorch Dataset or fast.ai Datasets
    dataset: Union[torch.utils.data.dataset.Dataset, Datasets],
    # A Hugging Face tokenizer
    hf_tokenizer: PreTrainedTokenizerBase,
    # A Hugging Face model
    hf_model: PreTrainedModel,
):
    """This method can be used to preprocess most Hugging Face Datasets for use in Blurr and other training
    libraries
    """
    if ("label") in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")

    hf_model_fwd_args = list(inspect.signature(hf_model.forward).parameters.keys())
    bad_cols = set(dataset.column_names).difference(hf_model_fwd_args)
    dataset = dataset.remove_columns(bad_cols)

    dataset.set_format("torch")
    return dataset
