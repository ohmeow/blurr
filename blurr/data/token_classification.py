# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/12_data-token-classification.ipynb.

# %% ../../nbs/12_data-token-classification.ipynb 3
from __future__ import annotations

import gc, importlib, sys, traceback

from accelerate.logging import get_logger
from dataclasses import dataclass
from dotenv import load_dotenv
from fastai.imports import *
from fastai.losses import CrossEntropyLossFlat
from fastai.data.block import TransformBlock, Category, CategoryMap
from fastai.torch_core import *
from fastai.torch_imports import *
from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel, AutoModelForTokenClassification
from transformers import logging as hf_logging
from transformers.data.data_collator import DataCollatorWithPadding


from .core import first_blurr_tfm, TextInput, TextCollatorWithPadding, BatchTokenizeTransform
from ..utils import get_hf_objects

# %% auto 0
__all__ = ['logger', 'BaseLabelingStrategy', 'OnlyFirstTokenLabelingStrategy', 'SameLabelLabelingStrategy', 'BILabelingStrategy',
           'get_task_hf_objects', 'tokenclass_tokenize_func', 'get_token_labels_from_input_ids',
           'get_word_labels_from_token_labels', 'TokenClassTextCollatorWithPadding', 'TokenClassTextInput',
           'show_batch', 'TokenTensorCategory', 'TokenCategorize', 'TokenCategoryBlock',
           'TokenClassBatchTokenizeTransform']

# %% ../../nbs/12_data-token-classification.ipynb 5
# silence all the HF warnings and load environment variables
warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()
logger = get_logger(__name__)

load_dotenv()

# %% ../../nbs/12_data-token-classification.ipynb 18
class BaseLabelingStrategy:
    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        label_names: Optional[List[str]],
        non_entity_label: str = "O",
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
    ) -> None:
        self.hf_tokenizer = hf_tokenizer
        self.ignore_token_id = ignore_token_id
        self.label_names = label_names
        self.non_entity_label = non_entity_label

    def align_labels_with_tokens(self, word_ids, word_labels):
        raise NotImplementedError()

# %% ../../nbs/12_data-token-classification.ipynb 20
class OnlyFirstTokenLabelingStrategy(BaseLabelingStrategy):
    """
    Only the first token of word is associated with the label (all other subtokens with the `ignore_index_id`). Works where labels
    are Ids or strings (in the later case we'll use the `label_names` to look up it's Id)
    """

    def align_labels_with_tokens(self, word_ids, word_labels):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # start of a new word
                current_word = word_id
                label = self.ignore_token_id if word_id is None else word_labels[word_id]
                new_labels.append(label if isinstance(label, int) else self.label_names.index(label))
            else:
                # special token or another subtoken of current word
                new_labels.append(self.ignore_token_id)

        return new_labels


class SameLabelLabelingStrategy(BaseLabelingStrategy):
    """
    Every token associated with a given word is associated with the word's label. Works where labels
    are Ids or strings (in the later case we'll use the `label_names` to look up it's Id)
    """

    def align_labels_with_tokens(self, word_ids, word_labels):
        new_labels = []
        for word_id in word_ids:
            if word_id == None:
                new_labels.append(self.ignore_token_id)
            else:
                label = word_labels[word_id]
                new_labels.append(label if isinstance(label, int) else self.label_names.index(label))

        return new_labels


class BILabelingStrategy(BaseLabelingStrategy):
    """
    If using B/I labels, the first token assoicated to a given word gets the "B" label while all other tokens related
    to that same word get "I" labels.  If "I" labels don't exist, this strategy behaves like the `OnlyFirstTokenLabelingStrategy`.
    Works where labels are Ids or strings (in the later case we'll use the `label_names` to look up it's Id)
    """

    def align_labels_with_tokens(self, word_ids, word_labels):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # start of a new word
                current_word = word_id
                label = self.ignore_token_id if word_id is None else word_labels[word_id]
                new_labels.append(label if isinstance(label, int) else self.label_names.index(label))
            elif word_id is None:
                # special token
                new_labels.append(self.ignore_token_id)
            else:
                # we're in the same word
                label = word_labels[word_id]
                label_name = self.label_names[label] if isinstance(label, int) else label

                # append the I-{ENTITY} if it exists in `labels`, else default to the `same_label` strategy
                iLabel = f"I-{label_name[2:]}"
                new_labels.append(
                    self.label_names.index(iLabel) if iLabel in self.label_names else self.label_names.index(self.non_entity_label)
                )

        return new_labels

# %% ../../nbs/12_data-token-classification.ipynb 22
def get_task_hf_objects(
    pretrained_model_name: str,
    label_names: list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
    verbose: bool = False,
):
    model_cls = AutoModelForTokenClassification
    n_labels = len(label_names)

    hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(
        pretrained_model_name, model_cls=model_cls, config_kwargs={"num_labels": n_labels}
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

# %% ../../nbs/12_data-token-classification.ipynb 25
# tokenize the dataset
def tokenclass_tokenize_func(
    examples,
    hf_tokenizer: PreTrainedTokenizerBase,
    labeling_strategy: BaseLabelingStrategy,
    words_attr: str = "words",
    word_labels_attr: str = "labels",
    max_length: int = None,
    padding: bool | str = True,
    truncation: bool | str = True,
    tok_kwargs: dict = {},
):
    inputs = hf_tokenizer(
        examples[words_attr], max_length=max_length, padding=padding, truncation=truncation, is_split_into_words=True, **tok_kwargs
    )

    all_labels = examples[word_labels_attr]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = inputs.word_ids(i)
        new_labels.append(labeling_strategy.align_labels_with_tokens(word_ids, labels))

    inputs["label"] = new_labels
    return inputs

# %% ../../nbs/12_data-token-classification.ipynb 30
def get_token_labels_from_input_ids(
    # A Hugging Face tokenizer
    hf_tokenizer: PreTrainedTokenizerBase,
    # List of input_ids for the tokens in a single piece of processed text
    input_ids: List[int],
    # List of label indexs for each token
    token_label_ids: List[int],
    # List of label names from witch the `label` indicies can be used to find the name of the label
    vocab: List[str],
    # The token ID that should be ignored when calculating the loss
    ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
    # The token used to identifiy ignored tokens (default: [xIGNx])
    ignore_token: str = "[xIGNx]",
) -> List[Tuple[str, str]]:
    """
    Given a list of input IDs, the label ID associated to each, and the labels vocab, this method will return a list of tuples whereby
    each tuple defines the "token" and its label name. For example:
    [('ĠWay', B-PER), ('de', B-PER), ('ĠGill', I-PER), ('iam', I-PER), ('Ġloves'), ('ĠHug', B-ORG), ('ging', B-ORG), ('ĠFace', I-ORG)]
    """
    # convert ids to tokens
    toks = hf_tokenizer.convert_ids_to_tokens(input_ids)
    # align "tokens" with labels
    tok_labels = [
        (tok, ignore_token if label_id == ignore_token_id else vocab[label_id])
        for tok_id, tok, label_id in zip(input_ids, toks, token_label_ids)
        if tok_id not in hf_tokenizer.all_special_ids
    ]
    return tok_labels

# %% ../../nbs/12_data-token-classification.ipynb 34
def get_word_labels_from_token_labels(
    hf_arch: str,
    # A Hugging Face tokenizer
    hf_tokenizer: PreTrainedTokenizerBase,
    # A list of tuples, where each represents a token and its label (e.g., [('ĠHug', B-ORG), ('ging', B-ORG), ('ĠFace', I-ORG), ...])
    tok_labels,
) -> List[Tuple[str, str]]:
    """
    Given a list of tuples where each tuple defines a token and its label, return a list of tuples whereby each tuple defines the
    "word" and its label. Method assumes that model inputs are a list of words, and in conjunction with the `align_labels_with_tokens` method,
    allows the user to reconstruct the orginal raw inputs and labels.
    """
    # recreate raw words list (we assume for token classification that the input is a list of words)
    words = hf_tokenizer.convert_tokens_to_string([tok_label[0] for tok_label in tok_labels]).split()

    if hf_arch == "canine":
        word_list = [f"{word} " for word in words]
    else:
        word_list = [word for word in words]

    # align "words" with labels
    word_labels, idx = [], 0
    for word in word_list:
        word_labels.append((word, tok_labels[idx][1]))
        idx += len(hf_tokenizer.tokenize(word))

    return word_labels

# %% ../../nbs/12_data-token-classification.ipynb 39
@dataclass
class TokenClassTextCollatorWithPadding(TextCollatorWithPadding):
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
        # The token ID that should be ignored when calculating the loss
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
        # Defaults to use Hugging Face's DataCollatorWithPadding(tokenizer=hf_tokenizer)
        data_collator_cls: type = DataCollatorWithPadding,
        # kwyargs specific for the instantiation of the `data_collator`
        data_collator_kwargs: dict = {},
    ):
        self.ignore_token_id = ignore_token_id

        super().__init__(
            hf_tokenizer=hf_tokenizer,
            hf_arch=hf_arch,
            hf_config=hf_config,
            hf_model=hf_model,
            n_inp=n_inp,
            data_collator_cls=data_collator_cls,
            data_collator_kwargs=data_collator_kwargs,
        )

    # used to give the labels/targets the right shape
    def _proc_targets(self, inputs_d, targs):
        # the code below comes pretty much straight from the `DataCollatorForTokenClassification` class
        max_seq_length = np.max([len(input_ids) for input_ids in inputs_d["input_ids"]])
        padding_side = self.hf_tokenizer.padding_side

        if padding_side == "right":
            targs = [
                (list(trg.numpy()) if torch.is_tensor(trg) else trg) + [self.ignore_token_id] * (max_seq_length - len(trg)) for trg in targs
            ]
        else:
            targs = [
                [self.ignore_token_id] * (max_seq_length - len(trg)) + (list(trg.numpy()) if torch.is_tensor(trg) else trg) for trg in targs
            ]

        if is_listy(targs[0]):
            targs = torch.stack([tensor(lbls) for lbls in targs])
        elif isinstance(targs[0], torch.Tensor) and len(targs[0].size()) > 0:
            targs = torch.stack(targs)
        else:
            targs = torch.tensor(targs)

        return targs

# %% ../../nbs/12_data-token-classification.ipynb 63
class TokenClassTextInput(TextInput):
    pass

# %% ../../nbs/12_data-token-classification.ipynb 66
@typedispatch
def show_batch(
    # This typedispatched `show_batch` will be called for `TokenClassTextInput` typed inputs
    x: TokenClassTextInput,
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
    hf_arch, hf_tokenizer = tfm.hf_arch, tfm.hf_tokenizer

    # if we've included our labels list, we'll use it to look up the value of our target(s)
    trg_labels = tfm.kwargs["label_names"] if ("label_names" in tfm.kwargs) else None
    if trg_labels is None and dataloaders.vocab is not None:
        trg_labels = dataloaders.vocab

    res = L()
    n_inp = dataloaders.n_inp

    n_samples = min(max_n, dataloaders.bs)
    for idx in range(n_samples):
        input_ids = x[idx]
        trgs = y[idx]
        sample = samples[idx] if samples is not None else None

        # align "tokens" with labels
        tok_labels = get_token_labels_from_input_ids(hf_tokenizer, input_ids, trgs, trg_labels)
        # align "words" with labels
        word_labels = get_word_labels_from_token_labels(hf_arch, hf_tokenizer, tok_labels)
        # stringify list of (word,label) for example
        res.append([f"{[ word_targ for idx, word_targ in enumerate(word_labels) if (trunc_at is None or idx < trunc_at) ]}"])

    display_df(pd.DataFrame(res, columns=["word / target label"])[:max_n])
    return ctxs

# %% ../../nbs/12_data-token-classification.ipynb 82
class TokenTensorCategory(TensorBase):
    pass

# %% ../../nbs/12_data-token-classification.ipynb 84
class TokenCategorize(Transform):
    """Reversible transform of a list of category string to `vocab` id"""

    def __init__(
        self,
        # The unique list of entities (e.g., B-LOC) (default: CategoryMap(vocab))
        vocab: List[str] = None,
        # The token used to identifiy ignored tokens (default: xIGNx)
        ignore_token: str = "[xIGNx]",
        # The token ID that should be ignored when calculating the loss (default: CrossEntropyLossFlat().ignore_index)
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
    ):
        self.vocab = None if vocab is None else CategoryMap(vocab, sort=False)
        self.ignore_token, self.ignore_token_id = ignore_token, ignore_token_id

        self.loss_func, self.order = CrossEntropyLossFlat(ignore_index=self.ignore_token_id), 1

    def setups(self, dsets):
        if self.vocab is None and dsets is not None:
            self.vocab = CategoryMap(dsets)
        self.c = len(self.vocab)

    def encodes(self, labels):
        # if `val` is the label name (e.g., B-PER, I-PER, etc...), lookup the corresponding index in the vocab using
        # `self.vocab.o2i`
        ids = [val if (isinstance(val, int)) else self.vocab.o2i[val] for val in labels]
        return TokenTensorCategory(ids)

    def decodes(self, encoded_labels):
        return Category([(self.vocab[lbl_id]) for lbl_id in encoded_labels if lbl_id != self.ignore_token_id])

# %% ../../nbs/12_data-token-classification.ipynb 87
def TokenCategoryBlock(
    # The unique list of entities (e.g., B-LOC) (default: CategoryMap(vocab))
    vocab: Optional[List[str]] = None,
    # The token used to identifiy ignored tokens (default: xIGNx)
    ignore_token: str = "[xIGNx]",
    # The token ID that should be ignored when calculating the loss (default: CrossEntropyLossFlat().ignore_index)
    ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
):
    """`TransformBlock` for per-token categorical targets"""
    return TransformBlock(type_tfms=TokenCategorize(vocab=vocab, ignore_token=ignore_token, ignore_token_id=ignore_token_id))

# %% ../../nbs/12_data-token-classification.ipynb 91
class TokenClassBatchTokenizeTransform(BatchTokenizeTransform):
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
        # To control whether the "labels" are included in your inputs. If they are, the loss will be calculated in
        # the model's forward function and you can simply use `PreCalculatedLoss` as your `Learner`'s loss function to use it
        include_labels: bool = True,
        # The token ID that should be ignored when calculating the loss
        ignore_token_id: int = CrossEntropyLossFlat().ignore_index,
        # The labeling strategy you want to apply when associating labels with word tokens
        labeling_strategy_cls: BaseLabelingStrategy = BILabelingStrategy,
        # the target label names
        target_label_names: Optional[List[str]] = None,
        # the label for non-entity
        non_entity_label: str = "O",
        # To control the length of the padding/truncation. It can be an integer or None,
        # in which case it will default to the maximum length the model can accept. If the model has no
        # specific maximum input length, truncation/padding to max_length is deactivated.
        # See [Everything you always wanted to know about padding and truncation](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation)
        max_length: Optional[int] = None,
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
        is_split_into_words: bool = True,
        # If using a slow tokenizer, users will need to prove a `slow_word_ids_func` that accepts a
        # tokenizzer, example index, and a batch encoding as arguments and in turn returnes the
        # equavlient of fast tokenizer's `word_ids``
        slow_word_ids_func: Optional[Callable] = None,
        # Any other keyword arguments you want included when using your `hf_tokenizer` to tokenize your inputs
        tok_kwargs: dict = {},
        # Keyword arguments to apply to `TokenClassBatchTokenizeTransform`
        **kwargs,
    ):

        super().__init__(
            hf_arch,
            hf_config,
            hf_tokenizer,
            hf_model,
            include_labels=include_labels,
            ignore_token_id=ignore_token_id,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            is_split_into_words=is_split_into_words,
            tok_kwargs=tok_kwargs,
            **kwargs,
        )

        self.target_label_names = target_label_names
        self.non_entity_label = non_entity_label
        self.slow_word_ids_func = slow_word_ids_func

        self.labeling_strategy = labeling_strategy_cls(
            hf_tokenizer, label_names=self.target_label_names, non_entity_label=self.non_entity_label, ignore_token_id=ignore_token_id
        )

    def encodes(self, samples, return_batch_encoding=False):
        encoded_samples, inputs = super().encodes(samples, return_batch_encoding=True)

        # if there are no targets (e.g., when used for inference)
        if len(encoded_samples[0]) == 1:
            return encoded_samples

        # get the type of our targets (by default will be TokenTensorCategory)
        target_cls = type(encoded_samples[0][1])

        updated_samples = []
        for idx, s in enumerate(encoded_samples):
            # with batch-time tokenization, we have to align each token with the correct label using the `word_ids` in the
            # batch encoding object we get from calling our *fast* tokenizer
            word_ids = inputs.word_ids(idx) if self.hf_tokenizer.is_fast else self.slow_word_ids_func(self.hf_tokenizer, idx, inputs)
            targ_ids = target_cls(self.labeling_strategy.align_labels_with_tokens(word_ids, s[-1].tolist()))

            if self.include_labels and len(targ_ids) > 0:
                s[0]["label"] = targ_ids

            updated_samples.append((s[0], targ_ids))

        if return_batch_encoding:
            return updated_samples, inputs

        return updated_samples
