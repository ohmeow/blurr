# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_data-seq2seq-summarization.ipynb (unless otherwise specified).

__all__ = []

# Cell
from fastai.data.block import DataBlock
from transformers import AutoModelForSeq2SeqLM, logging

from ...utils import BLURR
from .core import Seq2SeqTextBlock, Seq2SeqBatchTokenizeTransform

logging.set_verbosity_error()