"""Tests for blurr.vision.classification — CPU-friendly with tiny models."""

import pytest
import torch
from datasets import Dataset, DatasetDict

from blurr.vision._core import VisionBatchTransform, load_image
from blurr.vision.classification import VisionClassificationLearner
from .conftest import TINY_VIT


class TestVisionBatchTransform:
    def test_path_loading_and_tokenization(self, vision_df):
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(TINY_VIT)
        tfm = VisionBatchTransform(processor, labels=["blue", "red"])

        samples = [
            (vision_df.iloc[0]["image"], torch.tensor(0)),
            (vision_df.iloc[1]["image"], torch.tensor(1)),
        ]
        result = tfm.encodes(samples)

        assert len(result) == 2
        inp_dict, target = result[0]
        assert isinstance(inp_dict, dict)
        assert "pixel_values" in inp_dict
        assert "labels" in inp_dict
        assert target is not None

    def test_multilabel_labels_are_float(self, vision_multilabel_df):
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(TINY_VIT)
        tfm = VisionBatchTransform(processor, is_multilabel=True)

        row = vision_multilabel_df.iloc[0]
        samples = [
            (row["image"], torch.tensor([row["red"], row["green"], row["blue"]], dtype=torch.float32)),
        ]
        result = tfm.encodes(samples)
        inp_dict, _ = result[0]
        assert inp_dict["labels"].dtype == torch.float32

    def test_load_image_accepts_pil_and_path(self, vision_df):
        from PIL import Image

        path = vision_df.iloc[0]["image"]
        img_from_path = load_image(path)
        assert isinstance(img_from_path, Image.Image)

        img_from_pil = load_image(img_from_path)
        assert isinstance(img_from_pil, Image.Image)


class TestVisionClassificationLearner:
    def test_from_df_creates_learner(self, vision_df):
        learn = VisionClassificationLearner.from_df(
            vision_df,
            TINY_VIT,
            image_col="image",
            label_col="label",
            valid_col="is_valid",
            bs=2,
        )
        assert learn is not None
        assert len(learn.dls.train_ds) > 0
        assert len(learn.dls.valid_ds) > 0

    def test_training_step(self, vision_df):
        learn = VisionClassificationLearner.from_df(
            vision_df,
            TINY_VIT,
            image_col="image",
            label_col="label",
            valid_col="is_valid",
            bs=2,
        )
        learn.fit(1, lr=1e-3)

    def test_blurr_predict_shape(self, vision_df):
        learn = VisionClassificationLearner.from_df(
            vision_df,
            TINY_VIT,
            image_col="image",
            label_col="label",
            valid_col="is_valid",
            bs=2,
        )
        results = learn.blurr_predict(vision_df.iloc[0]["image"])
        assert len(results) == 1
        assert "label" in results[0]
        assert "score" in results[0]
        assert "probs" in results[0]

    def test_multilabel_from_df(self, vision_multilabel_df):
        learn = VisionClassificationLearner.from_df(
            vision_multilabel_df,
            TINY_VIT,
            image_col="image",
            label_col=["red", "green", "blue"],
            is_multilabel=True,
            valid_col="is_valid",
            bs=2,
        )
        results = learn.blurr_predict(vision_multilabel_df.iloc[0]["image"])
        assert len(results) == 1
        assert "labels" in results[0]
        assert "scores" in results[0]

    def test_from_hf_datasetdict(self, vision_df):
        train_df = vision_df[vision_df["is_valid"] == False].reset_index(drop=True)
        valid_df = vision_df[vision_df["is_valid"] == True].reset_index(drop=True)

        ds = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(valid_df),
            }
        )

        learn = VisionClassificationLearner.from_hf(
            ds,
            TINY_VIT,
            image_col="image",
            label_col="label",
            bs=2,
        )
        assert learn is not None

    def test_from_data_prebuilt_dls(self, vision_df):
        learn1 = VisionClassificationLearner.from_df(
            vision_df,
            TINY_VIT,
            image_col="image",
            label_col="label",
            valid_col="is_valid",
            bs=2,
        )
        learn2 = VisionClassificationLearner.from_data(learn1.dls, TINY_VIT)
        assert learn2 is not None
