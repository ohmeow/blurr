"""Tests for blurr.multimodal.classification — tiny CPU models."""

from datasets import Dataset, DatasetDict

from blurr.multimodal.classification import MultimodalClassificationLearner
from .conftest import TINY_BLIP_VQA, TINY_CLIP


class TestMultimodalClassificationLearner:
    def test_from_df_classification(self, multimodal_df):
        learn = MultimodalClassificationLearner.from_df(
            multimodal_df,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            label_col="label",
            task="classification",
            valid_col="is_valid",
            bs=2,
        )
        assert learn is not None

    def test_classification_training_step(self, multimodal_df):
        learn = MultimodalClassificationLearner.from_df(
            multimodal_df,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            label_col="label",
            task="classification",
            valid_col="is_valid",
            bs=2,
        )
        learn.fit(1, lr=1e-3)

    def test_blurr_predict(self, multimodal_df):
        learn = MultimodalClassificationLearner.from_df(
            multimodal_df,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            label_col="label",
            task="classification",
            valid_col="is_valid",
            bs=2,
        )
        item = (multimodal_df.iloc[0]["image"], multimodal_df.iloc[0]["text"])
        results = learn.blurr_predict(item)
        assert len(results) == 1
        assert "label" in results[0]
        assert "score" in results[0]
        assert "probs" in results[0]

    def test_zero_shot_predict(self, multimodal_df):
        learn = MultimodalClassificationLearner.from_df(
            multimodal_df,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            label_col="label",
            task="classification",
            valid_col="is_valid",
            bs=2,
        )
        results = learn.blurr_zero_shot_predict(
            multimodal_df.iloc[0]["image"],
            ["red square", "blue square"],
        )
        assert len(results) == 1
        assert "label" in results[0]
        assert "score" in results[0]

    def test_contrastive_mode(self, multimodal_df):
        learn = MultimodalClassificationLearner.from_df(
            multimodal_df,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            task="contrastive",
            valid_col="is_valid",
            bs=2,
        )
        assert learn is not None
        learn.fit(1, lr=1e-3)

    def test_vqa_mode(self, vqa_df):
        learn = MultimodalClassificationLearner.from_df(
            vqa_df,
            TINY_BLIP_VQA,
            image_col="image",
            text_col="text",
            label_col="label",
            task="vqa",
            valid_col="is_valid",
            bs=2,
        )
        assert learn is not None

    def test_from_hf_datasetdict(self, multimodal_df):
        train_df = multimodal_df[multimodal_df["is_valid"] == False].reset_index(drop=True)
        valid_df = multimodal_df[multimodal_df["is_valid"] == True].reset_index(drop=True)

        ds = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(valid_df),
            }
        )

        learn = MultimodalClassificationLearner.from_hf(
            ds,
            TINY_CLIP,
            image_col="image",
            text_col="text",
            label_col="label",
            task="classification",
            bs=2,
        )
        assert learn is not None
