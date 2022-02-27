# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"Singleton": "00_utils.ipynb",
         "str_to_type": "00_utils.ipynb",
         "print_versions": "00_utils.ipynb",
         "set_seed": "00_utils.ipynb",
         "BlurrUtil": "00_utils.ipynb",
         "BLURR": "00_utils.ipynb",
         "HF_TASKS": "00_utils.ipynb",
         "HF_ARCHITECTURES": "00_utils.ipynb",
         "Preprocessor": "01_data-core.ipynb",
         "ClassificationPreprocessor": "01_data-core.ipynb",
         "TextInput": "01_data-core.ipynb",
         "BatchTokenizeTransform": "01_data-core.ipynb",
         "BatchDecodeTransform": "01_data-core.ipynb",
         "blurr_sort_func": "01_data-core.ipynb",
         "TextBlock": "01_data-core.ipynb",
         "BlurrBatchCreator": "01_data-core.ipynb",
         "BlurrBatchDecodeTransform": "01_data-core.ipynb",
         "BlurrDataLoader": "01_data-core.ipynb",
         "get_blurr_tfm": "01_data-core.ipynb",
         "first_blurr_tfm": "01_data-core.ipynb",
         "preproc_hf_dataset": "01_data-core.ipynb",
         "blurr_splitter": "01_modeling-core.ipynb",
         "BaseModelWrapper": "01_modeling-core.ipynb",
         "PreCalculatedLoss": "01_modeling-core.ipynb",
         "PreCalculatedCrossEntropyLoss": "01_modeling-core.ipynb",
         "PreCalculatedBCELoss": "01_modeling-core.ipynb",
         "PreCalculatedMSELossFlat": "01_modeling-core.ipynb",
         "BaseModelCallback": "01_modeling-core.ipynb",
         "Learner.blurr_predict": "01_modeling-core.ipynb",
         "Learner.blurr_generate": "01_modeling-core.ipynb",
         "Blearner": "01_modeling-core.ipynb",
         "BlearnerForSequenceClassification": "01_modeling-core.ipynb",
         "LMType": "02_data-language-modeling.ipynb",
         "LMStrategy": "02_data-language-modeling.ipynb",
         "HF_LMBeforeBatchTransform": "02_data-language-modeling.ipynb",
         "HF_CausalLMInput": "02_data-language-modeling.ipynb",
         "CausalLMStrategy": "02_data-language-modeling.ipynb",
         "HF_MLMInput": "02_data-language-modeling.ipynb",
         "BertMLMStrategy": "02_data-language-modeling.ipynb",
         "LM_MetricsCallback": "02_modeling-language-modeling.ipynb",
         "Learner.blurr_fill_mask": "02_modeling-language-modeling.ipynb",
         "BlearnerForLM": "02_modeling-language-modeling.ipynb",
         "TokenClassPreprocessor": "03_data-token-classification.ipynb",
         "BaseLabelingStrategy": "03_data-token-classification.ipynb",
         "OnlyFirstTokenLabelingStrategy": "03_data-token-classification.ipynb",
         "SameLabelLabelingStrategy": "03_data-token-classification.ipynb",
         "BILabelingStrategy": "03_data-token-classification.ipynb",
         "get_token_labels_from_input_ids": "03_data-token-classification.ipynb",
         "get_word_labels_from_token_labels": "03_data-token-classification.ipynb",
         "TokenTensorCategory": "03_data-token-classification.ipynb",
         "TokenCategorize": "03_data-token-classification.ipynb",
         "TokenCategoryBlock": "03_data-token-classification.ipynb",
         "TokenClassTextInput": "03_data-token-classification.ipynb",
         "TokenClassBatchTokenizeTransform": "03_data-token-classification.ipynb",
         "calculate_token_class_metrics": "03_modeling-token-classification.ipynb",
         "TokenClassMetricsCallback": "03_modeling-token-classification.ipynb",
         "TokenAggregationStrategies": "03_modeling-token-classification.ipynb",
         "Learner.blurr_predict_tokens": "03_modeling-token-classification.ipynb",
         "BlearnerForTokenClassification": "03_modeling-token-classification.ipynb",
         "QAPreprocessor": "04_data-question-answering.ipynb",
         "QATextInput": "04_data-question-answering.ipynb",
         "QABatchTokenizeTransform": "04_data-question-answering.ipynb",
         "squad_metric": "04_modeling-question-answering.ipynb",
         "QAModelCallback": "04_modeling-question-answering.ipynb",
         "QAMetricsCallback": "04_modeling-question-answering.ipynb",
         "compute_qa_metrics": "04_modeling-question-answering.ipynb",
         "PreCalculatedQALoss": "04_modeling-question-answering.ipynb",
         "MultiTargetLoss": "04_modeling-question-answering.ipynb",
         "Learner.blurr_predict_answers": "04_modeling-question-answering.ipynb",
         "BlearnerForQuestionAnswering": "04_modeling-question-answering.ipynb",
         "Seq2SeqPreprocessor": "10_data-seq2seq-core.ipynb",
         "Seq2SeqTextInput": "10_data-seq2seq-core.ipynb",
         "Seq2SeqBatchTokenizeTransform": "10_data-seq2seq-core.ipynb",
         "Seq2SeqBatchDecodeTransform": "10_data-seq2seq-core.ipynb",
         "default_text_gen_kwargs": "10_data-seq2seq-core.ipynb",
         "Seq2SeqTextBlock": "10_data-seq2seq-core.ipynb",
         "blurr_seq2seq_splitter": "10_modeling-seq2seq-core.ipynb",
         "Seq2SeqMetricsCallback": "10_modeling-seq2seq-core.ipynb",
         "SummarizationPreprocessor": "11_data-seq2seq-summarization.ipynb",
         "Learner.blurr_summarize": "11_modeling-seq2seq-summarization.ipynb",
         "BlearnerForSummarization": "11_modeling-seq2seq-summarization.ipynb",
         "TranslationPreprocessor": "12_data-seq2seq-translation.ipynb",
         "BlearnerForTranslation": "12_modeling-seq2seq-translation.ipynb"}

modules = ["utils.py",
           "data/core.py",
           "modeling/core.py",
           "data/language_modeling.py",
           "modeling/language_modeling.py",
           "data/token_classification.py",
           "modeling/token_classification.py",
           "data/question_answering.py",
           "modeling/question_answering.py",
           "data/seq2seq/core.py",
           "modeling/seq2seq/core.py",
           "data/seq2seq/summarization.py",
           "modeling/seq2seq/summarization.py",
           "data/seq2seq/translation.py",
           "modeling/seq2seq/translation.py",
           "examples/blurr_high_level_api.py",
           "examples/glue.py",
           "examples/glue_low_level_api.py",
           "examples/multilabel_classification.py",
           "examples/causal_lm_gpt2.py"]

doc_url = "https://ohmeow.github.io/blurr/"

git_url = "https://github.com/ohmeow/blurr/tree/master/"

def custom_doc_links(name):
    return None
