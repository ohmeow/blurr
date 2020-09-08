# blurr
> A library that integrates huggingface transformers with version 2 of the fastai framework


## Install

You can now pip install blurr via `pip install ohmeow-blurr`

Or, even better as this library is under *very* active development, create an editable install like this:
```
git clone https://github.com/ohmeow/blurr.git
cd blurr
pip install -e ".[dev]"
```

## How to use

The initial release includes everything you need for sequence classification and question answering tasks.  Support for token classification and summarization are incoming. Please check the documentation for more thorough examples of how to use this package.

The following two packages need to be installed for blurr to work:
1. fastai2 (see http://docs.fast.ai/ for installation instructions)
2. huggingface transformers (see https://huggingface.co/transformers/installation.html for details)

### Imports

```python
import torch
from transformers import *
from fastai.text.all import *

from blurr.data.all import *
from blurr.modeling.all import *
```

### Get your data

```python
path = untar_data(URLs.IMDB_SAMPLE)

model_path = Path('models')
imdb_df = pd.read_csv(path/'texts.csv')
```

### Get your 🤗 objects

```python
task = HF_TASKS_AUTO.SequenceClassification

pretrained_model_name = "bert-base-uncased"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR_MODEL_HELPER.get_hf_objects(pretrained_model_name,  task=task)
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


### Build your Data 🧱 and your DataLoaders

```python
# single input
blocks = (HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer), CategoryBlock)

dblock = DataBlock(blocks=blocks, 
                   get_x=ColReader('text'), get_y=ColReader('label'), 
                   splitter=ColSplitter(col='is_valid'))

dls = dblock.dataloaders(imdb_df, bs=4)
```

```python
dls.show_batch(hf_tokenizer=hf_tokenizer, max_n=2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>un - bleeping - believable! meg ryan doesn't even look her usual pert lovable self in this, which normally makes me forgive her shallow ticky acting schtick. hard to believe she was the producer on this dog. plus kevin kline : what kind of suicide trip has his career been on? whoosh... banzai!!! finally this was directed by the guy who did big chill? must be a replay of jonestown - hollywood style. wooofff!</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt; br / &gt; &lt; br / &gt; i saw this on the sci - fi channel. it came on right after the first one. for some reason this movie kept me interested. i don't know why, stop asking. &lt; br / &gt; &lt; br / &gt; - - - spoilers - - - okay... it was cheesy how this guy got involved with the making of the movie. in the first movie, he had a " reason " to kill people, but in this sequal, half of the killings / attempted killings were basicly for no reason. stanley killed the director due to creative differences, he captured the co - writer due to creative differences, but what was the deal with trying to kill off the cast? no cast, no movie. he wanted it to " look real when they died "? if this was supposed to be such a high budget movie, use the special effects, man. of course like the first one, the captured girl gets away, and stanley ends up getting messed up, and dissapears. woooooow ( sarcasm ). this movie had potential. and the saddest thing of all... the really sad part... i would watch a " cabin by the lake 3 ". only because i like judd nelson, and he's the only good part about this sequal.</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


### ... and 🚂

```python
#slow
model = HF_BaseModelWrapper(hf_model)

learn = Learner(dls, 
                model,
                opt_func=partial(Adam, decouple_wd=True),
                loss_func=CrossEntropyLossFlat(),
                metrics=[accuracy],
                cbs=[HF_BaseModelCallback],
                splitter=hf_splitter)

learn.create_opt() 
learn.freeze()

learn.fit_one_cycle(3, lr_max=1e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.643160</td>
      <td>0.432290</td>
      <td>0.835000</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.331133</td>
      <td>0.440774</td>
      <td>0.850000</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.269965</td>
      <td>0.315455</td>
      <td>0.900000</td>
      <td>00:31</td>
    </tr>
  </tbody>
</table>


```python
#slow
learn.show_results(hf_tokenizer=hf_tokenizer, max_n=2)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this very funny british comedy shows what might happen if a section of london, in this case pimlico, were to declare itself independent from the rest of the uk and its laws, taxes &amp; post - war restrictions. merry mayhem is what would happen. &lt; br / &gt; &lt; br / &gt; the explosion of a wartime bomb leads to the discovery of ancient documents which show that pimlico was ceded to the duchy of burgundy centuries ago, a small historical footnote long since forgotten. to the new burgundians, however, this is an unexpected opportunity to live as they please, free from any interference from whitehall. &lt; br / &gt; &lt; br / &gt; stanley holloway is excellent as the minor city politician who suddenly finds himself leading one of the world's tiniest nations. dame margaret rutherford is a delight as the history professor who sides with pimlico. others in the stand - out cast include hermione baddeley, paul duplis, naughton wayne, basil radford &amp; sir michael hordern. &lt; br / &gt; &lt; br / &gt; welcome to burgundy!</td>
      <td>positive</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>first of all ; it's very dilettantish to try describe way of history only from positions of guns, germs and steel. the same tried to do marxists from economical positions. &lt; br / &gt; &lt; br / &gt; the reason of western success can't be just dumb luck, the advantages of domesticated plants and animals. we see, that all around the world any advantages and bonuses are complete useless if they aren't wisely managed. in the japan there isn't huge natural resources, but japan is one of the top world economies, the same situation in singapore, but in nigeria, country with rich oil resources, there are only middle - low success. both of this nations had and still have access to western technology and inventions, but why such gap? &lt; br / &gt; &lt; br / &gt; in the end of movie daimond declared, that it's very important to understand factors of guns, germs and steel, to understand. maybe the main factor of world's difference is not geography, but people ability to understand and use things? the mental ability to understand. and in this case geography is only subordinated.</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


## ❗ Updates

**09/07/2020** 
* Added tests for question/answer and summarization transformer models
* Updated summarization to support BART, T5, and Pegasus

**08/20/2020** 
* Updated everything to work latest version of fastai (tested against 2.0.0)
* Added batch-time padding, so that by default now, `HF_TokenizerTransform` doesn't add any padding tokens and all huggingface inputs are padded simply to the max sequence length in each batch rather than to the max length (passed in and/or acceptable to the model).  This should create efficiencies across the board, from memory consumption to GPU utilization.  The old tried and true method of padding during tokenization requires you to pass in `padding='max_length` to `HF_TextBlock`.
* Removed code to remove fastai2 @patched summary methods which had previously conflicted with a couple of the huggingface transformers

**08/13/2020** 
* Updated everything to work latest transformers and fastai
* Reorganized code to bring it more inline with how huggingface separates out their "tasks".

**07/06/2020** 
* Updated everything to work huggingface>=3.02
* Changed a lot of the internals to make everything more efficient and performant along with the latest version of huggingface ... meaning, I have broken things for folks using previous versions of blurr :).

**06/27/2020** 
* Simplified the `BLURR_MODEL_HELPER.get_hf_objects` method to support a wide range of options in terms of building the necessary huggingface objects (architecture, config, tokenizer, and model).  Also added `cache_dir` for saving pre-trained objects in a custom directory.
* Misc. renaming and cleanup that may break existing code (please see the docs/source if things blow up)
* Added missing required libraries to requirements.txt (e.g., nlp)

**05/23/2020** 
* Initial support for text generation (e.g., summarization, conversational agents) models now included. Only tested with BART so if you try it with other models before I do, lmk what works ... and what doesn't

**05/17/2020** 
* Major code restructuring to make it easier to build out the library.
* `HF_TokenizerTransform` replaces `HF_Tokenizer`, handling the tokenization and numericalization in one place.  DataBlock code has been dramatically simplified.
* Tokenization correctly handles huggingface tokenizers that require `add_prefix_space=True`.
* `HF_BaseModelCallback` and `HF_BaseModelCallback` are required and work together in order to allow developers to tie into any callback friendly event exposed by fastai2 and also pass in named arguments to the huggingface models.
* `show_batch` and `show_results` have been updated for Question/Answer and Token Classification models to represent the data and results in a more easily intepretable manner than the defaults.

**05/06/2020** 
* Initial support for Token classification (e.g., NER) models now included
* Extended fastai's `Learner` object with a `predict_tokens` method used specifically in token classification
* `HF_BaseModelCallback` can be used (or extended) instead of the model wrapper to ensure your inputs into the huggingface model is correct (recommended). See docs for examples (and thanks to fastai's Sylvain for the suggestion!)
* `HF_Tokenizer` can work with strings or a string representation of a list (the later helpful for token classification tasks)
* `show_batch` and `show_results` methods have been updated to allow better control on how huggingface tokenized data is represented in those methods

## ⭐ Props

A word of gratitude to the following individuals, repos, and articles upon which much of this work is inspired from:

- The wonderful community that is the [fastai forum](https://forums.fast.ai/) and especially the tireless work of both Jeremy and Sylvain in building this amazing framework and place to learn deep learning.
- All the great tokenizers, transformers, docs and examples over at [huggingface](https://huggingface.co/)
- [FastHugs](https://github.com/morganmcg1/fasthugs)
- [Fastai with 🤗Transformers (BERT, RoBERTa, XLNet, XLM, DistilBERT)](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2)
- [Fastai integration with BERT: Multi-label text classification identifying toxicity in texts](https://medium.com/@abhikjha/fastai-integration-with-bert-a0a66b1cecbe)
- [A Tutorial to Fine-Tuning BERT with Fast AI](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/)
