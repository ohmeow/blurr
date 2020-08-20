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
1. fastai2 (see http://dev.fast.ai/ for installation instructions)
2. huggingface transformers (see https://huggingface.co/transformers/installation.html for details)

### Imports

```python
import torch
from transformers import *
from fastai2.text.all import *

from blurr.data.all import *
from blurr.modeling.all import *
```

### Get your data 💾

```python
path = untar_data(URLs.IMDB_SAMPLE)

model_path = Path('models')
imdb_df = pd.read_csv(path/'texts.csv')
```

### Get your 🤗 huggingface objects

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


### Build your 🧱🧱🧱 DataBlock 🧱🧱🧱 and your DataLoaders

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
      <td>" national treasure " ( 2004 ) is a thoroughly misguided hodge - podge of plot entanglements that borrow from nearly every cloak and dagger government conspiracy cliche that has ever been written. the film stars nicholas cage as benjamin franklin gates ( how precious is that, i ask you? ) ; a seemingly normal fellow who, for no other reason than being of a lineage of like - minded misguided fortune hunters, decides to steal a'national treasure'that has been hidden by the united states founding fathers. after a bit of subtext and background that plays laughably ( unintentionally ) like indiana jones meets the patriot, the film degenerates into one misguided whimsy after another attempting to create a'stanley goodspeed'regurgitation of nicholas cage and launch the whole convoluted mess forward with a series of high octane, but disconnected misadventures. &lt; br / &gt; &lt; br / &gt; the relevancy and logic to having george washington and his motley crew of patriots burying a king's ransom someplace on native soil, and then, going through the meticulous plan of leaving clues scattered throughout u. s. currency art work, is something that director jon turteltaub never quite gets around to explaining. couldn't washington found better usage for such wealth during the start up of the country? hence, we are left with a mystery built on top of an enigma that is already on shaky ground by the time ben appoints himself the new custodian of this untold wealth. ben's intentions are noble if confusing. he's set on protecting the treasure. for who and when? your guess is as good as mine. &lt; br / &gt; &lt; br / &gt; but there are a few problems with ben's crusade. first up, his friend, ian holmes ( sean bean ) decides that he can't wait for ben to make up his mind about stealing the declaration of independence from the national archives ( oh, yeah brilliant idea! ). presumably, the back of that famous document holds the secret answer to the ultimate fortune. so ian tries to kill ben. the assassination attempt is, of course, unsuccessful, if overly melodramatic. it also affords ben the opportunity to pick up, and pick on, the very sultry curator of the archives, abigail chase ( diane kruger ). she thinks ben is clearly a nut at least at the beginning. but true to action / romance form</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


### ... and train 🚂

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
      <td>0.644810</td>
      <td>0.584691</td>
      <td>0.655000</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.364105</td>
      <td>0.297011</td>
      <td>0.900000</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.261818</td>
      <td>0.298283</td>
      <td>0.880000</td>
      <td>00:32</td>
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
      <td>highly enjoyable, very imaginative, and filmic fairytale all rolled into one, stardust tells the story of a young man living outside a fantasy world going inside it to retrieve a bit of a fallen star only to find the star is alive, young, and beautiful. a kingdom whose king is about to die has said king unleash a competition on his several sons to see who can retrieve a ruby first to be king whilst a trio of witches want the star to carve up and use to keep them young. these three plot threads weave intricately together throughout the entire picture blended with good acting, dazzling special effects, and some solid sentiment and humour as well. stardust is a fun film and has some fun performances from the likes of claire danes as the star ( i could gaze at her for quite some time ) to michelle pfeiffer ( i could gaze at her at full magical powers even longer ) playing the horrible witch to robert deniro playing a nancy - boy air pirate to perfection. charlie cox as the lead tristan is affable and credible and we get some very good work from a group of guys playing the sons out to be king who are constantly and consistently trying to kill off each other. mark strong, jason flemyng, and ruppert everett plays their roles well in both life and death ( loved this whole thread as well ). peter o'toole plays the dying killer daddy and watch for funny man ricky gervais who made me laugh more than anything in the entire film in his brief five minutes ( nice feet ). but the real power in the film is the novel by neil gaiman and the script made from his creative and fertile mind. stardust creates its own mythology and its own world and it works.</td>
      <td>positive</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>


## ❗ Updates

**08/20/2020** 
* Updated everything to work latest version of fastai2 (tested against 0.0.30)
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
