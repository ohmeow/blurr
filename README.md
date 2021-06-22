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

### Get `n_labels` from data for config later

```python
n_labels = len(imdb_df['label'].unique())
```

### Get your 🤗 objects

```python
model_cls = AutoModelForSequenceClassification

pretrained_model_name = "bert-base-uncased"

config = AutoConfig.from_pretrained(pretrained_model_name)
config.num_labels = n_labels

hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)
```

### Build your Data 🧱 and your DataLoaders

```python
# single input
blocks = (HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model), CategoryBlock)
dblock = DataBlock(blocks=blocks,  get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())

dls = dblock.dataloaders(imdb_df, bs=4)
```

```python
dls.show_batch(dataloaders=dls, max_n=2)
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
      <td>raising victor vargas : a review &lt; br / &gt; &lt; br / &gt; you know, raising victor vargas is like sticking your hands into a big, steaming bowl of oatmeal. it's warm and gooey, but you're not sure if it feels right. try as i might, no matter how warm and gooey raising victor vargas became i was always aware that something didn't quite feel right. victor vargas suffers from a certain overconfidence on the director's part. apparently, the director thought that the ethnic backdrop of a latino family on the lower east side, and an idyllic storyline would make the film critic proof. he was right, but it didn't fool me. raising victor vargas is the story about a seventeen - year old boy called, you guessed it, victor vargas ( victor rasuk ) who lives his teenage years chasing more skirt than the rolling stones could do in all the years they've toured. the movie starts off in ` ugly fat'donna's bedroom where victor is sure to seduce her, but a cry from outside disrupts his plans when his best - friend harold ( kevin rivera ) comes - a - looking for him. caught in the attempt by harold and his sister, victor vargas runs off for damage control. yet even with the embarrassing implication that he's been boffing the homeliest girl in the neighborhood, nothing dissuades young victor from going off on the hunt for more fresh meat. on a hot, new york city day they make way to the local public swimming pool where victor's eyes catch a glimpse of the lovely young nymph judy ( judy marte ), who's not just pretty, but a strong and independent too. the relationship that develops between victor and judy becomes the focus of the film. the story also focuses on victor's family that is comprised of his grandmother or abuelita ( altagracia guzman ), his brother nino ( also played by real life brother to victor, silvestre rasuk ) and his sister vicky ( krystal rodriguez ). the action follows victor between scenes with judy and scenes with his family. victor tries to cope with being an oversexed pimp - daddy, his feelings for judy and his grandmother's conservative catholic upbringing. &lt; br / &gt; &lt; br / &gt; the problems that arise from raising victor vargas are a few, but glaring errors. throughout the film you get to know certain characters like vicky, nino, grandma, judy and even</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>the shop around the corner is one of the sweetest and most feel - good romantic comedies ever made. there's just no getting around that, and it's hard to actually put one's feeling for this film into words. it's not one of those films that tries too hard, nor does it come up with the oddest possible scenarios to get the two protagonists together in the end. in fact, all its charm is innate, contained within the characters and the setting and the plot... which is highly believable to boot. it's easy to think that such a love story, as beautiful as any other ever told, * could * happen to you... a feeling you don't often get from other romantic comedies, however sweet and heart - warming they may be. &lt; br / &gt; &lt; br / &gt; alfred kralik ( james stewart ) and clara novak ( margaret sullavan ) don't have the most auspicious of first meetings when she arrives in the shop ( matuschek &amp; co. ) he's been working in for the past nine years, asking for a job. they clash from the very beginning, mostly over a cigarette box that plays music when it's opened - - he thinks it's a ludicrous idea ; she makes one big sell of it and gets hired. their bickering takes them through the next six months, even as they both ( unconsciously, of course! ) fall in love with each other when they share their souls and minds in letters passed through po box 237. this would be a pretty thin plotline to base an entire film on, except that the shop around the corner is expertly fleshed - out with a brilliant supporting cast made up of entirely engaging characters, from the fatherly but lonely hugo matuschek ( frank morgan ) himself, who learns that his shop really is his home ; pirovitch ( felix bressart ), kralik's sidekick and friend who always skitters out of the room when faced with the possibility of being asked for his honest opinion ; smarmy pimp - du - jour vadas ( joseph schildkraut ) who ultimately gets his comeuppance from a gloriously righteous kralik ; and ambitious errand boy pepi katona ( william tracy ) who wants nothing more than to be promoted to the position of clerk for matuschek &amp; co. the unpretentious love story between '</td>
      <td>positive</td>
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
      <td>0.643662</td>
      <td>0.518397</td>
      <td>0.750000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.370039</td>
      <td>0.358643</td>
      <td>0.845000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.244052</td>
      <td>0.303162</td>
      <td>0.880000</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>


```python
#slow
learn.show_results(learner=learn, max_n=2)
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
      <td>the trouble with the book, " memoirs of a geisha " is that it had japanese surfaces but underneath the surfaces it was all an american man's way of thinking. reading the book is like watching a magnificent ballet with great music, sets, and costumes yet performed by barnyard animals dressed in those costumesso far from japanese ways of thinking were the characters. &lt; br / &gt; &lt; br / &gt; the movie isn't about japan or real geisha. it is a story about a few american men's mistaken ideas about japan and geisha filtered through their own ignorance and misconceptions. so what is this movie if it isn't about japan or geisha? is it pure fantasy as so many people have said? yes, but then why make it into an american fantasy? &lt; br / &gt; &lt; br / &gt; there were so many missed opportunities. imagine a culture where there are no puritanical hang - ups, no connotations of sin about sex. sex is natural and normal. how is sex handled in this movie? right. like it was dirty. the closest thing to a sex scene in the movie has sayuri wrinkling up her nose and grimacing with distaste for five seconds as if the man trying to mount her had dropped a handful of cockroaches on her crotch. &lt; br / &gt; &lt; br / &gt; does anyone actually enjoy sex in this movie? nope. one character is said to be promiscuous but all we see is her pushing away her lover because it looks like she doesn't want to get caught doing something dirty. such typical american puritanism has no place in a movie about japanese geisha. &lt; br / &gt; &lt; br / &gt; did sayuri enjoy her first ravishing by some old codger after her cherry was auctioned off? nope. she lies there like a cold slab of meat on a chopping block. of course she isn't supposed to enjoy it. and that is what i mean about this movie. why couldn't they have given her something to enjoy? why does all the sex have to be sinful and wrong? &lt; br / &gt; &lt; br / &gt; behind mameha the chairman was sayuri's secret patron, and as such he was behind the auction of her virginity. he could have rigged the auction and won her himself. nobu didn't even bid. so why did the chairman let that old codger win her and, reeking of old - man stink,</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt; br / &gt; &lt; br / &gt; i'm sure things didn't exactly go the same way in the real life of homer hickam as they did in the film adaptation of his book, rocket boys, but the movie " october sky " ( an anagram of the book's title ) is good enough to stand alone. i have not read hickam's memoirs, but i am still able to enjoy and understand their film adaptation. the film, directed by joe johnston and written by lewis colick, records the story of teenager homer hickam ( jake gyllenhaal ), beginning in october of 1957. it opens with the sound of a radio broadcast, bringing news of the russian satellite sputnik, the first artificial satellite in orbit. we see a images of a blue - gray town and its people : mostly miners working for the olga coal company. one of the miners listens to the news on a hand - held radio as he enters the elevator shaft, but the signal is lost as he disappears into the darkness, losing sight of the starry sky above him. a melancholy violin tune fades with this image. we then get a jolt of elvis on a car radio as words on the screen inform us of the setting : october 5, 1957, coalwood, west virginia. homer and his buddies, roy lee cook ( william lee scott ) and sherman o'dell ( chad lindberg ), are talking about football tryouts. football scholarships are the only way out of the town, and working in the mines, for these boys. " why are the jocks the only ones who get to go to college, " questions homer. roy lee replies, " they're also the only ones who get the girls. " homer doesn't make it in football like his older brother, so he is destined for the mines, and to follow in his father's footsteps as mine foreman. until he sees the dot of light streaking across the october sky. then he wants to build a rocket. " i want to go into space, " says homer. after a disastrous attempt involving a primitive rocket and his mother's ( natalie canerday ) fence, homer enlists the help of the nerdy quentin wilson ( chris owen ). quentin asks homer, " what do you want to know about rockets? " homer quickly anwers, " everything. " his science teacher at big creek high school, miss frieda riley ( laura dern ) greatly supports homer, and</td>
      <td>positive</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>


## ❗ Updates

**06/16/2021**

* Updated to work with fastai 2.4
* Removed `blurr_summary` as `Learner.summary` works with fastai 2.4
* Updated `Learner.lr_find` code in docs to the updated API in fastai 2.4


**06/10/2021**

* Updated to work with Huggingface 4.6.x
* Added MLM fine-tuning
* Reorganized code/docs

**05/04/2021**

***The "May the Fourth be with you" release:***
* Updated to work with Huggingface 4.5.x and Fastai 2.3.1 (there is a bug in 2.3.0 that breaks blurr so make sure you are using the latest)
* Fixed Github issues [#36](https://github.com/ohmeow/blurr/issues/36), [#34](https://github.com/ohmeow/blurr/issues/34)
* Misc. improvements to get blurr in line with the upcoming Huggingface 5.0 release


* A few **breaking changes**:

1. `BLURR_MODEL_HELPER` is now just `BLURR`


2. Task specific auto models need to be built using the new Huggingface `AutoModelFor<Insert task here>` classes. The docs have been updated to show you how it works; the prior way of building such models not longer works.


**12/31/2020**

***The "Goodbye 2020" release with lots of goodies for blurr users:***
* Updated the Seq2Seq models to use some of the latest huggingface bits like `tokenizer.prepare_seq2seq_batch`.
* Separated out the Seq2Seq and Token Classification metrics into metrics-specific callbacks for a better separation of concerns. As a best practice, you should now *only* use them as `fit_one_cycle`, etc.. callbacks rather than attach them to your `Learner`.
* **NEW**: Translation are now available in blurr, joining causal language modeling and summarization in our core Seq2Seq stack
* **NEW**: Integration of huggingface's Seq2Seq metrics (rouge, bertscore, meteor, bleu, and sacrebleu). Plenty of info on how to set this up in the docs.
* **NEW**: Added `default_text_gen_kwargs`, a method that given a huggingface config, model, and task (optional), will return the default/recommended kwargs for any text generation models.
* A lot of code cleanup (e.g., refactored naming and removal of redundant code into classes/methods)
* **More model support** and **more tests** across the board!  Check out the docs for more info
* Misc. validation improvements and bug fixes.

As I'm sure there is plenty I can do to make this library better, please don't hesitate to join in and help the effort by submitting PRs, pointing out problems with my code, or letting me know what and how I can improve things generally. Some models, like mbart and mt5 for example, aren't giving good results and I'd love to get any and all feedback from the community on how to resolve such issues ... so hit me up, I promise I won't bit :)

**12/20/2020** 
* Updated `Learner.blurr_predict` and `Learner.blurr_predict_tokens` to support single or multiple items
* Added ONNX support for sequence classification, token classification, and question/answering tasks. `blurrONNX` provides ONNX friendly variants of `Learner.blurr_predict` and `Learner.blurr_predict_tokens` in the form of `blurrONNX.predict` and `blurrONNX.predict_tokens` respectively.  Like their Learner equivalents, these methods support single or multiple items for inferece.  See the docs/code for examples and speed gains you get with ONNX.
* Added quantization support when converting your blurr models to ONNX.
* Requires fast.ai >= 2.1.5 and huggingface transformers >= 4.x

**12/12/2020** 
* Updated to work with the latest version of fast.ai (2.1.8) and huggingface transformers >= 4.x
* Fixed `Learner.blurr_summary` to work with fast.ai >= 2.1.8
* Fixed inclusion of `add_prefix_space` in tokenizer `BLURR_MODEL_HELPER`
* Fixed token classification `show_results` for tokenizers that add a prefix space
* Notebooks run with environment variable "TOKENIZERS_PARALLELISM=false" to avoid fast tokenizer warnings
* Updated docs

**11/12/2020** 
* Updated documentation
* Updated model callbacks to support mixed precision training regardless of whether you are calculating the loss yourself or letting huggingface do it for you.

**11/10/2020** 
* Major update just about everywhere to facilitate a breaking change in fastai's treatment of `before_batch` transforms.
* Reorganized code as I being to work on LM and other text2text tasks
* Misc. fixes

**10/08/2020** 
* Updated all models to use [ModelOutput](https://huggingface.co/transformers/main_classes/output.html) classes instead of traditional tuples. `ModelOutput` attributes are assigned to the appropriate fastai bits like `Learner.pred` and `Learner.loss` and anything else you've requested the huggingface model to return is available via the `Learner.blurr_model_outputs` dictionary (see next two bullet items)
* Added ability to grab attentions and hidden state from `Learner`. You can get at them via `Learner.blurr_model_outputs` dictionary if you tell `HF_BaseModelWrapper` to provide them.
* Added `model_kwargs` to `HF_BaseModelWrapper` should you need to request a huggingface model to return something specific to it's type. These outputs will be available via the `Learner.blurr_model_outputs` dictionary as well.

**09/16/2020** 
* Major overhaul to do *everything* at batch time (including tokenization/numericalization). If this backfires, I'll roll everything back but as of now, I think this approach not only meshes better with how huggingface tokenization works and reduce RAM utilization for big datasets, but also opens up opportunities for incorporating augmentation, building adversarial models, etc....  Thoughts?
* Added tests for summarization bits
* New change may require some small modifications (see docs or ask on issues thread if you have problems you can't fiture out).  I'm NOT doing a release until pypi until folks have a chance to work with the latest.

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
- [fastinference](https://muellerzr.github.io/fastinference/)
