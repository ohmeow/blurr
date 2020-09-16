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
      <td>now that che ( 2008 ) has finished its relatively short australian cinema run ( extremely limited release : 1 screen in sydney, after 6wks ), i can guiltlessly join both hosts of " at the movies " in taking steven soderbergh to task. &lt; br / &gt; &lt; br / &gt; it's usually satisfying to watch a film director change his style / subject, but soderbergh's most recent stinker, the girlfriend experience ( 2009 ), was also missing a story, so narrative ( and editing? ) seem to suddenly be soderbergh's main challenge. strange, after 20 - odd years in the business. he was probably never much good at narrative, just hid it well inside " edgy " projects. &lt; br / &gt; &lt; br / &gt; none of this excuses him this present, almost diabolical failure. as david stratton warns, " two parts of che don't ( even ) make a whole ". &lt; br / &gt; &lt; br / &gt; epic biopic in name only, che ( 2008 ) barely qualifies as a feature film! it certainly has no legs, inasmuch as except for its uncharacteristic ultimate resolution forced upon it by history, soderbergh's 4. 5hrs - long dirge just goes nowhere. &lt; br / &gt; &lt; br / &gt; even margaret pomeranz, the more forgiving of australia's at the movies duo, noted about soderbergh's repetitious waste of ( hd digital storage ) : " you're in the woods... you're in the woods... you're in the woods... ". i too am surprised soderbergh didn't give us another 2. 5hrs of that somewhere between his existing two parts, because he still left out massive chunks of che's " revolutionary " life! &lt; br / &gt; &lt; br / &gt; for a biopic of an important but infamous historical figure, soderbergh unaccountably alienates, if not deliberately insults, his audiences by &lt; br / &gt; &lt; br / &gt; 1. never providing most of che's story ; &lt; br / &gt; &lt; br / &gt; 2. imposing unreasonable film lengths with mere dullard repetition ; &lt; br / &gt; &lt; br / &gt; 3. ignoring both true hindsight and a narrative of events ; &lt; br / &gt; &lt; br / &gt; 4. barely developing an idea, or a character ;</td>
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
      <td>0.617441</td>
      <td>0.512695</td>
      <td>0.720000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.320651</td>
      <td>0.410647</td>
      <td>0.845000</td>
      <td>00:23</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.302557</td>
      <td>0.313904</td>
      <td>0.885000</td>
      <td>00:23</td>
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
      <td>the trouble with the book, " memoirs of a geisha " is that it had japanese surfaces but underneath the surfaces it was all an american man's way of thinking. reading the book is like watching a magnificent ballet with great music, sets, and costumes yet performed by barnyard animals dressed in those far from japanese ways of thinking were the characters. &lt; br / &gt; &lt; br / &gt; the movie isn't about japan or real geisha. it is a story about a few american men's mistaken ideas about japan and geisha filtered through their own ignorance and misconceptions. so what is this movie if it isn't about japan or geisha? is it pure fantasy as so many people have said? yes, but then why make it into an american fantasy? &lt; br / &gt; &lt; br / &gt; there were so many missed opportunities. imagine a culture where there are no puritanical hang - ups, no connotations of sin about sex. sex is natural and normal. how is sex handled in this movie? right. like it was dirty. the closest thing to a sex scene in the movie has sayuri wrinkling up her nose and grimacing with distaste for five seconds as if the man trying to mount her had dropped a handful of cockroaches on her crotch. &lt; br / &gt; &lt; br / &gt; does anyone actually enjoy sex in this movie? nope. one character is said to be promiscuous but all we see is her pushing away her lover because it looks like she doesn't want to get caught doing something dirty. such typical american puritanism has no place in a movie about japanese geisha. &lt; br / &gt; &lt; br / &gt; did sayuri enjoy her first ravishing by some old codger after her cherry was auctioned off? nope. she lies there like a cold slab of meat on a chopping block. of course she isn't supposed to enjoy it. and that is what i mean about this movie. why couldn't they have given her something to enjoy? why does all the sex have to be sinful and wrong? &lt; br / &gt; &lt; br / &gt; behind mameha the chairman was sayuri's secret patron, and as such he was behind the auction of her virginity. he could have rigged the auction and won her himself. nobu didn't even bid. so why did the chairman let that old codger win her and, reeking of old - man stink, get</td>
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
