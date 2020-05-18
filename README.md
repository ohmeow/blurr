# Title
> summary


<h1>
    <img src="https://ohmeow.github.io/blurr/images/blurr-logo.png" style="width:40px;float:left;"/> blurr
</h1>
<div style="margin-top:60px;">
    Named after the <b>fast</b>est <b>transformer</b> (well, at least of the Autobots), 
    <b>blurr</b> provides both a comprehensive and extensible framework for training and deploying *all* 🤗 
    <a href="https://huggingface.co/transformers/">huggingface</a> transformer models with 
    <div style="display:inline-block">
        <a href="http://dev.fast.ai/">
            <img src="https://avatars0.githubusercontent.com/u/20547620?s=200&v=4" style="width:20px;float:left" />
            fastai v2
        </a>.
    </div>
</div>

Utilizing features like fastai's new `@typedispatch` and `@patch` decorators, and a simple class hiearchy, **blurr** provides fastai developers with the ability to train and deploy transformers for sequence classification, question answer, token classification, summarization, and language modeling tasks. Though much of this works out-of-the-box, users will be able to customize the tokenization strategies and model inputs based on task and/or architecture as needed.

**Supports**:
- Sequence Classification (multiclassification and multi-label classification)
- Token Classification
- Question Answering

*Support for summarization, language modeling, and translation tasks is coming soon!!!*

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
task = HF_TASKS_AUTO.ForSequenceClassification

pretrained_model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(pretrained_model_name)

hf_arch, hf_tokenizer, hf_config, hf_model = BLURR_MODEL_HELPER.get_auto_hf_objects(pretrained_model_name, 
                                                                                    task=task, 
                                                                                    config=config)
```

### Build your 🧱🧱🧱 DataBlock and your DataLoaders

```python
# single input
blocks = (HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer), CategoryBlock)

dblock = DataBlock(blocks=blocks, get_x=ColReader('text'), get_y=ColReader('label'), 
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
      <td>0.680400</td>
      <td>0.661521</td>
      <td>0.545000</td>
      <td>00:19</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.636980</td>
      <td>0.605327</td>
      <td>0.695000</td>
      <td>00:19</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.604755</td>
      <td>0.599580</td>
      <td>0.725000</td>
      <td>00:19</td>
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
      <td>how viewers react to this new " adaption " of shirley jackson's book, which was promoted as not being a remake of the original 1963 movie ( true enough ), will be based, i suspect, on the following : those who were big fans of either the book or original movie are not going to think much of this one... and those who have never been exposed to either, and who are big fans of hollywood's current trend towards " special effects " being the first and last word in how " good " a film is, are going to love it. &lt; br / &gt; &lt; br / &gt; things i did not like about this adaption : &lt; br / &gt; &lt; br / &gt; 1. it was not a true adaption of the book. from the articles i had read, this movie was supposed to cover other aspects in the book that the first one never got around to. and, that seemed reasonable, no film can cover a book word for word unless it is the length of the stand! ( and not even then ) but, there were things in this movie that were never by any means ever mentioned or even hinted at, in the movie. reminded me of the way they decided to kill off the black man in the original movie version of the shining. i didn't like that, either. what the movie's press release should have said is... " we got the basic, very basic, idea from shirley jackson's book, we kept the same names of the house and several ( though not all ) of the leading character's names, but then we decided to write our own story, and, what the heck, we watched the changeling and the shining and ghost first, and decided to throw in a bit of them, too. " &lt; br / &gt; &lt; br / &gt; 2. they completely lost the theme of a parapyschologist inviting carefully picked guest who had all had brushes with the paranormal in their pasts, to investigate a house that truly seemed to have been " born bad ". no, instead, this " doctor " got everyone to the house under the false pretense of studying their " insomnia " ( he really invited them there to scare them to death and then see how they reacted to their fear... like lab rats, who he mentioned never got told they are part of an experiment... nice guy ). this doctor, who did not have the same name, by the way, was as different from the</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>the trouble with the book, " memoirs of a geisha " is that it had japanese surfaces but underneath the surfaces it was all an american man's way of thinking. reading the book is like watching a magnificent ballet with great music, sets, and costumes yet performed by barnyard animals dressed in those costumesso far from japanese ways of thinking were the characters. &lt; br / &gt; &lt; br / &gt; the movie isn't about japan or real geisha. it is a story about a few american men's mistaken ideas about japan and geisha filtered through their own ignorance and misconceptions. so what is this movie if it isn't about japan or geisha? is it pure fantasy as so many people have said? yes, but then why make it into an american fantasy? &lt; br / &gt; &lt; br / &gt; there were so many missed opportunities. imagine a culture where there are no puritanical hang - ups, no connotations of sin about sex. sex is natural and normal. how is sex handled in this movie? right. like it was dirty. the closest thing to a sex scene in the movie has sayuri wrinkling up her nose and grimacing with distaste for five seconds as if the man trying to mount her had dropped a handful of cockroaches on her crotch. &lt; br / &gt; &lt; br / &gt; does anyone actually enjoy sex in this movie? nope. one character is said to be promiscuous but all we see is her pushing away her lover because it looks like she doesn't want to get caught doing something dirty. such typical american puritanism has no place in a movie about japanese geisha. &lt; br / &gt; &lt; br / &gt; did sayuri enjoy her first ravishing by some old codger after her cherry was auctioned off? nope. she lies there like a cold slab of meat on a chopping block. of course she isn't supposed to enjoy it. and that is what i mean about this movie. why couldn't they have given her something to enjoy? why does all the sex have to be sinful and wrong? &lt; br / &gt; &lt; br / &gt; behind mameha the chairman was sayuri's secret patron, and as such he was behind the auction of her virginity. he could have rigged the auction and won her himself. nobu didn't even bid. so why did the chairman let that old codger win her and, reeking of old - man stink,</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


## ❗ Updates

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

## ⭐⭐⭐ Props

A word of gratitude to the following individuals, repos, and articles upon which much of this work is inspired from:

- The wonderful community that is the [fastai forum](https://forums.fast.ai/) and especially the tireless work of both Jeremy and Sylvain in building this amazing framework and place to learn deep learning.
- All the great tokenizers, transformers, docs and examples over at [huggingface](https://huggingface.co/)
- [FastHugs](https://github.com/morganmcg1/fasthugs)
- [Fastai with 🤗Transformers (BERT, RoBERTa, XLNet, XLM, DistilBERT)](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2)
- [Fastai integration with BERT: Multi-label text classification identifying toxicity in texts](https://medium.com/@abhikjha/fastai-integration-with-bert-a0a66b1cecbe)
- [A Tutorial to Fine-Tuning BERT with Fast AI](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/)
