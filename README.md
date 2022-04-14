# blurr
> A library designed for fastai developers who want to train and deploy Hugging Face transformers


Named after the **fast**est **transformer** (well, at least of the Autobots), **BLURR** provides both a comprehensive and extensible framework for training and deploying 🤗 [huggingface](https://huggingface.co/transformers/) transformer models with [fastai](http://docs.fast.ai/) >= 2.0.

Utilizing features like fastai's new `@typedispatch` and `@patch` decorators, along with a simple class hiearchy, **BLURR** provides fastai developers with the ability to train and deploy transformers on a variety of tasks. It includes a high, mid, and low-level API that will allow developers to use much of it out-of-the-box or customize it as needed.

**Supported Text/NLP Tasks**:
- Sequence Classification (multiclassification and multi-label classification)
- Token Classification
- Question Answering
- Summarization
- Tranlsation
- Language Modeling (Causal and Masked)

**Supported Vision Tasks**:
- *In progress*

**Supported Audio Tasks**:
- *In progress*

## Install

You can now pip install blurr via `pip install ohmeow-blurr`

Or, even better as this library is under *very* active development, create an editable install like this:
```
git clone https://github.com/ohmeow/blurr.git
cd blurr
pip install -e ".[dev]"
```

## How to use

Please check the documentation for more thorough examples of how to use this package.

The following two packages need to be installed for blurr to work:
1. fastai (see http://docs.fast.ai/ for installation instructions)
2. huggingface transformers (see https://huggingface.co/transformers/installation.html for details)

### Imports

```python
import torch
from transformers import *
from fastai.text.all import *

from blurr.text.data.all import *
from blurr.text.modeling.all import *
```

### Get your data

```python
path = untar_data(URLs.IMDB_SAMPLE)

model_path = Path("models")
imdb_df = pd.read_csv(path / "texts.csv")

```

### Get `n_labels` from data for config later

```python
n_labels = len(imdb_df["label"].unique())

```

### Get your 🤗 objects

```python
model_cls = AutoModelForSequenceClassification

pretrained_model_name = "bert-base-uncased"

config = AutoConfig.from_pretrained(pretrained_model_name)
config.num_labels = n_labels

hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)

```

### Build your Data 🧱 and your DataLoaders

```python
# single input
blocks = (TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model), CategoryBlock)
dblock = DataBlock(blocks=blocks, get_x=ColReader("text"), get_y=ColReader("label"), splitter=ColSplitter())

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
      <th>target</th>
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
# slow
model = BaseModelWrapper(hf_model)

learn = Learner(
    dls,
    model,
    opt_func=partial(Adam, decouple_wd=True),
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy],
    cbs=[BaseModelCallback],
    splitter=blurr_splitter,
)

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
      <td>0.552160</td>
      <td>0.424542</td>
      <td>0.820000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.429788</td>
      <td>0.279572</td>
      <td>0.895000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.274406</td>
      <td>0.271617</td>
      <td>0.900000</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>


```python
# slow
learn.show_results(learner=learn, max_n=2)

```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
      <th>prediction</th>
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


### Using the high-level Blurr API

Using the high-level API we can reduce DataBlock, DataLoaders, and Learner creation into a ***single line of code***.

Included in the high-level API is a general `BLearner` class (pronouned **"Blurrner"**) that you can use with hand crafted DataLoaders, as well as, task specific BLearners like `BLearnerForSequenceClassification` that will handle everything given your raw data sourced from a pandas DataFrame, CSV file, or list of dictionaries (for example a huggingface datasets dataset)

```python
# slow
learn = BlearnerForSequenceClassification.from_data(imdb_df, pretrained_model_name, dl_kwargs={"bs": 4})

```

```python
# slow
learn.fit_one_cycle(1, lr_max=1e-3)

```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>f1_score</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.511549</td>
      <td>0.411349</td>
      <td>0.817680</td>
      <td>0.835000</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>


```python
# slow
learn.show_results(learner=learn, max_n=2)

```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
      <th>prediction</th>
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


## ⭐ Props

A word of gratitude to the following individuals, repos, and articles upon which much of this work is inspired from:

- The wonderful community that is the [fastai forum](https://forums.fast.ai/) and especially the tireless work of both Jeremy and Sylvain in building this amazing framework and place to learn deep learning.
- All the great tokenizers, transformers, docs, examples, and people over at [huggingface](https://huggingface.co/)
- [FastHugs](https://github.com/morganmcg1/fasthugs)
- [Fastai with 🤗Transformers (BERT, RoBERTa, XLNet, XLM, DistilBERT)](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2)
- [Fastai integration with BERT: Multi-label text classification identifying toxicity in texts](https://medium.com/@abhikjha/fastai-integration-with-bert-a0a66b1cecbe)
- [fastinference](https://muellerzr.github.io/fastinference/)

