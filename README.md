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

hf_arch, hf_config, hf_tokenizer, hf_model = NLP.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)

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
      <td>i rented the dubbed - english version of lensman, hoping that since it came from well - known novels it would have some substance. while there were hints of substance in the movie, it mostly didn't rise above the level of kiddie cartoon. maybe the movie was a bad adaptation of the book, or it lost a lot in the dubbed version. or maybe even the source novels were lightweight. but for whatever reason, there wasn't much there. &lt; br / &gt; &lt; br / &gt; i noticed lots of details that were derivative, sloppy, poorly dramatized, or otherwise deficient. some examples : the opening scenes looked borrowed from the 2001 " star gate " scene and the star wars image of hyperspace. the robot on the harvester looked like an anthropomorphized " r2 - d2 ". &lt; br / &gt; &lt; br / &gt; it starts out trying to borrow its comic relief style of star wars, but mercifully ( since the humor doesn't work ) gives up on comedy and plays it serious. in that sense, it's superior to the star wars franchise, which started with a clever sense of humor, and eventually deteriorated to jar - jar's annoying silliness. &lt; br / &gt; &lt; br / &gt; the agricultural details were apparently drawn by someone who had never seen a farm. the harvester was driving through the unharvested middle of a field, dumping silage onto unharvested crops, rather than working from one side to the other and dumping the silage onto already - harvested rows or into a truck. corn ( maize ) was pouring out the grain chute, but the farm lands were drawn like a wheat field. &lt; br / &gt; &lt; br / &gt; when it was time for kim's father had to face his fate, there wasn't any dramatic weight to the scene. that could have been partly the fault of the english - language voice actor, but the drawings didn't show much weight either. kim's reactions in that scene were similarly unconvincing. &lt; br / &gt; &lt; br / &gt; similarly, when a character named henderson was killed, chris showed very little reaction, even though they were apparently supposed to have been close. ( henderson's death is no spoiler ; his name isn't revealed until his death scene. ) she seems to promptly forget him. someone's expression of sympathy shows more feeling than she does. i think the voice actor deserves most of the blame in that</td>
      <td>negative</td>
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
      <td>0.566530</td>
      <td>0.379829</td>
      <td>0.835000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.349256</td>
      <td>0.327990</td>
      <td>0.875000</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.271702</td>
      <td>0.276152</td>
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
      <td>0.547502</td>
      <td>0.527624</td>
      <td>0.757576</td>
      <td>0.760000</td>
      <td>00:22</td>
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

