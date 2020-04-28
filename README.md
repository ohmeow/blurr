# blurr
> An extensible integration of huggingface transformer models with fastai v2.


## Install

The library will eventually be available on pypi, but for now ... creating an editable install is the way to go (especially as this is under very active development):
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
from blurr.utils import *
from blurr.data import *
from blurr.modeling import *

import torch
from transformers import *
from fastai2.text.all import *
```

### Get your data

```python
path = untar_data(URLs.IMDB_SAMPLE)

model_path = Path('models')
imdb_df = pd.read_csv(path/'texts.csv')
```

### Get your huggingface objects

```python
task = HF_TASKS_AUTO.ForSequenceClassification

pretrained_model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(pretrained_model_name)

hf_arch, hf_tokenizer, hf_config, hf_model = BLURR_MODEL_HELPER.get_auto_hf_objects(pretrained_model_name, 
                                                                                    task=task, 
                                                                                    config=config)
```

### Build your DataBlock and your DataLoaders

```python
# single input
blocks = (
    HF_TextBlock.from_df(text_cols_lists=[['text']], hf_arch=hf_arch, hf_tokenizer=hf_tokenizer),
    CategoryBlock
)

def get_x(x): return x.text0

dblock = DataBlock(blocks=blocks, get_x=get_x, get_y=ColReader('label'), splitter=ColSplitter(col='is_valid'))

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
      <td>[CLS] raising victor vargas : a review &lt; br / &gt; &lt; br / &gt; you know, raising victor vargas is like sticking your hands into a big, steaming bowl of oatmeal. it's warm and gooey, but you're not sure if it feels right. try as i might, no matter how warm and gooey raising victor vargas became i was always aware that something didn't quite feel right. victor vargas suffers from a certain overconfidence on the director's part. apparently, the director thought that the ethnic backdrop of a latino family on the lower east side, and an idyllic storyline would make the film critic proof. he was right, but it didn't fool me. raising victor vargas is the story about a seventeen - year old boy called, you guessed it, victor vargas ( victor rasuk ) who lives his teenage years chasing more skirt than the rolling stones could do in all the years they've toured. the movie starts off in ` ugly fat'donna's bedroom where victor is sure to seduce her, but a cry from outside disrupts his plans when his best - friend harold ( kevin rivera ) comes - a - looking for him. caught in the attempt by harold and his sister, victor vargas runs off for damage control. yet even with the embarrassing implication that he's been boffing the homeliest girl in the neighborhood, nothing dissuades young victor from going off on the hunt for more fresh meat. on a hot, new york city day they make way to the local public swimming pool where victor's eyes catch a glimpse of the lovely young nymph judy ( judy marte ), who's not just pretty, but a strong and independent too. the relationship that develops between victor and judy becomes the focus of the film. the story also focuses on victor's family that is comprised of his grandmother or abuelita ( altagracia guzman ), his brother nino ( also played by real life brother to victor, silvestre rasuk ) and his sister vicky ( krystal rodriguez ). the action follows victor between scenes with judy and scenes with his family. victor tries to cope with being an oversexed pimp - daddy, his feelings for judy and his grandmother's conservative catholic upbringing. &lt; br / &gt; &lt; br / &gt; the problems that arise from raising victor vargas are a few, but glaring errors. throughout the film you get to know certain characters like vicky, nino, grandma, judy and even [SEP]</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[CLS] many neglect that this isn't just a classic due to the fact that it's the first 3d game, or even the first shoot -'em - up. it's also one of the first stealth games, one of the only ( and definitely the first ) truly claustrophobic games, and just a pretty well - rounded gaming experience in general. with graphics that are terribly dated today, the game thrusts you into the role of b. j. ( don't even * think * i'm going to attempt spelling his last name! ), an american p. o. w. caught in an underground bunker. you fight and search your way through tunnels in order to achieve different objectives for the six episodes ( but, let's face it, most of them are just an excuse to hand you a weapon, surround you with nazis and send you out to waste one of the nazi leaders ). the graphics are, as i mentioned before, quite dated and very simple. the least detailed of basically any 3d game released by a professional team of creators. if you can get over that, however ( and some would suggest that this simplicity only adds to the effect the game has on you ), then you've got one heck of a good shooter / sneaking game. the game play consists of searching for keys, health and ammo, blasting enemies ( aforementioned nazis, and a " boss enemy " per chapter ) of varying difficulty ( which, of course, grows as you move further in the game ), unlocking doors and looking for secret rooms. there is a bonus count after each level is beaten... it goes by how fast you were ( basically, if you beat the'par time ', which is the time it took a tester to go through the same level ; this can be quite fun to try and beat, and with how difficult the levels are to find your way in, they are even challenging after many play - throughs ), how much nazi gold ( treasure ) you collected and how many bad guys you killed. basically, if you got 100 % of any of aforementioned, you get a bonus, helping you reach the coveted high score placings. the game ( mostly, but not always ) allows for two contrastingly different methods of playing... stealthily or gunning down anything and everything you see. you can either run or walk, and amongst your weapons is also a knife... running is heard instantly the moment you enter the same room as the guard, as [SEP]</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>


### ... and train

```python
#slow
model = HF_BaseModelWrapper(hf_model)

learn = Learner(dls, 
                model,
                opt_func=partial(Adam, decouple_wd=True),
                loss_func=CrossEntropyLossFlat(),
                metrics=[accuracy],
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
      <td>0.701890</td>
      <td>0.662111</td>
      <td>0.540000</td>
      <td>00:19</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.639886</td>
      <td>0.648174</td>
      <td>0.590000</td>
      <td>00:19</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.583190</td>
      <td>0.604533</td>
      <td>0.715000</td>
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
      <td>[CLS] how viewers react to this new " adaption " of shirley jackson's book, which was promoted as not being a remake of the original 1963 movie ( true enough ), will be based, i suspect, on the following : those who were big fans of either the book or original movie are not going to think much of this one... and those who have never been exposed to either, and who are big fans of hollywood's current trend towards " special effects " being the first and last word in how " good " a film is, are going to love it. &lt; br / &gt; &lt; br / &gt; things i did not like about this adaption : &lt; br / &gt; &lt; br / &gt; 1. it was not a true adaption of the book. from the articles i had read, this movie was supposed to cover other aspects in the book that the first one never got around to. and, that seemed reasonable, no film can cover a book word for word unless it is the length of the stand! ( and not even then ) but, there were things in this movie that were never by any means ever mentioned or even hinted at, in the movie. reminded me of the way they decided to kill off the black man in the original movie version of the shining. i didn't like that, either. what the movie's press release should have said is... " we got the basic, very basic, idea from shirley jackson's book, we kept the same names of the house and several ( though not all ) of the leading character's names, but then we decided to write our own story, and, what the heck, we watched the changeling and the shining and ghost first, and decided to throw in a bit of them, too. " &lt; br / &gt; &lt; br / &gt; 2. they completely lost the theme of a parapyschologist inviting carefully picked guest who had all had brushes with the paranormal in their pasts, to investigate a house that truly seemed to have been " born bad ". no, instead, this " doctor " got everyone to the house under the false pretense of studying their " insomnia " ( he really invited them there to scare them to death and then see how they reacted to their fear... like lab rats, who he mentioned never got told they are part of an experiment... nice guy ). this doctor, who did not have the same name, by the way, was as different from the [SEP]</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[CLS] the trouble with the book, " memoirs of a geisha " is that it had japanese surfaces but underneath the surfaces it was all an american man's way of thinking. reading the book is like watching a magnificent ballet with great music, sets, and costumes yet performed by barnyard animals dressed in those costumesso far from japanese ways of thinking were the characters. &lt; br / &gt; &lt; br / &gt; the movie isn't about japan or real geisha. it is a story about a few american men's mistaken ideas about japan and geisha filtered through their own ignorance and misconceptions. so what is this movie if it isn't about japan or geisha? is it pure fantasy as so many people have said? yes, but then why make it into an american fantasy? &lt; br / &gt; &lt; br / &gt; there were so many missed opportunities. imagine a culture where there are no puritanical hang - ups, no connotations of sin about sex. sex is natural and normal. how is sex handled in this movie? right. like it was dirty. the closest thing to a sex scene in the movie has sayuri wrinkling up her nose and grimacing with distaste for five seconds as if the man trying to mount her had dropped a handful of cockroaches on her crotch. &lt; br / &gt; &lt; br / &gt; does anyone actually enjoy sex in this movie? nope. one character is said to be promiscuous but all we see is her pushing away her lover because it looks like she doesn't want to get caught doing something dirty. such typical american puritanism has no place in a movie about japanese geisha. &lt; br / &gt; &lt; br / &gt; did sayuri enjoy her first ravishing by some old codger after her cherry was auctioned off? nope. she lies there like a cold slab of meat on a chopping block. of course she isn't supposed to enjoy it. and that is what i mean about this movie. why couldn't they have given her something to enjoy? why does all the sex have to be sinful and wrong? &lt; br / &gt; &lt; br / &gt; behind mameha the chairman was sayuri's secret patron, and as such he was behind the auction of her virginity. he could have rigged the auction and won her himself. nobu didn't even bid. so why did the chairman let that old codger win her and, reeking of old - man stink, [SEP]</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>

