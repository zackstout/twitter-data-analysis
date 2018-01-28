import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# nltk.download()

# if we just run the above, we'll get a nice window pop up that lets you download everything you need.

# tokenizers: word or sentence.
# corpora = body of text, e.g. medical journals.
# lexicon = dictionary of words and meanings

example = 'Hi there. Are you a fox? I am a bear. Well it is nice to meet you Mr. fox.'

print(sent_tokenize(example))
stop_words = set(stopwords.words("english"))

print(stop_words)
