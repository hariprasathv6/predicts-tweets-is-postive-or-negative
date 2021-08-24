#Sentimental analysis using the tweet_sample modules data

#import twittersample module 
#import the data
import nltk
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
print(text)
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
print(tweet_tokens[0])

#pos 
from nltk.tag import pos_tag
print(pos_tag(tweet_tokens[0]))

print(remove_noise(tweet_tokens[0], stop_words))

#stemming module

#stop words module (remove the stopwords from the text)   
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(remove_noise(tweet_tokens[0], stop_words))


def lemmatized_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence=[]
    for word,tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos ="v"
        else:
            pos = "a"
        lemmatized_senetence.append(lemmatizer.lemmatize(word,pos))  
    return lemmatized_sentence    
    
    
from nltk.stem.wordnet import WordNetLemmatizer 

import re,string

#Remove the noise from the data
def remove_noise(tweet_tokens,stop_words=()):
    cleaned_tokens =[]
    
    for token,tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos='a'
            
    lemmatizer = WordNetLemmatizer()
    token = lemmatizer.lemmatize(token,pos)
    
        
    
    if len(token)>0 and token not in string.punctuation and token.lower() not in stop_words:
        cleaned_tokens.append(token.lower())
            
    return cleaned_tokens





#tokenizing the data (split the paragraph,sentence to words)
positive_tweets_tokens = twitter_samples.tokenized("positive_tweets.json")
negative_tweets_tokens = twitter_samples.tokenized("negative_tweets.json")

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list =[]

for tokens in positive_tweets_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens,stop_words))
    
for tokens in negative_tweets_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens,stop_words))
    


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
all_pos_words = get_all_words(positive_cleaned_tokens_list)  
all_neg_words = get_all_words(negative_cleaned_tokens_list) 


from nltk import FreqDist
freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True] for token in tweet_tokens)
        
positive_tokens_for_model=get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model=get_tweets_for_model(negative_cleaned_tokens_list)

import random
positive_dataset = [(tweet_dict,"Positive")
                       for tweet_dict in positive_tokens_for_model]  
negative_dataset = [(tweet_dict,"Negative")
                         for tweet_dict in negative_tokens_for_model] 
print(positive_dataset)

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data=dataset[:7000]
test_data=dataset[7000:]



#model build
from nltk import classify
from nltk import NaiveBayesClassifier
#model =NaiveBayesClassifier()
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is :",classify.accuracy(classifier,test_data))



import nltk
nltk.download('punkt')
#
custom_tweets = "I ordered just once %from TerribleCo, they #screwed   7 up, never  @ram used the app again."
custom_tokens = remove_noise(nltk.word_tokenize(custom_tweets))
#custom_tokens
print(custom_tweets,classifier.classify(dict([token,True] for token in custom_tokens)))
print("Accuracy :",classify.accuracy(classifier,custom_tokens))
