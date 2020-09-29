## Introduction

The primary goal of this project is to create a model that can accurately classify whether a social media comment is sarcastic or not. When given data containing a comment, parent comment, and topic, the model should be able to use this information in order to determine whether the the comment sentiment is sarcastic. Such a model could be incredibly important in the field of sentiment analysis as current NLP being used to understand subjective opinions doesn’t account for the fact that people may be sarcastic in their thoughts. This would entail that sentiment analysis cannot account for the intentions of people’s comments and needs to be more contextual. An application of this model would be for natural language bots/virtual assistants that may generate their own sarcastic comments  to seem more human-like. The overall impact of this on NLP and sentiment analysis would be very helpful to understanding further conversational, contextual analysis done by AI.

## Methods

The training data that we plan to use for this comes from the following:

- [Kaggle - Detect Sarcasm in Comments](https://www.kaggle.com/sachinichake/detect-sarcasm-in-comments?select=Train.csv)
- [Kaggle - Sarcasm on Reddit](https://www.kaggle.com/danofer/sarcasm)
- [Corpus of Sarcasm in Twitter Conversations](https://mendeley.figshare.com/articles/Corpus_of_Sarcasm_in_Twitter_Conversations/8962883)

Our data consists solely of tweets and comments on Reddit. The datasets provided by Kaggle contain over 1 millions preprocessed comments and other data. Our model will likely be trained by splitting the dataset into a train and test group. Another possible way for us to run this is by training it on one dataset and then testing it on others to find its accuracy. 

Our current plan is to take a variety of popular NLP algorithms like LSTM (long short-term memory) or BERT (Bidirectional Encoder Representations from Transformers) and modify them so that they can better detect sarcasm from multiple input strings. Another idea may be to analyze contextual sentences for keywords and underlying sentiment as sarcastic comments will likely have the opposite tone. This is subject to change as more research into this topic may reveal that there would be a more optimal way to implement a sarcasm detector.

## Results

Our team hopes to generate a model that would be able to accurately classify comments as sarcastic given some degree of context. In a limited sense, this would allow us to see the number of tweets and reddit comments that are sarcastic. More generally though, our results could be very informative for the field of sentiment analysis and natural language processing. 

## Discussion

After our model is finished, we would ideally like to test it on a variety of different data sets so we can determine its accuracy. This would help us determine if it is overfit to the data that we have. The methods that we have outlined for how we would perform testing for our model lead us to believe that if it is overfit, it would be a fault of the algorithm due to the nature of the way that we are running tests. 

One of the ways we would like to use our model is to do an analysis of the tweets done by a variety of famous figures. Political figures, athletes, movie stars, etc. are all followed on twitter and a sarcasm model run on their tweets would be important to determine how seriously they should be taken. In some cases, it could even be a measure of how appropriately they handle their position. 

Another way our model would be effective is in the field of NLP itself. A model that is able to understand context dependent speech would be useful for chatbots or other interactive, intelligent models. We see these becoming more and more relevant as technology progresses towards devices that contain Siri, Alexa, etc. Our methodology could be used to train and progress a variety of speech based, conversational bots.

## References

- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf)
- [https://www.aclweb.org/anthology/C14-1022.pdf](https://www.aclweb.org/anthology/C14-1022.pdf)
- [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.800.4972&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.800.4972&rep=rep1&type=pdf)

## Touch-Point 1 Deliverables

#### Project Proposal Video

#### Project Proposal Slide
