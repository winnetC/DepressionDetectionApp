ACCESS_TOKEN = "1050742769692631041-2MwPvr6kQwj4vwyzyWlYyUA9wjkU5x"
ACCESS_TOKEN_SECRET = "DSWN3yfTP8H1gTWcfe7Wtwl3sh7yoisquXfMgQIXY55MH"
CONSUMER_KEY = "GWyZuCXP2U5PG05CbpSw3EIbR"
CONSUMER_SECRET = "Gxv0OV2eqRuMKRsLKAE1SMYHyLUwMkWN3AelCQdwVVffFNSQM1"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pandas as pd
import numpy as np
import tweepy as tw
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from textblob import TextBlob
import re
import matplotlib.pyplot as plt

class TweetAnalyzer():
    """
    Functionality for analyzing and categorising content from tweets
    """
    def clean_tweet(self,tweet):
        return ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())

    def analyze_sentiment(self,tweet):
        analysis=TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self,tweets):
        df=pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['tweets'])
        df['id']=np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        # df['date'] = np.array([tweet.created_at for tweet in tweets])
        # df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        return df

class TwitterClient():
    def __init__(self,twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client=API(self.auth)

        self.twitter_user=twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets=[]
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth

class TwitterStreamer():
    """
    Class for streaming and processing live tweets
    """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles twitter authentication and the connection to the twitter streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth=self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)
        #this line filters Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)

class TwitterListener(StreamListener):
    """
    This is a basic listener class that just prints received tweets to stout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            # print(data)
            with open(self.fetched_tweets_filename,'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" %str(e))
        return True

    def on_error(self, status):
        if status==420:
            # Returning False on data method in case rate limit occurs
            return False
        print(status)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.svm import SVC

def DTreeClassifier():
  dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
  dtree.fit(Xtrain, Ytrain)
  return dtree

def VotClassifier():
  log_clf = LogisticRegression()
  rnd_clf = RandomForestClassifier()
  svm_clf = SVC()
  voting_clf = VotingClassifier( estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
  voting_clf.fit(Xtrain, Ytrain)
  return voting_clf

def RForestClassifier():
  rnd_clf = RandomForestClassifier()
  rnd_clf.fit(Xtrain,Ytrain)
  return rnd_clf

depressionDf = pd.read_csv('UserlistB.csv')
depressionDf['Depression_level'],_ = pd.factorize(depressionDf['Depression_level'])
# print(depressionDf.head(10))
X=depressionDf.drop(['Depression_level','Username'],axis=1)
Y=depressionDf['Depression_level']
scale=MinMaxScaler()
X=scale.fit_transform(X)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2, random_state=42)

tweet_analyzer=TweetAnalyzer()
twitter_client=TwitterClient()
api=twitter_client.get_twitter_client_api()

def diagnosis_for_neutral():
    neutral=[st.text("Chances of getting depression are equally as likely as those of not getting depression."),
             st.text(""),
             st.text("Watch out for these symptoms:"),
             st.text(""),
             st.text("   -Feelings of sadness, tearfulness,emptiness or hopelessness."),
             st.text("   -Angry outbursts, irritability or frustration, even over small matters."),
             st.text("   -Loss of interest or pleasure in most or all normal activities."),
             st.text("   -Sleep disturbances, including insomnia or sleeping too much."),
             st.text("   -Frequent or recurrent thoughts of death, suicidal thoughts, suicide attempts or suicide"),
             st.text("   -Unexplained physical problems, such as back pain or headaches"),
             st.text("   -Tiredness and lack of energy, so even small tasks take extra effort."),
             st.text("   -Anxiety, agitation or restlessness."),
             st.text("   -Slowed thinking, speaking or body movements."),
             st.text("   -Feelings of worthlessness or guilt, fixating on past failures or self-blame."),
             st.text("   -Trouble thinking, concentrating, making decisions and remembering things,")]
    return neutral

def diagnosis_for_low():
    low=st.text("Youre doing great !")
    return low

def diagnosis_for_high():
    high=[st.text("Tips to manage depression:"),
          st.text(""),
          st.text("*Take care of your physical health"),
          st.text("   -Get active!"),
          st.text("   -Nourish your body"),
          st.text("   -Get adequate sleep"),
          st.text(""),
          st.text("*Take a closer look at your thoughts"),
          st.text("   -Limit overthinking"),
          st.text("   -Challenge negative thoughts"),
          st.text(""),
          st.text("*Identify unhelpful behaviors and replace them with healthy, helpful behaviors"),
          st.text("   -Set realistic and achievable daily goals"),
          st.text("   -Avoid procastination"),
          st.text("   -Stay connected to friends and family"),
          st.text("   -Engage in healthy joyful activities"),
          st.text(""),
          st.text("*Practice self-compassion"),
          st.text(""),
          st.text("*Review micro-success daily"),
          st.text(""),
          st.text("HOWEVER, if you are experiencing severe depressive symptoms, it may be time to seek out professional help")]
    return high

def main():
    st.title("Depression Prediction App")
    submenu=["Plot","Prediction"]
    st.subheader("What is Depression ?")
    st.text("")
    st.text("Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. ")
    st.text("Also called major depressive disorder or clinical depression, it affects how you feel, ")
    st.text("think and behave and can lead to a variety of emotional and physical problems.")
    st.text("")
    st.text("")
    activity=st.selectbox("Activity",submenu)
    if activity == "Plot":
        st.subheader("Data Vis Plot")
        dataF=pd.read_csv('UserlistB.csv')
        st.dataframe(dataF.head(10))
        dataF['Depression_level'].value_counts().plot(kind='bar')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        if st.checkbox("Area Chart"):
            all_columns=dataF.columns.to_list()
            feat_choices=st.multiselect("Choose a Feature",all_columns)
            new_df=dataF[feat_choices]
            st.area_chart(new_df)

    elif activity == "Prediction":
        st.subheader("Predictive Analysis")
        st.text(" ")
        st.info("Please provide the required information below for the prediction")
        st.text(" ")
        name=st.text_input("Enter Twitter Username: ")

        modelChoice=st.selectbox("Select Model",["DecisionTree Cl","Voting Cl","RandomForest Cl"])
        if st.button("Predict"):
            if name=="":
                st.text("Enter Username!")
            else:
                username = name
                status = ""
                # extracting the first 100 tweets of each user
                try:
                    tweets = api.user_timeline(screen_name=username, count=100)
                    # print(tweets)

                    # determine the sentiment of each tweet that has been cleaned by the clean_tweet function
                    df2 = tweet_analyzer.tweets_to_data_frame(tweets)
                    df2['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df2['tweets']])
                    # print(df2.head())

                    # calculating the mean of the sentiments of 100 tweets and classifying the value
                    value = df2['sentiment'].describe()['mean']
                    if value > 0:
                        status = "low"
                    # print(status)
                    elif value < 0:
                        status = "high"
                    # print(status)
                    else:
                        status = "neutral"
                    # print(status)

                    # extracting total likes,retweets and length of the 100 texts of each user
                    likes = sum(df2['likes'])
                    retweets = sum(df2['retweets'])
                    length = sum(df2['len'])

                    data = [likes, retweets, length]
                    singleSample = np.array(data).reshape(1, -1)

                    if modelChoice == "DecisionTree Cl":
                        prediction=DTreeClassifier().predict(singleSample)
                        predProb=DTreeClassifier().predict_proba(singleSample)
                        if prediction == 0:
                            st.success("Depression Level: LOW")
                            predProbabilityScore = {"Low": predProb[0][0] * 100, "High": predProb[0][1] * 100, "Neutral": predProb[0][2] * 100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(predProbabilityScore)
                            st.text("")
                            diagnosis_for_low()
                        elif prediction == 1:
                            st.warning("Depression Level: HIGH")
                            predProbabilityScore = {"Low": predProb[0][0] * 100, "High": predProb[0][1] * 100,"Neutral": predProb[0][2] * 100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(predProbabilityScore)
                            st.text("")
                            diagnosis_for_high()
                        else:
                            st.success("Depression Level: NEUTRAL")
                            predProbabilityScore = {"Low": predProb[0][0] * 100, "High": predProb[0][1] * 100,"Neutral": predProb[0][2] * 100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(predProbabilityScore)
                            st.text("")
                            diagnosis_for_neutral()

                    elif modelChoice == "Voting Cl":
                        prediction=VotClassifier().predict(singleSample)
                        if prediction == 0:
                            st.success("Depression Level: LOW")
                            st.text("")
                            diagnosis_for_low()
                        elif prediction == 1:
                            st.warning("Depression Level: HIGH")
                            st.text("")
                            diagnosis_for_high()
                        else:
                            st.success("Depression Level: NEUTRAL")
                            st.text("")
                            diagnosis_for_neutral()

                    else:
                        prediction = RForestClassifier().predict(singleSample)
                        pred_prob = RForestClassifier().predict_proba(singleSample)
                        if prediction == 0:
                            st.success("Depression Level: LOW")
                            pred_probability_score={"Low":pred_prob[0][0]*100,"High":pred_prob[0][1]*100,"Neutral":pred_prob[0][2]*100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(pred_probability_score)
                            st.text("")
                            diagnosis_for_low()
                        elif prediction == 1:
                            st.warning("Depression Level: HIGH")
                            predProbabilityScore = {"Low": pred_prob[0][0] * 100, "High": pred_prob[0][1] * 100,"Neutral": pred_prob[0][2] * 100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(predProbabilityScore)
                            st.text("")
                            diagnosis_for_high()
                        else:
                            st.success("Depression Level: NEUTRAL")
                            predProbabilityScore = {"Low": pred_prob[0][0] * 100, "High": pred_prob[0][1] * 100,"Neutral": pred_prob[0][2] * 100}
                            st.subheader("Prediction Probability Score using {}".format(modelChoice))
                            st.json(predProbabilityScore)
                            st.text("")
                            diagnosis_for_neutral()
                except tw.TweepError as e:
                    st.warning("Oops. {} is either a private user or that page does not exist. Enter a different username...".format(username))
                    # st.text("Enter a different username...")
                    # print("Tweepy Error: {}".format(e))


if __name__ == '__main__':
    main()