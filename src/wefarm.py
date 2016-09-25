import pandas as pd
import numpy as np
import nltk
import re
import gensim
#import pyLDAvis.gensim
import warnings
import csv
from datetime import datetime as dt
from sklearn.externals import joblib
import operator
import random

from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use @default decorator instead.*')

class Message(object):
    def __init__(self, thread_id, date_time, message_id, user_id, language, msg_type, msg_body):
        self.thread_id = thread_id
        self.date_time = date_time
        self.message_id = message_id
        self.user_id = user_id
        self.language = language
        self.msg_type = msg_type
        self.msg_body = msg_body

def loadData(f):
    users = {}
    messages = []
    with open(f, "r") as data:
        # "thread_id","date_time","message_id","user_id","language","type","body"
        # datetime format: 2015-02-09 14:27:05
        reader = csv.DictReader(data)
        for row in reader:
            user_id = row["user_id"]
            message = Message(row["thread_id"],
                              dt.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S"),
                              row["message_id"],
                              row["user_id"],
                              row["language"],
                              row["type"],
                              row["body"])
            if user_id not in users:
                users[user_id] = []
            users[user_id].append(message)
            messages.append(message)
    return users, messages

def getMessageGroups(messages, grouper):
    groupedMessages = {}
    for message in messages:
        messageGroup = getattr(message, grouper)
        if messageGroup not in groupedMessages:
            groupedMessages[messageGroup] = []
        groupedMessages[messageGroup].append(message)
    return groupedMessages
        

def cleaner(row):
    '''Function to clean the text data and prep for further analysis'''
    stops = set(stopwords.words("english"))     # Creating a set of Stopwords
    p_stemmer = PorterStemmer()                 # Creating the stemmer model
    text = row['body'].lower()                   # Converts to lower case
    text = re.sub("[^a-zA-Z]"," ",text)          # Removes punctuation
    text = text.split()                          # Splits the data into individual words 
    text = [w for w in text if not w in stops]   # Removes stopwords
    text = [p_stemmer.stem(i) for i in text]     # Stemming (reducing words to their root)
    return text


def messages_vectorizer(messages):
    '''Function to take a message object and convert it to a list of terms'''
    stops = set(stopwords.words("english"))     # Creating a set of Stopwords
    p_stemmer = PorterStemmer()                 # Creating the stemmer model
    text = ''
    for m in messages:
        text = text + ' ' + m.msg_body.lower()          # Converts to lower case
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.split()                          # Splits the data into individual words 
    text = [w for w in text if not w in stops]   # Removes stopwords
    text = [p_stemmer.stem(i) for i in text]     # Stemming (reducing words to their root)
    return text

def model(data, state, num_topics):
    data_dict = corpora.Dictionary(data)                       # Creates an id <-> term dictionary
    data_corpus = [data_dict.doc2bow(text) for text in data]     # convert tokenized documents into a document-term matrix
    data_model = gensim.models.ldamodel.LdaModel(data_corpus, 
                                                   num_topics=num_topics, 
                                                   id2word = data_dict,
                                                   passes=20,
                                                   random_state=state)        #  generate LDA model

    #data_vis = pyLDAvis.gensim.prepare(data_model, data_corpus, data_dict)        # Visualise LDA Model
    #pyLDAvis.save_html(data=data_vis,
    #                    fileobj=out + 'Data_vis.html')
    #data_vis
    return data_model, data_corpus, data_dict

def generate_topic_desc_csv(the_model, word_dict):
    with open('../Outputs/topics.csv', 'wb') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',',quotechar='"')
        for t in range(10):
            topic_words= ['T'+str(t)]    
            for k,v in the_model.get_topic_terms(t,topn=25):
                word = word_dict[k]
                topic_words.append(word)        
            mywriter.writerow(topic_words)

def generate_users_desc_csv(grouped_user_ids,the_model,the_corpus,n_topics=10):
    with open('../Outputs/users.csv', 'wb') as csvfile:    
        mywriter = csv.writer(csvfile, delimiter=',',quotechar='"')
        
        for i in range(len(grouped_user_ids)):
            user_id= grouped_user_ids[i]
            scores= the_model.get_document_topics(the_corpus[i])
            scores_dict={}
            for score in scores:
                scores_dict[score[0]]=score[1]
            scores_arr=[]
            for j in range(n_topics):
                if j in scores_dict.keys():
                    scores_arr.append(scores_dict[j])               
                else:
                    scores_arr.append(0.0)
            output_arr=[]
            output_arr.append(user_id)
            for s in scores_arr:
                output_arr.append(str(s))            
            mywriter.writerow(output_arr)

def generate_message_predictions_csv(messages,):
    with open('../Outputs/message_predictions.csv', 'wb') as csvfile:    
        mywriter = csv.writer(csvfile, delimiter=',',quotechar='"')
        
        for message in messages:
            scores = topic_scores_for_message(message)
            
            scores_dict={}
            for score in scores:
                scores_dict[score[0]]=score[1]
            scores_arr=[]
            for j in range(10):
                if j in scores_dict.keys():
                    scores_arr.append(scores_dict[j])
                else:
                    scores_arr.append(0.0)
            output_arr=[message.thread_id, dt.strftime(message.date_time, "%Y-%m-%d %H:%M:%S"), message.message_id, message.user_id, message.language, message.msg_type, message.msg_body]
            for s in scores_arr:
                output_arr.append(str(s))
            mywriter.writerow(output_arr)

def top_users_for_topic(topics_to_users, t, max_val=50):
    scores = topics_to_users[t]
    sorted_scores= sorted(scores.items(), key=operator.itemgetter(1))
    sorted_scores.reverse()
    return sorted_scores[:max_val]
    
def topic_scores_for_message(message, word_dict, model):
    message_to_predict = messages_vectorizer([message])
    message_to_predict = word_dict.doc2bow(message_to_predict)
    scores = model.get_document_topics(message_to_predict)
    return scores

def top_users_for_message(message, user_scores, word_dict, model):
    message_scores = topic_scores_for_message(message, word_dict, model)
    topic = max(message_scores, key=lambda item:item[1])[0]
    top_users = [x for (x, y) in top_users_for_topic(user_scores, topic)]
    return topic, top_users

def questions_to_recommended_users(messages,grouped_messages,topics_to_users, word_dict, model):
    for message in messages[:100]:
        if(message.msg_type=='Q'):
            topic, users = top_users_for_message(message, topics_to_users, word_dict, model)
            print 'Question: '
            print message.msg_body, topic
            
            print 'recommended users: '
            shuffled = random.shuffle(users)
            for u in users[:10]:             
                users_answers = filter (lambda x: x.msg_type=='A',grouped_messages[u])
                if(len(users_answers)==0): 
                    continue
                print 'User: ', u
                for a in users_answers: 
                    print a.msg_body
            print

def topics_to_users_fn(grouped_user_ids,the_model,the_corpus,n_topics=10):
    topics_to_users={}
    for j in range(n_topics):
        topics_to_users[j]={}
    for i in range(len(grouped_user_ids)):
        user_id= grouped_user_ids[i]
        scores= the_model.get_document_topics(the_corpus[i])
        scores_dict={}
        for score in scores:
            scores_dict[score[0]]=score[1]
        scores_arr=[]
        for j in range(n_topics):
            if j in scores_dict.keys():
                topics_to_users[j][user_id]=scores_dict[j]                
            else:
                topics_to_users[j][user_id]=0.0
    return topics_to_users

def test_prediction():
    # processing a message to make a prediction
    users_topic_model = joblib.load('../Outputs/users_topic_model.pkl' ) 

    the_model = users_topic_model[0]
    the_corpus = users_topic_model[1]
    word_dict = users_topic_model[2]

    print(the_corpus[0])
    test_message = messages[0]
    print(test_message.msg_body.lower())
    the_model.get_document_topics(the_corpus[0])
    test_message = messages_vectorizer([test_message])
    print(test_message)
    test_vectorized = word_dict.doc2bow(test_message)

    the_model.get_document_topics(test_vectorized)

    for k, v in the_model.get_topic_terms(4, topn=25):
        word = word_dict[k]
        print(word)


def main():
    # set up parameters
    # set random seed
    random_seed = 135
    state = np.random.RandomState(random_seed)

    path = '../data/'    # Data Directory 
    out = '../Outputs/'  # Output Directory
    msg = 'messages.csv'                       # Input Dataset

    sample_size = 5000
    num_topics = 10
    # Load data
    print("Loading data")
    df_msg_in = pd.read_csv(path + msg)
    df_msg_en = df_msg_in[(df_msg_in['language'] == 'EN')] 
    # Group by users
    users, messages = loadData(path + msg)
    grouped_messages =  getMessageGroups(messages, 'user_id')

    #preparing data to build the topic model
    print("Preparing data")
    grouped_user_data =[]

    min_messages = 3
    max_messages = 300

    grouped_messages_by_user = grouped_messages
    for k in grouped_messages.keys():
        v = grouped_messages[k]
        if (len(v)<max_messages and len(v)>min_messages):
            grouped_user_data.append(messages_vectorizer(v))

    grouped_user_ids=[]
    for k in grouped_messages.keys():
        v = grouped_messages[k]
        if (len(v)<max_messages and len(v)>min_messages):
            grouped_user_ids.append(k)

    # build the topic model and save it
    # print("Building the model")
    # users_topic_model = model(grouped_user_data, state, num_topics)
    # the_model = users_topic_model[0]
    # the_corpus = users_topic_model[1]
    # word_dict = users_topic_model[2]
    # joblib.dump(users_topic_model, '../Outputs/users_topic_model.pkl' ) 

    # load the model that has just been built
    print("loading the model")
    users_topic_model = joblib.load('../Outputs/users_topic_model.pkl')
    the_model = users_topic_model[0]
    the_corpus = users_topic_model[1]
    word_dict = users_topic_model[2]

    print("Write topic csv")
    generate_topic_desc_csv(the_model, word_dict)

    topics_to_users = topics_to_users_fn(grouped_user_ids,the_model,the_corpus)

    print("Generate users for questions")
    questions_to_recommended_users(messages,grouped_messages,topics_to_users, word_dict, the_model)
    


if __name__ == "__main__": main()



