'''
To fecth email from your email address first you need to goto your email and enable imap
access and allow access through less secure source then only you will be able to access
your email.
'''
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter.font as tkFont
import imaplib
from tkinter import messagebox
from collections import Counter
from scipy.sparse import csr_matrix
import math
import functools
import operator
import seaborn as sns
import email

vocab = {}
model = 0


#to find the vocabulary
def custum_fit(data):
    global vocab
    unique_words = set()

    for each_sentence in data:
        for each_word in each_sentence:
            unique_words.add(each_word)
    
    for index, word in enumerate(sorted(list(unique_words))):
        vocab[word]=index
        
vocabflag = 0
'''
This function is used to find the frequency of each words in the email
Initially we use custum_fit function for training data to find all possible words(vocabulary)
Then for testing data we check for each word in vocab how many times that word has 
occured in the testing email
Then we return a csr matrix of that data
'''
def custum_transform(data):
    global vocab
    if vocabflag==0:
        custum_fit(data)
    row, col, val = [],[],[]
    for idx, sentence in enumerate(data):
        count_word = dict(Counter(sentence))
        for word, count in count_word.items():
            col_index = vocab.get(word)
            if col_index is not None:
                if col_index>=0:
                    row.append(idx)
                    col.append(col_index)
                    val.append(count)
                    
    return csr_matrix((val,(row,col)),shape=(len(data),len(vocab)))

#Function used to find the accuracy of our model
def accuracy(actual,prediction):
    actual = list(actual)
    if len(actual) != len(prediction):
        print('invalid input')
        return
    total = len(actual)
    correct = 0
    for i in range(total):
        if actual[i]==prediction[i]:
            correct = correct + 1
    return correct/total

#It is used to find the confusion matrix
def confusion(actual,prediction):
    actual = list(actual)
    if len(actual) != len(prediction):
        print('invalid input')
        return
    mat = np.zeros((2,2))
    for i in range(len(actual)):
        mat[actual[i],prediction[i]] =  mat[actual[i],prediction[i]] + 1
    return mat


class MultiNB:
    def __init__(self,alpha=1):
        #alpha is smoothing factor
        self.alpha = alpha
    
    def _prior(self): 
        """
        Calculates prior for each unique class in y. P(y)
        """
        P = np.zeros((self.n_classes_))
        _, self.dist = np.unique(self.y,return_counts=True)
        for i in range(self.classes_.shape[0]):
            P[i] = self.dist[i] / self.n_samples
        return P
            
    def fit(self, X, y):
        '''
        P(xi∣y)=Nyi+α/Ny+αn this is the formula we used
        It will basically find the probability of each word in the email being in 
        each of the class. 
        '''
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_priors_ = self._prior()
        
        self.docidx, self.wordidx = X.nonzero()
        count = X.data
        classidx = []
        for idx in self.docidx:
            classidx.append(self.y.iloc[idx])
        df = pd.DataFrame()
        df['docidx'] = np.array(self.docidx)
        df['wordidx'] = np.array(self.wordidx)
        df['count'] = np.array(count)
        df['classidx'] = np.array(classidx)
        #print(df.info)
        global vocab
        self.N_yi = df.groupby(['classidx','wordidx'])
        #for key,item in self.N_yi:
           # print(self.N_yi.get_group(key))
        self.N_y = df.groupby('classidx')
        #self.N_yi = (self.N_yi['count'].sum()
       
        self.Pr =  (self.N_yi['count'].sum() + self.alpha) / (self.N_y['count'].sum() +(self.alpha*len(vocab)))    
        #Unstack series
        self.Pr = self.Pr.unstack()
        
        #Replace NaN or columns with 0 as word count with a/(count+|V|+1)
        for c in range(0,2):
            self.Pr.loc[c,:] = self.Pr.loc[c,:].fillna(self.alpha/(self.N_y['count'].sum()[c] +  len(vocab) ))
        self.Pr_dict = self.Pr.to_dict()
        #for key, item in self.pb_ij:
            #print(pb_ij.get_group(key), "\n\n")
            #print(key,item)
    '''
    For a give email x it will find the probability of it belonging to the class h.
    It will basically return the product of prob. of each word in x belonging to class h
    which was previously calculated in fit function.
    '''
    def _likelyhood(self, x, h):
        tmp = []
        X = x.toarray()[0]
        indices = X.nonzero()[0]
        for i in indices:
            if i in self.Pr_dict.keys():
                tmp.append(float(math.pow(self.Pr_dict[i][h],X[i])))
        return np.exp(np.log(tmp).sum())
    
    '''
    It will find to which class the given email belongs to.
    '''
    def predict(self, X):
        samples, features = X.shape
        self.predict_proba = np.zeros((samples,self.n_classes_))
        
        for i in range(X.shape[0]):
           # joint_likelyhood = np.zeros((self.n_classes_))
            for h in range(self.n_classes_):
                self.predict_proba[i,h]  = self.class_priors_[h] * self._likelyhood(X[i],h)
                #print(self.predict_proba[i,h])
                # P(y) P(X|y) 
        #print(self.predict_proba)
        indices = np.argmax(self.predict_proba,axis=1)
        return self.classes_[indices]
    
allemails = []
'''
It will iterate through each email in the dataframe which was given as input 
then apply clean text function to each email and append the result to a list.
'''
def find_mails(data):
    global allemails
    data = clean_text(data)
    allemails.append(data)

#Used to get the email from our gmail account
imap_url = 'imap.gmail.com'
con = None

#Load the data
df = pd.read_csv("completeSpamAssassin.csv");
print(df.head)

#drop the unwanted column
df.drop(['Unnamed: 0'],axis=1,inplace=True)

target = 'Label'
feature = 'Body'

#Loading the stopwords in english
stopwordslist = []
with open('stopwords.txt','r') as file:
    for row in file:
        stopwordslist.append(row.split('\n')[0])
        


punctuation = string.punctuation
punctuations = []
for char in punctuation:
    punctuations.append(char)

'''
This function is used to remove all the \n, numbers, punctuations and stopwords.
'''
def clean_text(text):
  text = str(text)
  #Remove backslash n
  #text = [text.split('\\n')]
  #text = ''.join(text[0])
  text.replace('\n','')
  #print('After joining;\n',text)
  
  #Remove punctuations
  newtext = ''
  for char in text:
      if char not in punctuations:
          newtext += char
  #print("After removing punctuations:\n",newtext)
  
  #Removing stopwords
  words = newtext.split()
  corpus = []
  for word in words:
      if word.lower() not in stopwordslist:
          corpus.append(word.lower())
  #print('After removing the stopwords:\n',corpus)

  #Remove numbers
  for words in corpus:
      word = ''
      for char in words:
          if char.isalpha():
              word +=char
      corpus[corpus.index(words)] = word
      
  #Remove null from corpus
  while("" in corpus):
      corpus.remove("")
  #print(corpus)
  
  return corpus

#print(df[feature],type(df[feature]))

#Importing CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=clean_text)

#Importing Multinomial Byes classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
me = MultiNB()

#function to get the body of the email
def get_body(msg):
    if msg.is_multipart():
        return get_body(msg.get_payload(0))
    else:
        return msg.get_payload()
'''
It will search your email account to find the email that matches the requirement and 
return their id.
'''    
def search(key,value,con):
    result, data = con.search(None,key,value)
    return data
#To fetch the email from its id
def get_email(result_bytes):
    msgs = []
    for num in result_bytes[0].split():
        typ, data = con.fetch(num,'(RFC822)')
        msgs.append(data)
    return msgs

#Used to remove html tags from the email
def cleanemail(email):
    i = 0
    length = len(email)
    cleantext1 = ''
    while i<length:
        if email[i]=='<':
            while email[i]!='>':
                i+=1               
        else:
            cleantext1+=email[i]
        i+=1
    return cleantext1

#gui starts here
#Creating the parent window
root = Tk()

f = Frame(root,height=520,width=460)
f.pack()

#giving title
root.title('Spam Detection')

#specifying geometry
root.geometry('460x520')

root.iconbitmap('icon.ico')

label1 = Label(f,text='Spam Email Detection',font=tkFont.Font(size=20))
label1.place(x=90,y=60)

name1 = Label(f,text='Ganesh Wagle')
name2 = Label(f,text='Lancy Joy Lobo')

usn1 = Label(f,text='4nm18cs058')
usn2 = Label(f,text='4nm18cs079')

name1.place(x=90,y=120)
name2.place(x=90,y=160)

usn1.place(x=290,y=120)
usn2.place(x=290,y=160)


def email_entry():
    global f
    f.destroy()
    f = Frame(root,height=520,width=460)
    f.pack()
    
    usr_label = Label(f,text='Enter the email id')
    usr_label.place(x=6,y=6)
    
    usr = Text(f,bg='light gray')
    usr.place(x=200,y=6,height=20,width=200)
    
    pas_label = Label(f,text = 'Enter the password')
    pas_label.place(x=6,y=30)
    
    pas = Entry(f,show='*',bg='light gray')
    pas.place(x=200,y=30,height=20,width=200)
    
    key_label = Label(f,text='Enter the search key')
    key_label.place(x=6,y=54)
    
    key = Text(f,bg='light gray')
    key.place(x=200,y=54,height=20,width=200)
    
    search_label = Label(f,text='Enter the search value')
    search_label.place(x=6,y=78)
    
    search_value = Text(f,bg='light gray')
    search_value.place(x=200,y=78,height=20,width=200)
    
    res = Text(f,bg='light gray')
    res.place(x=6,y=160,height=300,width=446)
    
    
    def start():
        usrname = usr.get('1.0','end')
        password = pas.get()
        global con
        con = imaplib.IMAP4_SSL(imap_url)
       # print(usrname[:-1],password)
        try:
            
            con.login(usrname[:-1],password)
            con.select('"[Gmail]/All Mail"')
            print('Connection established')
        except:
            messagebox.showerror("error",'Check your credentials again')
            #con.close()
           # con.logout()
            print('Check your crentials again')
            return
        key_value = key.get('1.0','end')
        value_search = search_value.get('1.0','end')
        try:
            resul = search(key_value[:-1],value_search[:-1],con)
            global classifier,me,model
            final_result = 'Emails that which satisfy your query are '
            for num in resul[0].split():
                final_result += str(num)
            print(final_result)
            msgs = get_email(resul)
           #print(msgs)
            for msg in msgs:
                mail = '\nSubject:'+email.message_from_bytes(msg[0][1])['Subject']
                mail = mail + '\n' + cleanemail(get_body(email.message_from_bytes(msg[0][1])))
                #print(mail)
                final_result += mail+'\n'
                #df1 = pd.DataFrame()
                #df1['email'] = pd.Series(mail,dtype='object')
                #mail = vectorizer.transform(df1['email'])
                mail = clean_text(mail)
                mail = custum_transform([mail])
                check = ''
                prediction = None
                if model==1:
                    print(me.predict(mail))
                    prediction = me.predict(mail)[0]
                else:
                    prediction = classifier.predict(mail)[0]
                    
                if prediction==1:
                    check = "a Spam!!!\n\n"
                else:
                    check = "Not a Spam.\n\n"
                final_result += "\nThis email is "+check
            res.delete('1.0','end')
            res.insert(INSERT,final_result)
        except:
           messagebox.showerror("error",'Check your Search Inputs')
           con.close()
           con.logout()
           print('Check search value')
    
    search_but  = Button(f,text='Search and predict',command=start)
    search_but.place(x=145,y=120,height=20,width=150)
    
    paste_email = Button(f,text='Paste email',command=predict_email)
    paste_email.place(x=170,y=470,height=30,width=100)
    
    
        

def predict_email():
    global f,model
    f.destroy()
    f = Frame(root,height=520,width=460)
    f.pack()
    label1 = Label(f,text='Paste your email here:',font=tkFont.Font(size=10))
    label1.place(x=6,y=6)
    messageWindow = Text(f,bg='light gray')
    messageWindow.place(x=6,y=40,height=380,width=446)
    label2 = Label(f,text="",font=tkFont.Font(size=13))
    
    def predict():
        nonlocal label2
        email1 = messageWindow.get('1.0','end')
        #df1 = pd.DataFrame()
        global classifier,me
        
        email1 = clean_text(email1)
        email1 = custum_transform([email1])
        
        
        prediction = None
        if model==1:
            print(me.predict(email1))
            prediction = me.predict(email1)[0]
        else:
            prediction = classifier.predict(email1)[0]
            
        if prediction==1:
            label2.config(text ="Entered email is a Spam!!!.",fg='red' )
        else:
            label2.config(text ="Entered email is Not a Spam.",fg='green' )
            
       
        label2.place(x=120,y=430)
        
    predict_button = Button(f,text='Predict',command=predict)
    predict_button.place(x=6,y=430,height=30,width=100)
    
    goto_email = Button(f,text='Goto email',command=email_entry)
    goto_email.place(x=6,y=470,height=30,width=100)

         
def train_ours():
    global f,root,model
    model = 1
    f.destroy()
    f = Frame(root,height=520,width=460)
    f.pack()
    root.geometry('460x520')
    #Convert a collection of tokens to a matrix of tokens
    #global vectorizer
    #message = vectorizer.fit_transform(df[feature])
    df['Body'].apply(find_mails)
    message = custum_transform(allemails)
    global vocabflag
    vocabflag = 1
    #Split the data into 80% training and 20% testing
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(message,df[target],test_size=0.20,random_state=0)
    print(x_train.shape,type(x_train))
    
    global me
    me.fit(x_train,y_train)
    
    #Evaluate the model on the training data set
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    pred = me.predict(x_train)
    
    #On testing data
    pred1 = me.predict(x_test)
    
    con1 = confusion(y_train,pred)
    con2 = confusion(y_test,pred1)
    
    print('\nOn training data:\nAccuracy:'+str(accuracy(y_train, pred)))
    msg1 = 'On training data:\nAccuracy:'+str(accuracy(y_train, pred))
    msg2 = 'Confustion Matrix:\n'+str(con1)
    print('On testing data:\nAccuracy:'+str(accuracy(y_test, pred1)))
    msg3 = '\n\nOn testing data:\nAccuracy:'+str(accuracy(y_test, pred1))
    msg4 = 'Confustion Matrix:\n'+str(con2)
    
    final_message=msg1+'\n'+msg2+msg3+'\n'+msg4
    #create the message window
    messageWindow = Text(f,bg='light gray')
    messageWindow.place(x=6,y=6,height=190,width=446)
    messageWindow.insert(INSERT,final_message)
    
    fig = Figure()
    a = fig.add_subplot(121)
    a.set_title('on training data')
    canvas = FigureCanvasTkAgg(fig, master=f)
    fig.patch.set_facecolor((.8242,.8242,.8242))
    sns.set(font_scale=.75) # for label size
    sns.heatmap(con1, annot=True, annot_kws={"size": 16},ax=a,fmt='g') 
    a = fig.add_subplot(122)
    a.set_title('on testing data')
    canvas = FigureCanvasTkAgg(fig, master=f)
    sns.set(font_scale=.75) # for label size
    sns.heatmap(con2, annot=True, annot_kws={"size": 16},ax=a,fmt='g') 
    fig.tight_layout()
    canvas.get_tk_widget().place(x=6,y=202,height=230,width=448)
    canvas.draw()
    
    but = Button(root,text='Use your Email',command=email_entry)
    but.place(x=16,y=460,height=30,width=100)
    
         
    predict = Button(root,text='Paste the Email',command=predict_email)
    predict.place(x=340,y=460,height=30,width=100)
    
         
def train_builtin():
    global f,root
    f.destroy()
    f = Frame(root,height=520,width=460)
    f.pack()
    root.geometry('460x520')
    #Convert a collection of tokens to a matrix of tokens
    #global vectorizer
    #message = vectorizer.fit_transform(df[feature])
    df['Body'].apply(find_mails)
    message = custum_transform(allemails)
    global vocabflag
    vocabflag = 1
    #Split the data into 80% training and 20% testing
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(message,df[target],test_size=0.20,random_state=0)
    print(x_train.shape,type(x_train))
    
    #Create and train the Naive Bayes Classifier
    global classifier
    classifier = classifier.fit(x_train,y_train)

    #Evaluate the model on the training data set
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    pred = classifier.predict(x_train)
    
    #On testing data
    pred1 = classifier.predict(x_test)
    
    con1 = confusion(y_train,pred)
    con2 = confusion(y_test,pred1)
    
    print('\nOn training data:\nAccuracy:'+str(accuracy(y_train, pred)))
    msg1 = 'On training data:\nAccuracy:'+str(accuracy(y_train, pred))
    msg2 = 'Confustion Matrix:\n'+str(con1)
    print('On testing data:\nAccuracy:'+str(accuracy(y_test, pred1)))
    msg3 = '\n\nOn testing data:\nAccuracy:'+str(accuracy(y_test, pred1))
    msg4 = 'Confustion Matrix:\n'+str(con2)
    
    final_message=msg1+'\n'+msg2+msg3+'\n'+msg4
    #create the message window
    messageWindow = Text(f,bg='light gray')
    messageWindow.place(x=6,y=6,height=190,width=446)
    messageWindow.insert(INSERT,final_message)
       
    fig = Figure()
    a = fig.add_subplot(121)
    a.set_title('on training data')
    canvas = FigureCanvasTkAgg(fig, master=f)
    fig.patch.set_facecolor((.8242,.8242,.8242))
    sns.set(font_scale=.75) # for label size
    sns.heatmap(con1, annot=True, annot_kws={"size": 16},ax=a,fmt='g') 
    a = fig.add_subplot(122)
    a.set_title('on testing data')
    canvas = FigureCanvasTkAgg(fig, master=f)
    sns.set(font_scale=.75) # for label size
    sns.heatmap(con2, annot=True, annot_kws={"size": 16},ax=a,fmt='g') 
    fig.tight_layout()
    canvas.get_tk_widget().place(x=6,y=202,height=230,width=448)
    canvas.draw()
    
    but = Button(root,text='Use your Email',command=email_entry)
    but.place(x=16,y=460,height=30,width=100)
    
         
    predict = Button(root,text='Paste the Email',command=predict_email)
    predict.place(x=340,y=460,height=30,width=100)
    
def words():
    global f
    global root
    global allemails
    
    f.destroy()
    
    f = Frame(root,height=558,width=460)
    f.pack()
    
    df.loc[df['Label']==0]['Body'].apply(find_mails)
    allemails = functools.reduce(operator.iconcat, allemails, [])
    #print(allemails)
    count = Counter(allemails)
    words = [word[0] for word in count.most_common(20)]
    values = [value[1] for value in count.most_common(20)]
    fig = Figure(figsize=(6,6))
    fig.patch.set_facecolor((.8242,.8242,.8242))
    a = fig.add_subplot(211)
    a.bar(words,values)
    a.set_title('20 Most common words in ham',fontsize=13)
    a.set_xlabel('Words',fontsize=13)
    a.tick_params(axis='x', rotation=90)
    a.set_ylabel('Count',fontsize=13)
    a.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    #canvas = FigureCanvasTkAgg(fig, master=f)
    #canvas.get_tk_widget().place(x=6,y=6,height=390,width=446)
   # canvas.draw()
    
    
    allemails = []
    df.loc[df['Label']==1]['Body'].apply(find_mails)
    allemails = functools.reduce(operator.iconcat, allemails, [])
    
    count = Counter(allemails)
    words = [word[0] for word in count.most_common(21)]
    words.pop(3)
    values = [value[1] for value in count.most_common(21)]
    values.pop(3)
    
    #fig = Figure(figsize=(10,3))
    b = fig.add_subplot(212)
    b.bar(words,values)
    b.set_title('20 Most common words in spam',fontsize=13)
    b.set_xlabel('Words',fontsize=13)
    b.tick_params(axis='x', rotation=90)
    b.tick_params(axis='both', which='major', labelsize=12)
    b.set_ylabel('Count',fontsize=13)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=f)
    canvas.get_tk_widget().place(x=6,y=6,height=490,width=446)
    canvas.draw()

    allemails = []
    
    but1 = Button(root,text='Our Model',command=train_ours)
    but1.place(x=16,y=512,height=20,width=100)
    
    but2 = Button(root,text='Builtin Model',command=train_builtin)
    but2.place(x=340,y=512,height=20,width=100)
    root.geometry('460x558')
    
def preprocess():
    #destroy the old frame
    global f
    f.destroy()
    
    f = Frame(root,height=520,width=460)
    f.pack()
    #Removing the duplicates
    df.drop_duplicates(inplace=True)
    #Removing the null values
    df.dropna(inplace=True)
    msg = '''After processing the dataset:
Shape      : '''+str(df.shape)+'''
Duplicates : '''+str(df.duplicated().sum())+'''
NAN        :\n'''+str(df.isnull().sum())
    
    #creating text
    text1 = Text(f,bg='light gray')
    text1.place(x = 6, y = 16,height=128,width=446)
    
    text1.insert(INSERT,msg)
    
    #Plotting graph
    x_axis = ['ham','spam','null','duplicates']
    nonspam = df[target].value_counts()[0]
    spam = df[target].value_counts()[1] 
    dup = df.duplicated().sum()
    null = df.isnull().sum()[0]
    y_axis = [nonspam,spam,null,dup]
    fig = Figure()
    fig.patch.set_facecolor((.8242,.8242,.8242))
    a = fig.add_subplot(111)
    a.bar(x_axis,y_axis)
    a.set_title('After Preprocessing',fontsize=15)
    a.set_xlabel('Label',fontsize=13)
    a.set_ylabel('Count',fontsize=13)
    a.tick_params(axis='both', which='major', labelsize=12)
    canvas = FigureCanvasTkAgg(fig, master=f)
    canvas.get_tk_widget().place(x=6,y=165,height=290,width=446)
    canvas.draw()
    but = Button(root,text='Next',command=words)
    but.place(x=175,y=475,height=20,width=100)
    
    
def initial_screen():
    #destroy the old frame
    global f
    f.destroy()
    #Create a new frame
    f = Frame(root,height=520,width=460)
    f.pack()
    pre = Button(f,text='Preprocess',command=preprocess)
    pre.place(x=175,y=475,height=20,width=100)
    #creating text
    text1 = Text(f,bg='light gray')

    msg = '''Before processing the dataset:
Shape      : '''+str(df.shape)+'''
Duplicates : '''+str(df.duplicated().sum())+'''
NAN        :\n'''+str(df.isnull().sum())
         


    text1.insert(INSERT,msg)
    text1.place(x = 6, y = 16,height=128,width=446)    
    #Plotting graph
    x_axis = ['ham','spam','null','duplicates']
    nonspam = df[target].value_counts()[0]
    spam = df[target].value_counts()[1] 
    dup = df.duplicated().sum()
    null = df.isnull().sum()[0]
    y_axis = [nonspam,spam,null,dup]
    fig = Figure()
    fig.patch.set_facecolor((.8242,.8242,.8242))
    a = fig.add_subplot(111)
    a.bar(x_axis,y_axis)
    a.set_title('Initial dataset',fontsize=15)
    a.set_xlabel('Label',fontsize=13)
    a.set_ylabel('Count',fontsize=13)
    a.tick_params(axis='both', which='major', labelsize=12)
    canvas = FigureCanvasTkAgg(fig, master=f)
    canvas.get_tk_widget().place(x=6,y=165,height=290,width=446)
    canvas.draw()
    

button = Button(f,text='Next',command=initial_screen)
button.place(x=205,y=230,width=70)

root.mainloop()


