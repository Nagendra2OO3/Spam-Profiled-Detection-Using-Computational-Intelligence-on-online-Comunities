import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle

#df = pd.read_csv('SpamEmails/emails.csv')
#df.drop_duplicates(inplace = True)

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#df['text'].head().apply(process_text)
#cv = CountVectorizer(analyzer=process_text,stop_words = "english", lowercase = True)
#messages_bow = cv.fit_transform(df['text'])
#X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)
#classifier = MultinomialNB()
#classifier.fit(X_train, y_train)
#cpickle.dump(classifier, open('naiveBayes.pkl', 'wb'))
#print('Predicted value: ',classifier.predict(X_test))
#print('done')
#cpickle.dump(cv.vocabulary_,open("feature.pkl","wb"))
#print('done')
classifier = cpickle.load(open('naiveBayes.pkl', 'rb'))
#msg = "Subject: unbelievable new homes made easy  im wanting to show you this  homeowner  you have been pre - approved for a $ 454 , 169 home loan at a 3 . 72 fixed rate .  this offer is being extended to you unconditionally and your credit is in no way a factor .  to take advantage of this limited time opportunity  all we ask is that you visit our website and complete  the 1 minute post approval form  look foward to hearing from you ,  dorcas pittman"
#msg = "Subject: tuesday morning meeting first thing ? ? ?  vince :  i am sorry i couldnt connect with you last week . how would your tuesday  morning first thing , say 800 or 830 am be to get together to discuss the  demo proposal and other issues ? i can come by your office very conveniently  tinvestors , who get  in before the institutional investors , could make a fortune . . . . more  read the full report click here !  investors are making huge gains as the world bets online .  could gaming transactions inc . ( gtts ) be the next stock to rock ?  recently ,  we read an article in the economist that highlighted online gaming and  how it has become a socially acceptable form of entertainment . over the  next few days , as we thought about what sort of impact this trend could  have , we started to notice that online gambling was being discussed all  over the media - in newspapers , online and on television . it became  obvious to us that more and more people were jumping on the internet to  bet on games . we also came across some staggering statistics .  merrill lynch , for example , has predicted that gambling has the  potential to account for a full 1 % of global economic activity ! another  source , ecommercetimes . com , recently reported that the scope of this  business is so enormous that some have even claimed that it is the  single most important factor in the growth of e - commerce . the online  gaming industry , in other words , appears to be booming , and it may be  an ideal time to is time to invest .  we decided to  find out who the key players were in the business . after speaking with  a number of industry insiders , the trail led us to an emerging ,  publicly traded company called gaming transactions inc . ( ggts ) . after a  close look at ggts , we decided that this company could produce huge returns on investment in the upcoming months .  although  ggts is a new company , it has some surprisingly experienced players at  the helm an uncommon thing to find in an industry only ten years old .  the company has come out with online versions of the addictive game keno .  and its management team was smart enough to secure the rights to the  keno . com website , which , if youre marketing keno , is as good as it  gets . keno has the widest spread for the house of any mainstream  gambling game . ggts is also about to launch a suite of other online  gambling games , including poker and sports - book betting . once it offers  these popular games , and given that it already has keno . com online , the  company could bring in great revenues , attracting a lot of attention in  the investment community and driving up its stock price .  after a successful north american launch ,  gaming transactions inc . ( gtts ) has translated its games into chinese  and is about to hit asia . this initiative looks like a wise business  decision : many analysis anticipate that china will be the biggest  source of online gambling revenue by 2007 , so the company could be poised for massive expansion in terms of both profits and global reach .  what  does all this tell us ? a brand new company , a popular , well - known game ,  one of the biggest spreads for the house , a growing market , experienced  management , and a stock price that is trading under a dollar it adds up to the potential for huge gains for early investors . if youre interested in more information on the market and gaming transactions inc . , click here to read a free ten - page report . . .  to join market movers mailings http : / / ggtsstock . com to find out more .  first source data inc .  4535 west sahara ave # 217  las vegas nevada 89102  disclosure and disclaimer  investment news indepth reports ( hereinafter inir ) , operated by first  source data , inc . ( hereinafter fsd ) , is a business news publication  of regular and general circulation . this issue is not a research  report , does not purport to provide an analysis of any companys  financial position , and is not in any way to be construed as an offer  or solicitation to buy or sell any security . gaming transactions inc .  ( hereinafter ggts ) is the featured company . fsd managed the  publishing and distribution of this publication . the information  contained herein is being republished in reliance on statements made by  ggts management , and publicly disseminated information issued by third  parties regarding ggts and the online gaming industry , which are  presumed to be reliable , but neither fsd nor its editors , employees , or  agents accept any responsibility for the accuracy of such statements or  information , or the contents herein which are derived therefrom .  readers should independently verify all statements made in this  advertisement .  fsd has received compensation for the production and distribution of  this newsletter . the compensation received is in the amount of one  hundred and twenty eight thousand dollars and was received from  accelerated capital limited ( hereinafter acl ) for this advertising  effort . acl is a shareholder of ggts . because fsd received compensation  for its services , there is an inherent conflict of interest in the  statements and opinions contained in this newsletter and such  statements and opinions cannot be considered independent .  internet - based companies , and those involving online gaming in  particular , are almost always very high risk investments , and investors  should be aware that they could potentially lose any investment made in  such companies in its entirety . we strongly encourage readers to  undertake their own due diligence to decide the best course of action  in connection with any investment decision that they might make . any  investment should be made only after consulting with a qualified  investment advisor .  media matrix 7025 county rd . 46 a dtel 071 # 349 lake mary , fl 32746 this e - mail message is an advertisement and / or solicitation ."
msg = "subject:	Core Job to Software Development - Digvijay's Transition Story"
#msg = process_text(msg)
#print(msg)
cvv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("feature.pkl", "rb")))
cv1 = CountVectorizer(vocabulary=cvv.get_feature_names(),stop_words = "english", lowercase = True)
test = cv1.fit_transform([msg])
print('Predicted value: ',classifier.predict(test))


