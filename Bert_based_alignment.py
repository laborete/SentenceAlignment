#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import time
import csv
#import dict_builder as db
import re
import pandas as pd
import requests
import numpy as np
#from __future__ import unicode_literals 
#from bosonnlp import BosonNLP
#nlp = BosonNLP('Yy7Tpb83.27824.Zdrfgp6JtKTf')
import os
import random
import pickle
import nltk
Porter = nltk.stem.PorterStemmer()


# In[2]:


from bert_serving.client import BertClient
from scipy import spatial
bc = BertClient()


# In[3]:


def readfile(filename,method='utf-8'):
    rawlist = []
    with open(filename,'r+', encoding=method) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            rawlist.append(row)
    return rawlist

def writefile(filename,inputlist,method='utf-8'):
    with open(filename,'a+', encoding=method) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in inputlist:
            writer.writerow(row)


# In[4]:


def regex(a):
    reg = []
    #a = re.sub('[a-zA-Z]','',a)
    a = re.sub('[,、：；！？，。－"\']','#',a)
    a = re.sub('[「」(){}:;!?]','',a)
    a = re.sub('\n','',a)
    #a = re.sub('[(),{}\u3000\*\|=-\[\]\n.]','',a)
    #a = re.sub(u"\\（.*?\\）|\\『.*?』|\\「.*?」|\\〔.*?〕|\\[.*?]|\\〈.*?〉", '', a)
    reg = a.split('#')
    return reg

def en_regex(a):
    reg = []
    #a = re.sub('[a-zA-Z]','',a)
    a = re.sub('[{}.:;!?：；！？，。－"\']','#',a)
    a = re.sub('[、「」()]','',a)
    a = re.sub(',',' ',a)
    a = re.sub('\n','',a)
    #a = re.sub('[(),{}\u3000\*\|=-\[\]\n.]','',a)
    #a = re.sub(u"\\（.*?\\）|\\『.*?』|\\「.*?」|\\〔.*?〕|\\[.*?]|\\〈.*?〉", '', a)
    reg = a.split('#')
    return reg

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# # 從這裡開始跑

# In[5]:


chcut = []
with (open('chcut.pickle', "rb")) as openfile:
    chcut = pickle.load(openfile)
encut = []
with (open('encut.pickle', "rb")) as openfile:
    encut = pickle.load(openfile)


# # 用不到

# ch_dic = {}
# for i in range(0,len(chcut)):
#     for j in range(0,len(chcut[i])):
#         for word in chcut[i][j]:
#             if ch_dic.get(word) == None:
#                 ch_dic[word] = [i]
#             else:
#                 if i not in ch_dic[word]: 
#                     ch_dic[word].append(i)

# tmp = []
# encut = []
# for i in range(0,len(alignment)):
#     for sent in alignment[i][1]:
#         tmp.append(sent.split())
#     encut.append(tmp)
#     tmp = []

# en_dic = {}
# for i in range(0,len(encut)):
#     for j in range(0,len(encut[i])):
#         for word in encut[i][j]:
#             #word = Porter.stem(word)
#             if en_dic.get(word) == None:
#                 en_dic[word] = [i]
#             else:
#                 if i not in en_dic[word]: 
#                     en_dic[word].append(i)

# bertvectors = {}
# for i in range(0,len(encut)):
#     if i%50==0:
#         print(i)
#     for j in range(0,len(encut[i])):
#         for word in encut[i][j]:
#             #word = Porter.stem(word)
#             #print(word)
#             if not word in bertvectors:
#                 if not word == '':
#                     bertvectors[word] = bc1.encode([word])
#                     
# for i in range(0,len(chcut)):
#     if i%50==0:
#         print(i)
#     for j in range(0,len(chcut[i])):
#         for word in chcut[i][j]:
#             try:
#                 tmp = adict[word]
#             except:
#                 tmp = [word]
#             for meaning in tmp:
#                 if meaning not in bertvectors:
#                     if not meaning == '':
#                         bertvectors[meaning] = bc.encode([meaning])

# #計算有幾個match的文章
# def co_occur(list1,list2):
#     return len(list(set(list1).intersection(list2)))
# 
# ch_dic_freq = []
# for key in ch_dic:
#     ch_dic_freq.append(len(ch_dic[key]))
# ch_dic_freq_mean = np.mean(ch_dic_freq)
# ch_dic_freq_sd = np.std(ch_dic_freq)

# def ch_weight(list1):
#     list2 = []
#     rlist = []
#     for word in list1:
#         list2.append([len(ch_dic[word]),word])
#     list2 = sorted(list2,reverse=True)
#     #print(list1)
#     for i in range(0,len(list1)):
#         for j in range(0,len(list2)):
#             if list1[i] == list2[j][1]:
#                 rlist.append(j+1)
#                 #print(list1[i],j+1)
#                 break
#     return(rlist)

# def s2s_calculator(cs,es,chdic=ch_dic,endic=en_dic):
#     s2s_rate = 0
#     w2s_rate = 0
#     rank = ch_weight(cs)
#     weightsum = sum(rank)
#     
#     for i in range(0,len(cs)):    
#         for en_word in es:
#             numerator = co_occur(chdic[cs[i]],endic[Porter.stem(en_word)])
#             denominator = len(chdic[cs[i]])+len(endic[Porter.stem(en_word)])-numerator
#             w2w_rate = float(numerator/denominator)
#             if w2w_rate>w2s_rate:
#                 #w2s_rate += float(w2w_rate/len(es))
#                 w2s_rate = w2w_rate
#         s2s_rate += float(w2s_rate*(rank[i]/weightsum))
#         #print(cs[i],rank[i]/weightsum,w2s_rate,s2s_rate)
#         w2s_rate = 0
#         w2w_rate = 0
#     return(s2s_rate)

# def another_s2s_calculator(cs,es,chdic=ch_dic,endic=en_dic):
#     s2s_rate = 0
#     w2s_rate = 0
#     counter = 0
#     for i in range(0,len(cs)):
#         if not len(chdic[cs[i]])>(ch_dic_freq_mean+5*ch_dic_freq_sd):
#             for en_word in es:
#                 numerator = co_occur(chdic[cs[i]],endic[Porter.stem(en_word)])
#                 denominator = len(chdic[cs[i]])+len(endic[Porter.stem(en_word)])-numerator
#                 w2w_rate = float(numerator/denominator)
#                 if w2w_rate>w2s_rate:
#                     #w2s_rate += float(w2w_rate/len(es))
#                     w2s_rate = w2w_rate
#             counter = counter+1
#             s2s_rate += float(w2s_rate)
#             #print(cs[i],counter,w2s_rate,(s2s_rate/counter))
#             w2s_rate = 0
#             w2w_rate = 0
#     if not counter==0:
#         return(s2s_rate/counter)
#     else:
#         return 0

# # 開始用的到

# In[6]:


with open('bertvectors1.pickle', 'rb') as file:
    tmpd1 = pickle.load(file)
with open('bertvectors2.pickle', 'rb') as file:
    tmpd2 = pickle.load(file)
with open('bertvectors3.pickle', 'rb') as file:
    tmpd3 = pickle.load(file)
with open('bertvectors4.pickle', 'rb') as file:
    tmpd4 = pickle.load(file)
bertvectors = {**tmpd1, **tmpd2, **tmpd3, **tmpd4}
len(bertvectors)


# In[7]:


with open('chinese_bertvectors.pickle', 'rb') as file:
    chinese_bertvectors = pickle.load(file)
len(chinese_bertvectors)


# In[8]:


adictsrc = readfile('zh_en_dict.csv')
adict = dict()
for i in range(0,len(adictsrc)):
    adict[adictsrc[i][0]] = adictsrc[i][1:]
len(adict)


# In[48]:


with open('trans_matrix.pickle', 'rb') as f:
    transmatrix = pickle.load(f)
with open('trans_matrix100.pickle', 'rb') as f:
    transmatrix100 = pickle.load(f)
with open('trans_matrixED.pickle', 'rb') as f:
    transmatrixED = pickle.load(f)


# # bert 相似度演算法

# In[13]:


def bert_calculator(cs,es):
    s2s_rate = 0
    for i in range(0,len(cs)):
        try:
            translations = adict[cs[i]]
        except:
            translations = [cs[i]]
        tmpcossim = 0
        #print('*',translations)
        for trans in translations:
            #print(trans)
            if not trans == '':
                try:
                    bert_ch = bertvectors[trans]
                except:
                    bert_ch = bc.encode([trans])
                for en_word in es:
                    try:
                        bert_en = bertvectors[en_word]
                    except:
                        bert_en = bc.encode([en_word])
                    teststat = 1-spatial.distance.cosine(bert_en,bert_ch)
                    #print(trans,'/',en_word,teststat)
                    if (teststat > tmpcossim):
                        tmpcossim = teststat
        s2s_rate = s2s_rate + tmpcossim
    if(len(cs))==0:
        return 0
    s2s_rate = float(s2s_rate/len(cs))
    return(s2s_rate)


# # bert 相似度演算法（利用轉置矩陣）

# In[60]:


def bert_matrix(cs,es):
    s2s_rate = 0
    for i in range(0,len(cs)):
        ch_word = cs[i]
        tmpcossim = 0
        if not ch_word == '':
            try:
                bert_ch = chinese_bertvectors[ch_word]
            except:
                bert_ch = bc.encode([ch_word])
            bert_ch = np.dot(bert_ch,transmatrix)
            #bert_ch = np.dot(bert_ch,transmatrixED)
            for en_word in es:
                try:
                    bert_en = bertvectors[en_word]
                except:
                    bert_en = bc.encode([en_word])
                teststat = 1-spatial.distance.cosine(bert_en,bert_ch)
                #print(teststat)
                if (teststat > tmpcossim):
                    tmpcossim = teststat
        s2s_rate = s2s_rate + tmpcossim
    if(len(cs))==0:
        return 0
    s2s_rate = float(s2s_rate/len(cs))
    return(s2s_rate)


# In[15]:


def choose_your_algo(c,e,method='bert'):
    rate = 0
    if method=='rank':
        rate = s2s_calculator(c,e)
    elif method=='average':
        rate = another_s2s_calculator(c,e,chdic,endic)
    elif method=='bert':
        rate = bert_calculator(c,e)
    elif method=='matrix':
        rate = bert_matrix(c,e)    
    return rate


# In[16]:


def word_output_combination(c,e,i,k):
    ctmp = ''
    etmp = ''
    for word in c:
        ctmp = ctmp + word
    for word in e:
        etmp = etmp+' '+word
    comber = str(i)
    for ktop in range(1,k):
        comber = comber +'+'+ str(i+ktop)
    return ctmp,etmp,comber


# In[17]:


def reset_variables(i,k,j):
    igate = i+k
    jgate = j+1
    ceiling = 0
    rate = 0
    return igate,jgate,ceiling,rate


# In[18]:


def greedy_operations(ceiling,tmp,en,i,k,j):
    ctmp,etmp,combstr = word_output_combination(tmp,en,i,k)
    if len(combstr)>1:
        print('Combination of {0}'.format(combstr))
    print('R:{0:4f}\nC:{1}\nE:{2}\n'.format(ceiling,ctmp,etmp))
    igate,jgate,ceiling,rate = reset_variables(i,k,j)
    return ctmp,etmp,igate,jgate,ceiling,rate


# # greedy linear 演算法

# In[19]:


def greedy(chlist,enlist,method='bert',threshold=0.85):
    igate = 0
    jgate = 0
    ceiling = 0
    rate = 0
    output = []
    ctmp = ''
    etmp = ''
    for i in range(0,len(chlist)):
        tmp = ''
        if i>=igate:
            for j in range(jgate,i+5):
                if j<len(enlist) and j>=0:
                    rate = choose_your_algo(chlist[i],enlist[j],method)    
                    ceiling = rate
                    #print(i,j,rate)
                    if rate>threshold:
                        tmp = chlist[i]
                        #print(tmp)
                        for k in range(1,5):
                            if (i+k)<len(chlist):
                                rate = choose_your_algo((tmp+chlist[i+k]),enlist[j],method)
                               
                                if rate > ceiling:
                                    tmp = tmp + chlist[i+k]
                                    ceiling = rate
                                    #print(tmp)
                                    if k==4:
                                    #Window用完
                                        ctmp,etmp,igate,jgate,ceiling,rate = greedy_operations(ceiling,
                                                                                           tmp,enlist[j],i,k,j)
                                        output.append([ctmp,etmp])
                                        break
                                else:
                                #結果沒變好直接吐出值
                                    ctmp,etmp,igate,jgate,ceiling,rate = greedy_operations(ceiling,
                                                                                           tmp,enlist[j],i,k,j)
                                    output.append([ctmp,etmp])
                                    break
                            else:
                            #中文用完
                                ctmp,etmp,igate,jgate,ceiling,rate = greedy_operations(ceiling,
                                                                                           tmp,enlist[j],i,k,j)
                                output.append([ctmp,etmp])
                                break
                        break
    return(output)


# In[41]:


ch_template = [['本','論文','主要','討論','機器','學習'],['機器人','在','現代'],['對','人類','很','重要'],['同時','電腦','也','很','重要'],['我','喜歡','論文']]
en_template = [['This','paper','mainly','discuss','machine','learning'],['Robots','are','important','to','people','nowadays','and','so','do','computers'],['I','like','paper']]


# In[42]:


for i in range(0,len(ch_template)): 
    for j in range(0,len(en_template)):
        print(i,j,bert_matrix(ch_template[i],en_template[j]))


# In[139]:


greedy(ch_template,en_template)


# In[29]:


for i in range(104,105):
    print('-----Comparing set {0} using bert-----'.format(i))
    greedy(chcut[i],encut[i],method='bert',threshold=0.85)


# In[66]:


for i in range(10,15):
    print('-----Comparing set {0} using matrix-----'.format(i))
    greedy(chcut[i],encut[i],method='matrix',threshold=0.78)


# In[105]:


for i in range(10,15):
    print('-----Comparing set {0} using bert-----'.format(i))
    greedy(chcut[i],encut[i],method='bert')


# In[9]:


evaluate_data = readfile('Bert_evaluation.csv')


# In[10]:


c_eval_src = []
e_eval_src = []
#while not pin > len(evaluate_data):
for i in range(0,len(evaluate_data)):
    #length = random.randint(3,15)
    #if not pin+length>len(evaluate_data):
    c_eval_src.append(evaluate_data[i][2].split('#'))
    e_eval_src.append(evaluate_data[i][0].split())


# In[11]:


pin = 0
tmp = []
tmp2 = []
c_eval = []
e_eval = []
for i in range(0,5):
    length = random.randint(5,15)
    pin = random.randint(0,len(evaluate_data))
    tmp = c_eval_src[pin:pin+length]
    tmp2 = e_eval_src[pin:pin+length]
    c_eval.append(tmp)
    e_eval.append(tmp2)


# In[12]:


len(c_eval[0])


# In[30]:


found=0
in_between=0
for i in range(0,1000):
    if i%100==0:
        print(i)
    score = bert_calculator(c_eval_src[i],e_eval_src[i])
    if score>0.85:
        found+=1
    elif score>0.8:
        in_between+=1


# In[43]:


print("Recall:",found/1000)
print("Recall(threshold=0.8):",(found+in_between)/1000)


# In[64]:


found=0
in_between=0
for i in range(0,1000):
    if i%100==0:
        print(i)
    score = bert_matrix(c_eval_src[i],e_eval_src[i+2000])
    if score>0.7:
        found+=1
    elif score>0.65:
        in_between+=1
print("Recall(threshold=0.7:",found/1000)
print("Recall(threshold=0.65):",(found+in_between)/1000)


# In[157]:


for i in range(0,5):
    print('-----Comparing set {0} using bert-----'.format(i))
    greedy(c_eval[i],e_eval[i])


# # 沒用

# c_eval_src = c_eval_src+chcut
# e_eval_src = e_eval_src+encut

# ch_eval_dic = {}
# for i in range(0,len(c_eval_src)):
#     for j in range(0,len(c_eval_src[i])):
#         for word in c_eval_src[i][j]:
#             if ch_eval_dic.get(word) == None:
#                 ch_eval_dic[word] = [i]
#                 #print(word)
#             else:
#                 if i not in ch_eval_dic[word]: 
#                     ch_eval_dic[word].append(i)

# en_eval_dic = {}
# for i in range(0,len(e_eval_src)):
#     for j in range(0,len(e_eval_src[i])):
#         for word in e_eval_src[i][j]:
#             word = Porter.stem(word)
#             if en_eval_dic.get(word) == None:
#                 en_eval_dic[word] = [i]
#             else:
#                 if i not in en_eval_dic[word]: 
#                     en_eval_dic[word].append(i)
