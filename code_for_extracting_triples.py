# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:17:26 2017

@author: Diabetes.co.uk
"""

import pandas as pd
import nltk
from nltk import word_tokenize
import string
import os
import sys
from collections import Counter
os.chdir(os.path.dirname(sys.argv[0]))

questions = pd.read_csv('CSVfiles\\QuestionsWithClassCSV.csv')
print(questions)

def strip_sentence(sentence):
    sentence = sentence.strip(",")
    sentence = ''.join(filter(lambda x: x in string.printable, sentence))  #strip out non-alpha-numerix
    sentence = sentence.translate(str.maketrans('','',string.punctuation)) #strip punctuation
    return(sentence)

# Parts Of Speech
def get_pos(sentence):
    sentenceParsed = word_tokenize(sentence)
    return(nltk.pos_tag(sentenceParsed))

def get_triples(pos):
    list_of_triple_strings = []
    pos = [ i[1] for i in pos ]  # extract the 2nd element of the POS tuples in list
    n = len(pos)
    
    if n > 2:  # need to have three items
        for i in range(0,n-2):
            t = "-".join(pos[i:i+3]) # pull out 3 list item from counter, convert to string
            list_of_triple_strings.append(t)
    return list_of_triple_strings 

qonly = questions['SENTENCE']

Triples = []

for row in qonly:
    sentence = strip_sentence(row)
    pos = get_pos(sentence)
    triples = get_triples(pos)
    Triples.append(str(triples)[1:-1])
    
questions['Triples'] = Triples

#code for returning the dataframe as a csv
#questions.to_csv('QuestionsWithTriples.csv')

#looking for the unique column entries in question classes
questions.Class.unique()
questions.head(0)

#Breaking down the datframe into each class
valuelist1 = ['informationanddefinition']
informationanddefinition = questions[questions.Class.isin(valuelist1)]

valuelist2 = ['complications']
complications = questions[questions.Class.isin(valuelist2)]

valuelist3 = ['manifestation']
manifestation = questions[questions.Class.isin(valuelist3)]

valuelist4 = ['cause']
cause = questions[questions.Class.isin(valuelist4)]

valuelist5 = ['diagnosis']
diagnosis = questions[questions.Class.isin(valuelist5)]

valuelist6 = ['other']
other = questions[questions.Class.isin(valuelist6)]

valuelist7 = ['management']
management = questions[questions.Class.isin(valuelist7)]

#keeping only the triples columns:
informationanddefinitionTriples = informationanddefinition['Triples']

complicationsTriples = complications['Triples']

manifestationTriples = manifestation['Triples']
    
causeTriples = cause['Triples']

diagnosisTriples = diagnosis['Triples']

otherTriples = other['Triples']

managementTriples = management['Triples']

#counting the unique triples of each question type:


def triple_total(column):
    '''This function tell the total number of triples with in the columns.'''
    a = []
    for row in column:
        a.append(row.split(','))
    a1 = []
    for sublist in a:
        for item in sublist:
            a1.append(item.strip(' '))
    return(len(a1))

def triple_counting(column):
    '''This function uses the counter function from collections package to 
    count the distinct triples that is extracted from the questions of the specific class.
    this function can be tinkered to be used to count the frequency of any extracted features 
    that is stored within a column.'''
    a = []
    for row in column:
        a.append(row.split(','))
    a1 = []
    for sublist in a:
        for item in sublist:
            a1.append(item.strip(' '))
    Results = Counter(a1)# after defining the output of the function as a variable, you can use .most_common() to have it sorted in most commen to least.
    return(Results)

def triple_percent(counterresult):
    '''This function uses the two functions from above, and calculate the percentage of the respective triple.'''
    a = triple_counting(counterresult)
    b = triple_total(counterresult)
    a_percent = [(i, a[i] / b * 100.0) for i in a]
    result = sorted(a_percent, key=lambda percent: percent[1], reverse = True)
    return(result)

#######the codes below allows us to check which triple is important for which question type
triple_percent(informationanddefinitionTriples)
triple_percent(complicationsTriples)
triple_percent(manifestationTriples)
triple_percent(causeTriples)
triple_percent(diagnosisTriples)
triple_percent(otherTriples)
triple_percent(managementTriples)

#Extracting the starting and ending triples:
def get_first_last_triples(sentence):
    first_last_triples = []
    sentenceParsed = word_tokenize(sentence)
    pos = nltk.pos_tag(sentenceParsed) #Parts Of Speech
    pos = [ i[1] for i in pos ]  # extract the 2nd element of the POS tuples in list
    
    n = len(pos)
    first = ""
    last = ""
    
    if n > 1:  # need to have three items
        first = "-".join(pos[0:3]) # pull out first 2 list items
        last = "-".join(pos[-3:]) # pull out last 2 list items
    
    first_last_triples = [first, last]
    return first_last_triples

    