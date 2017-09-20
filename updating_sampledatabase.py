# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:06:35 2017

@author: Diabetes.co.uk
"""

#this files allow you to update the sampledatabase with new questions and answers,
#the new entries need to be stores in a csv file named 'NewQuestionsWithAnswersAndClassCSV.csv' with in CSVfiles

import pandas as pd
import nltk
import functions_for_extracting_pronouns_and_entities_using_api as extract

sampledatabase = pd.read_csv('sampledatabase.csv', index_col = 0) #loading the original database

#######################################################################
### Analysing the new entries and appending them to the orginal sampledatabase
#######################################################################

NewQuestions = pd.read_csv('CSVfiles\\NewQuestionsWithAnswersAndClassCSV.csv')

Answers = NewQuestions['ANSWER']
Questionsonly = NewQuestions['SENTENCE']

firstsent = []

for row in Answers:
    results = nltk.sent_tokenize(row)
    firstsent.append(results[0].lower())
    
NewQuestions['Answerfirstsent'] = firstsent

#Extracting the adjectives, nouns, named entities of the sentences and storing it in new columns:
    
AnswerAdjectives = []
AnswerNouns = []
AnswerEntities = []

for rows in firstsent:
    tokens1 = extract.get_tokens(rows)
    aNOUN = extract.Nounswords(tokens1)
    AnswerNouns.append(aNOUN)
    #Adjectives
    aADJECTIVE = extract.Adjectivewords(tokens1)
    AnswerAdjectives.append(aADJECTIVE) 
    #Named entities
    named_entities1 = extract.entities_name1(rows)
    AnswerEntities.append(named_entities1)

QuestionsAdjectives = []
QuestionsNouns = []
QuestionsEntities = []

for rows in Questionsonly:
    tokens1 = extract.get_tokens(rows)
    aNOUN = extract.Nounswords(tokens1)
    QuestionsNouns.append(aNOUN)
    #Adjectives
    aADJECTIVE = extract.Adjectivewords(tokens1)
    QuestionsAdjectives.append(aADJECTIVE) 
    #Named entities
    named_entities1 = extract.entities_name1(rows)
    QuestionsEntities.append(named_entities1)

NewQuestions['QuestionsAdjectives'] = QuestionsAdjectives
NewQuestions['QuestionsNouns'] = QuestionsNouns
NewQuestions['QuestionsEntities'] = QuestionsEntities
NewQuestions['AnswerAdjectives'] = AnswerAdjectives
NewQuestions['AnswerNouns'] = AnswerNouns
NewQuestions['AnswerEntities'] = AnswerEntities

sampledatabase = sampledatabase.append(NewQuestions)
sampledatabase = sampledatabase.reset_index()
id1 = []
for i in range(1,len(sampledatabase['ID'])+1):
    id1.append(i)

sampledatabase['ID'] = id1
sampledatabase.to_csv('sampledatabasewithDuplicates.csv')

duplicates = sampledatabase.duplicated('SENTENCE', keep = False)
duplicateentries = sampledatabase[duplicates]
duplicateentries.to_csv('Duplicate_Questions.csv')

sampledatabasenoduplicates = sampledatabase.drop_duplicates('SENTENCE', keep = 'first')
sampledatabasenoduplicates = sampledatabasenoduplicates.reset_index()
id1 = []
for i in range(1,len(sampledatabasenoduplicates['ID'])+1):
    id1.append(i)

sampledatabasenoduplicates['ID'] = id1
sampledatabasenoduplicates.to_csv('sampledatabase1.csv')