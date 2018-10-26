#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:34:59 2018

@author: doorleyr
"""

from collections import OrderedDict
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import _tree



def tree_to_code(tree, feature_names):
    # takes a fitted decision tree and outputs a python function
    with open('results/mode_choice.py', 'w') as the_file: 
        the_file.write('def predictModeProbs():\n')
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        def recurse(node, depth):
            indent = "    " * (depth)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                the_file.write("{}if {} <= {}:\n".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                the_file.write("{}else:  # if {} > {}\n".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                # Write the return statement as a list of weighted mobility choices.
                choice_values = tree_.value[node][0]
                n_samples = sum([int(v) for v in choice_values])
                transformed_choice_values = [v/n_samples for v in choice_values]
                formatted_choice_values = ["{:.2f}".format(v) for v in transformed_choice_values]
                the_file.write("{}return [{}]\n".format(indent, ", ".join(formatted_choice_values)))
        recurse(0, 1)
    
def findPattern(sched, regPatterns):
    # takes a sequence of activities and maps it to one of the pre-defined motifs
    for r in reversed(regPatterns):
        #checking the most complex first
        if r.match(sched):
            return regPatterns[r]
    if sched=='R':
        return 1
    else:
        return 0

cbsa='35620' # New York-ish

whyDict={
        # map NHTS activities to a simpler list of activities
        1:'R',2:'R',
        3:'O',4:'O',5:'O',
        6:'A',
        7:'T',
        8:'O',
        9:'A',10: 'A',11: 'A',12: 'A',13: 'A',14: 'A',
        15: 'A',16: 'A',17: 'A',18: 'A',19: 'A',97: 'A'}

modeDict={
        # map NHTS modes to a simpler list of modes
        # 0: drive, 1: cycle, 2: walk, 3: PT
        -7:-99,-8:-99,-9:-99,
        1:2,
        2:1,
        3:0,4:0,5:0,6:0,
        7:-99,
        8:0,9:0,
        10:3,11:3,12:3,13:3,14:3,15:3,16:3,
        17:0,18:0,
        19:3,20:3,
        97:-99}

regPatterns=OrderedDict()
#define the motifs and create regular expressions to represent them
# Based on http://projects.transportfoundry.com/trbidea/schedules.html
regPatterns[re.compile('(R-)+R')] = 1  # 'home'
regPatterns[re.compile('(R-)+O-R')] = 2  # 'simpleWork'
regPatterns[re.compile('(R-)+O-(O-)+R')] = 3  #'multiPartWork'
regPatterns[re.compile('(R-)+A-R')] = 4  #'simpleNonWork'
regPatterns[re.compile('(R-)+A-(A-)+R')] = 5  #'multiNonWork'
regPatterns[re.compile('(R-)+A-([OA]-)*O-R')] = 6  #'compToWork'
regPatterns[re.compile('(R-)+O-([OA]-)*A-R')] = 7  #'compFromWork'
regPatterns[re.compile('(R-)+A-([OA]-)*O-([OA]-)*A-R')] = 8  #'compToFromWork'
regPatterns[re.compile('(R-)+O-([OA]-)*A-([OA]-)*O-R')] = 9  #'compAtWork'

# simplest version of each pattern
patternDict={1:'R',
             2: 'ROR',
             3: 'ROOR',
             4: 'RAR',
             5: 'RAAR',
             6: 'RAOR',
             7: 'ROAR',
             8: 'RAOAR',
             9: 'ROAOR'}   

# define the number of people in each occupation type for each block type in the library
blocks=pd.read_csv('blocks.csv')
occTypes=[c for c in blocks.columns if 'occupationCat' in c]
# create capacity constrained choice sets for resiential and third place locations
resChoiceSet=[int(blocks.iloc[i]['Block']) for i in range(len(blocks)) for r in range(int(blocks.loc[i]['residential']))]
thirdPlaceChoiceSet=[int(blocks.iloc[i]['Block']) for i in range(len(blocks)) for r in range(int(blocks.loc[i]['third_places']))]


maxDepth=3 # the max number of if statements to decide on a mode choice

# need to download the NHTS 2017 v1.1 csv files from: 
# https://nhts.ornl.gov/
trips=pd.read_csv('./nhts/trippub.csv')
persons=pd.read_csv('./nhts/perpub.csv')

#NHTS variables:
#TRPMILES: shortest path distance from GMaps
#EDUC: educational attainment
#HHFAMINC: HH income
#HHVEHCNT: vehicle count (HH)
#HH_CBSA: core-based stat area (35620: New York-Newark-Jersey City, NY-NJ-PA, 14460: Boston-Cambridge-Newton)
#HOMEOWN: home ownership
#PRMACT: primary activity previous week (employment school etc.)
#R_AGE_IMP: age (imputed)
#TRAVDAY: day of week, 1: Sunday, 7: Sat
#TRPTRANS: mode
#WHYFROM: trip origin purpose
#WHYTO: Trip Destination Purpose
#OCCAT:	Job category

########################## Clean and prepare the NHTS data ######################################

# get subset of data for New York, New Jersey and Newark urban areas
trips_nnn=trips.loc[((trips['HH_CBSA']==cbsa) & (trips['HBHUR']=='U'))]# ny, nj, newark, Urban
trips_nnn['uniquePersonId']=trips_nnn.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
trips_nnn=trips_nnn[(trips_nnn['WHYTO']>0)&(trips_nnn['WHYFROM']>0)]

#remove small number of trip legs which were just for changing mode of transportation
trips_nnn_nTrans=trips_nnn[((trips_nnn['WHYTO']!=7) & (trips_nnn['WHYFROM']!=7))]

# map nhts activities to simple list of activities
trips_nnn_nTrans['whyToMapped']=trips_nnn_nTrans.apply(lambda row: whyDict[row['WHYTO']], axis=1)
trips_nnn_nTrans['whyFromMapped']=trips_nnn_nTrans.apply(lambda row: whyDict[row['WHYFROM']], axis=1)

persons['uniquePersonId']=persons.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
persons_nnn=persons.loc[persons['uniquePersonId'].isin(trips_nnn_nTrans['uniquePersonId'])]

# make a new OCCAT or students
persons_nnn.at[persons_nnn['SCHTYP']>0,'OCCAT']=5

#remove records with unknown variables that we need
persons_nnn=persons_nnn.loc[persons_nnn['HHFAMINC']>=0]
persons_nnn=persons_nnn.loc[persons_nnn['PRMACT']>=0] 
persons_nnn=persons_nnn.loc[persons_nnn['LIF_CYC']>=0]
persons_nnn=persons_nnn.loc[persons_nnn['OCCAT']>=0]
#only keep people whose travel diary was on a weekday
persons_nnn=persons_nnn.loc[persons_nnn['TRAVDAY'].isin([2,3,4,5,6])]

# get the daily activity schedule for each person using the regular expressions
daySched={}
for id in set(trips_nnn_nTrans['uniquePersonId']):
    mappedSched=[trips_nnn_nTrans.loc[trips_nnn_nTrans['uniquePersonId']==id]['whyFromMapped'].iloc[0]]
    mappedSched.extend(trips_nnn_nTrans.loc[trips_nnn_nTrans['uniquePersonId']==id]['whyToMapped'].tolist())
#    schedule=list(filter(lambda a: a != 7, schedule)) # remove changes in transportation
#    assume each day starts at home
    if not mappedSched[0]=='R':
        mappedSched.insert(0, 'R')
    if not mappedSched[-1]=='R':
         mappedSched.extend(['R'])
    strSched='-'.join(mappedSched)
    schedPattern=findPattern(strSched, regPatterns)
    daySched[id]=schedPattern

# add the day schedule to the persons and trips dataframes
trips_nnn_nTrans['daySched']= trips_nnn_nTrans.apply(lambda row: daySched[row['uniquePersonId']], axis=1) 
persons_nnn['daySched']=persons_nnn.apply(lambda row: daySched[row['uniquePersonId']], axis=1)

# add the motif (simplest example of the mobility pattern) to the persons and trips dataframes
persons_nnn['motif']=persons_nnn.apply(lambda row: patternDict[row['daySched']], axis=1)
trips_nnn_nTrans['motif']=trips_nnn_nTrans.apply(lambda row: patternDict[row['daySched']], axis=1)
allMotifs=set(trips_nnn_nTrans['motif'])

personsSimple=persons_nnn[['HHFAMINC', 'LIF_CYC',  'OCCAT', 'R_AGE_IMP']]
personsSimple=personsSimple.rename(columns={'HHFAMINC':'hh_income', 'LIF_CYC':'hh_lifeCycle',  'OCCAT':'occupation_type', 'R_AGE_IMP':'age'})
personsSimple=pd.concat([personsSimple, pd.get_dummies(persons_nnn['motif'], prefix='motif')],  axis=1)

######### Create the synthetic population based on the block characteristics ############### 

simPop=pd.DataFrame()
for b in range(len(blocks)):
    for typ in occTypes:
        occat=int(typ.split('_')[1])
        N=blocks.iloc[b][typ]
        if N>0:
            sample=personsSimple.loc[personsSimple['occupation_type']==occat].sample(n=N, replace=True)
            sample['work_block']=blocks.iloc[b]['Block']
            simPop=simPop.append(sample)
simPop['home_block']=np.random.choice(resChoiceSet,len(simPop), replace=False)
simPop['third_places_block']=float('nan')
goesThirdPlaces=simPop.apply(lambda row: bool(sum([row['motif_'+m] for m in allMotifs if 'A' in m])), axis=1).tolist()# identify people who need a 3rd place
simPop.at[goesThirdPlaces, 'third_places_block']=np.random.choice(thirdPlaceChoiceSet,sum(goesThirdPlaces), replace=False)
simPop=simPop.reset_index(drop=True)        
simPop.to_csv('results/simPop.csv')

######################### Mode Choice Decision Tree ###################################
trips_nnn_nTrans=trips_nnn_nTrans.rename(columns={'HHFAMINC':'hh_income', 'LIF_CYC':'hh_lifeCycle',  'R_AGE_IMP':'age', 'TRPMILES': 'trip_leg_miles'})
dtFeats= ['hh_income', 'hh_lifeCycle', 'age',  'trip_leg_miles']

trips_nnn_nTrans=trips_nnn_nTrans.merge(persons_nnn, how='left', on='uniquePersonId', suffixes=('', '_copy'))
trips_nnn_nTrans['simpleMode']=trips_nnn_nTrans.apply(lambda row: modeDict[row['TRPTRANS']], axis=1)
trips_nnn_nTrans=trips_nnn_nTrans.loc[trips_nnn_nTrans['simpleMode']>=0]

catVars=['motif']
for cv in catVars:
    newDummies=pd.get_dummies(trips_nnn_nTrans[cv], prefix=cv)
    trips_nnn_nTrans=pd.concat([trips_nnn_nTrans, newDummies],  axis=1)
    dtFeats.extend(newDummies.columns.tolist())

dtData=trips_nnn_nTrans[dtFeats]


clf_mode = tree.DecisionTreeClassifier(max_depth=maxDepth, class_weight='balanced')            
clf_mode = clf_mode.fit(np.array(dtData), np.array(trips_nnn_nTrans['simpleMode']))
# vis importance
plt.figure(figsize=(18, 16))    
plt.bar(range(len(clf_mode.feature_importances_)), clf_mode.feature_importances_)
plt.xticks(range(len(clf_mode.feature_importances_)), dtFeats, rotation=45)

tree_to_code(clf_mode, dtFeats)
