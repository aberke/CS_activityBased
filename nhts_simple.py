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


# Set the data input filepath locations.
BLOCKS_DATA_CSV = 'blocks.csv'
# Need to download the NHTS 2017 v1.1 csv files from:
# https://nhts.ornl.gov/
NHTS_TRIPS_DATA_CSV = './nhts/trippub.csv'
NHTS_PERSONS_DATA_CSV = './nhts/perpub.csv'

# Set the output filepath locations.
# The output CSV contains the simulated population
SIMULATED_POPULATION_OUTPUT_CSV = 'results/simPop.csv'
MODE_CHOICE_PREDICTOR_FILEPATH = 'results/mode_choice.py'

# Set maximum decision tree depth.
# This will be the number of if statements to decide on a mode choice.
DT_MAX_DEPTH = 3

# Only use NHTS data from target population subset.
CBSA = '35620'  # New York-ish


def tree_to_code(tree, feature_names):
    # Takes a fitted decision tree and outputs a python function
    with open(MODE_CHOICE_PREDICTOR_FILEPATH, 'w') as the_file:
        the_file.write('def predict_mode_probs():\n')
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


# ********************************************************************
# Clean and prepare the NHTS data
# ********************************************************************

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
#OCCAT: Job category

# Read in the NHTS data and prune it.
trips = pd.read_csv(NHTS_TRIPS_DATA_CSV)
persons = pd.read_csv(NHTS_PERSONS_DATA_CSV)

# Take a subset of the total trips data for the New York, New Jersey and
# Newark urban areas (nnn)
trips_nnn = trips.loc[((trips['HH_CBSA'] == CBSA) & (trips['HBHUR'] == 'U'))]  # ny, nj, newark, Urban
# Only use subset of data with sensible trips data.
trips_nnn = trips_nnn[(trips_nnn['WHYTO'] > 0) & (trips_nnn['WHYFROM'] > 0)]
# Remove small number of trip legs which were just for changing mode of transportation.
trips_nnn_nTrans = trips_nnn[((trips_nnn['WHYTO'] != 7) & (trips_nnn['WHYFROM'] != 7))]

# Add a unique person ID to trips and persons data and combine the datasets.
UNIQUE_PERSON_ID = 'uniquePersonId'

def get_unique_person_id(row):
    return str(row['HOUSEID']) + '_' + str(row['PERSONID'])

trips_nnn_nTrans[UNIQUE_PERSON_ID] = trips_nnn_nTrans.apply(lambda row: get_unique_person_id(row), axis=1)
persons[UNIQUE_PERSON_ID] = persons.apply(lambda row: get_unique_person_id(row), axis=1)
persons_nnn = persons.loc[persons[UNIQUE_PERSON_ID].isin(trips_nnn_nTrans[UNIQUE_PERSON_ID])]
# make a new OCCAT or students
persons_nnn.at[persons_nnn['SCHTYP'] > 0, 'OCCAT'] = 5
# remove records with unknown variables that we need
persons_nnn = persons_nnn.loc[persons_nnn['HHFAMINC'] >= 0]
persons_nnn = persons_nnn.loc[persons_nnn['PRMACT'] >= 0]
persons_nnn = persons_nnn.loc[persons_nnn['LIF_CYC'] >= 0]
persons_nnn = persons_nnn.loc[persons_nnn['OCCAT'] >= 0]
# only keep people whose travel diary was on a weekday
weekdays = [2, 3, 4, 5, 6]
persons_nnn = persons_nnn.loc[persons_nnn['TRAVDAY'].isin(weekdays)]


# ********************************************************************
# Add & transform activity/mobility data
# ********************************************************************


# Maps NHTS activity data to simpler list of activities.
# It also transforms numeric encodings to character encodings that are more
# user friendly.
activity_why_dict = {
        1:'R',2:'R',
        3:'O',4:'O',5:'O',
        6:'A',
        7:'T',
        8:'O',
        9:'A',10: 'A',11: 'A',12: 'A',13: 'A',14: 'A',
        15: 'A',16: 'A',17: 'A',18: 'A',19: 'A',97: 'A'}


# Maps NHTS modes to a simpler list of modes.
mode_dict = {
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

reg_patterns = OrderedDict()
# define the motifs and create regular expressions to represent them
# Based on http://projects.transportfoundry.com/trbidea/schedules.html
reg_patterns[re.compile('(R-)+R')] = 1  # 'home'
reg_patterns[re.compile('(R-)+O-R')] = 2  # 'simpleWork'
reg_patterns[re.compile('(R-)+O-(O-)+R')] = 3  #'multiPartWork'
reg_patterns[re.compile('(R-)+A-R')] = 4  #'simpleNonWork'
reg_patterns[re.compile('(R-)+A-(A-)+R')] = 5  #'multiNonWork'
reg_patterns[re.compile('(R-)+A-([OA]-)*O-R')] = 6  #'compToWork'
reg_patterns[re.compile('(R-)+O-([OA]-)*A-R')] = 7  #'compFromWork'
reg_patterns[re.compile('(R-)+A-([OA]-)*O-([OA]-)*A-R')] = 8  #'compToFromWork'
reg_patterns[re.compile('(R-)+O-([OA]-)*A-([OA]-)*O-R')] = 9  #'compAtWork'

def find_pattern(sched, reg_patterns):
    # Takes a sequence of activities and maps it to one of the pre-defined motifs
    for r in reversed(reg_patterns):
        # checking the most complex first
        if r.match(sched):
            return reg_patterns[r]
    if sched == 'R':
        return 1
    else:
        return 0

# Use simplest version of each pattern
pattern_dict = {1: 'R',
                2: 'ROR',
                3: 'ROOR',
                4: 'RAR',
                5: 'RAAR',
                6: 'RAOR',
                7: 'ROAR',
                8: 'RAOAR',
                9: 'ROAOR'}


# Build a mapping from person id --> daily schedule in order to add mobility motifs
trips_nnn_nTrans['whyToMapped'] = trips_nnn_nTrans.apply(lambda row: activity_why_dict[row['WHYTO']], axis=1)
trips_nnn_nTrans['whyFromMapped'] = trips_nnn_nTrans.apply(lambda row: activity_why_dict[row['WHYFROM']], axis=1)
person_id_to_motif = dict()
for p_id in set(trips_nnn_nTrans[UNIQUE_PERSON_ID]):
    mapped_sched = [trips_nnn_nTrans.loc[trips_nnn_nTrans[UNIQUE_PERSON_ID]==p_id]['whyFromMapped'].iloc[0]]
    mapped_sched.extend(trips_nnn_nTrans.loc[trips_nnn_nTrans[UNIQUE_PERSON_ID]==p_id]['whyToMapped'].tolist())
    # Assume each day starts at home.
    if not mapped_sched[0] == 'R':
        mapped_sched.insert(0, 'R')
    if not mapped_sched[-1] == 'R':
        mapped_sched.extend(['R'])
    str_sched = '-'.join(mapped_sched)
    sched_pattern = find_pattern(str_sched, reg_patterns)
    person_id_to_motif[p_id] = pattern_dict[sched_pattern]

# Add the motif (simplest example of the mobility pattern) to the persons and trips dataframes.
MOTIF_KEY = 'motif'
persons_nnn[MOTIF_KEY] = persons_nnn.apply(lambda row: person_id_to_motif[row[UNIQUE_PERSON_ID]], axis=1)
trips_nnn_nTrans[MOTIF_KEY] = trips_nnn_nTrans.apply(lambda row: person_id_to_motif[row[UNIQUE_PERSON_ID]], axis=1)


# ********************************************************************
# Generate simulated population from blocks data
# ********************************************************************
motifs_set = set(persons_nnn[MOTIF_KEY])

persons_simple = persons_nnn[['HHFAMINC', 'LIF_CYC',  'OCCAT', 'R_AGE_IMP']]
persons_simple = persons_simple.rename(columns={'HHFAMINC': 'hh_income', 'LIF_CYC': 'hh_lifeCycle',  'OCCAT': 'occupation_type', 'R_AGE_IMP': 'age'})
persons_simple = pd.concat([persons_simple, pd.get_dummies(persons_nnn[MOTIF_KEY], prefix=MOTIF_KEY)],  axis=1)


def build_simulated_population(blocks, persons, motifs_set):
    """ Builds the simulated population based on blocks characteristics.
    The product that is built is a data frame where each row is an 'agent'
    that is assigned a block of residence, block of work, and mobility motif.

    Args:
        blocks -- dataframe where each row is data about a simulated city block.
        persons -- dataframe where each row is a person from NHTS data

    Returns (dataframe) simulated population.
    """

    # Make list of occupation type names.
    occupation_types = [c for c in blocks.columns if 'occupationCat' in c]
    # Create capacity constrained choice set for resiential and third place locations
    residential_choice_set = [int(blocks.iloc[i]['Block']) for i in range(len(blocks)) for r in range(int(blocks.loc[i]['residential']))]
    amenity_choice_set = [int(blocks.iloc[i]['Block']) for i in range(len(blocks)) for r in range(int(blocks.loc[i]['third_places']))]

    # Produce simulated population as persons sampled from the persons data
    # based on their occupation type.
    simulated_pop = pd.DataFrame()
    # Iterate over the blocks, and the occupation types for that block.
    # Sample from the persons data enough people of given occupation type to work
    # in that block
    for b in range(len(blocks)):
        for occupation_type in occupation_types:
            occat = int(occupation_type.split('_')[1])
            N = blocks.iloc[b][occupation_type]
            if N > 0:
                sample = persons.loc[persons['occupation_type']==occat].sample(n=N, replace=True)
                sample['work_block'] = blocks.iloc[b]['Block']
                simulated_pop = simulated_pop.append(sample)
    simulated_pop['home_block'] = np.random.choice(residential_choice_set, len(simulated_pop), replace=False)
    # Randomly assign 'third places' that simulated population agents go to.
    # This only applies to agents with mobility motifs that indicate they go to amenities.
    simulated_pop['third_places_block'] = float('nan')  # Default is that agent has no third place.
    # Identify people who need a 3rd place
    goes_to_amenity = simulated_pop.apply(lambda row: bool(sum([row['motif_'+m] for m in motifs_set if 'A' in m])), axis=1).tolist()
    simulated_pop.at[goes_to_amenity, 'third_places_block'] = np.random.choice(amenity_choice_set, sum(goes_to_amenity), replace=False)
    simulated_pop = simulated_pop.reset_index(drop=True)
    return simulated_pop


blocks = pd.read_csv(BLOCKS_DATA_CSV)
simulated_pop = build_simulated_population(blocks, persons_simple, motifs_set)
simulated_pop.to_csv(SIMULATED_POPULATION_OUTPUT_CSV)


# ********************************************************************
# Generate Mode Choice Decision Tree
# ********************************************************************
trips_nnn_nTrans = trips_nnn_nTrans.rename(columns={'HHFAMINC': 'hh_income', 'LIF_CYC': 'hh_lifeCycle', 'R_AGE_IMP': 'age', 'TRPMILES': 'trip_leg_miles'})
dt_features = ['hh_income', 'hh_lifeCycle', 'age',  'trip_leg_miles']

trips_nnn_nTrans = trips_nnn_nTrans.merge(persons_nnn, how='left', on=UNIQUE_PERSON_ID, suffixes=('', '_copy'))
trips_nnn_nTrans['simpleMode'] = trips_nnn_nTrans.apply(lambda row: mode_dict[row['TRPTRANS']], axis=1)
trips_nnn_nTrans = trips_nnn_nTrans.loc[trips_nnn_nTrans['simpleMode'] >= 0]

new_dummies = pd.get_dummies(trips_nnn_nTrans[MOTIF_KEY], prefix=MOTIF_KEY)
trips_nnn_nTrans = pd.concat([trips_nnn_nTrans, new_dummies],  axis=1)
dt_features.extend(new_dummies.columns.tolist())
dt_data = trips_nnn_nTrans[dt_features]

clf_mode = tree.DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, class_weight='balanced')            
clf_mode = clf_mode.fit(np.array(dt_data), np.array(trips_nnn_nTrans['simpleMode']))
tree_to_code(clf_mode, dt_features)
