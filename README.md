# Activity scheduler and mode choice model
The nhts_simple.py script uses data from the National Household Travel Survey to calibrate a simple activity-based mobility model.


## Inputs


##### NHTS data
The main source of data for this analysis is the National Household Travel Survey (NHTS).


###### CBSA

The U.S. geographic area from which the subset of NHTS data is used.
`CBSA` is defined as a constant within `nhts_simple.py`.  Change it to compute the model for other areas.


###### Blocks data

The design team collaborated with the technical team on the design of the city blocks used in the model simulation via a spreadsheet ([link](https://docs.google.com/spreadsheets/d/1aKRPp83EWWdBri6MwjuGcZVSki2xQuxd_OjMAdrrpS4)).

That spreadsheet is downloaded and saved to `blocks.csv`. The data defines one city block per row and specifies the relative capacity of that block for offices, residential, and amenities.  Based on this data, the simulated population is assigned to these blocks for the office, residential, and amenity places that they take trips to.


###### Survey participant attributes

The output is based on personal attributes from the survey participants, including:
occupation type, household income, household lifecycle, age, mobility motif.

For more documentation, see https://nhts.ornl.gov/assets/codebook_v1.1.pdf

Occupation types:

| Code 	| Description											|
|-------|-------------------------------------------------------|
| 1		| Sales or service										| 
| 2		| Clerical or administrative 							|
| 3		| Manufacturing, construction, maintenance, or farming	| 
| 4		| Professional, managerial, or technical 				|
| 5		| Student 												|


Household lifecycles: 
Life Cycle classification for the household, derived by attributes pertaining to age, relationship, and work status.

| Code 	| Description
|-------|-------------------------------------------------------|
| -9 | Not ascertained 1 |
| 01 | one adult, no children |
| 02 | 2+ adults, no children 27 |
| 03 | one adult, youngest child 0-5 |
| 04 | 2+ adults, youngest child 0-5 |
| 05 | one adult, youngest child 6-15 |
| 06 | 2+ adults, youngest child 6-15 |
| 07 | one adult, youngest child 16-21 |
| 08 | 2+ adults, youngest child 16-21 |
| 09 | one adult, retired, no children |
| 10 | 2+ adults, retired, no children 



###### maxDepth

The maximum depth of the decision tree which will be used to predict mode choice for each choice. This also corresponds to the number of if-else statements required to make the mode choice prediction.

## Outputs

simPop.csv is a dataframe containing the simulated population corresponding to each defined block. This includes personal characteristics of the population as well as a mobility motif for each person.

treeModeSimple.pdf is a visualisation of the calibrated decision tree for predicting mobility mode choice. The four possible options are as follows:

| 0       | 1       | 2       | 3       |
|---------|---------|---------|---------|
| driving | cycling | walking | shared transit |

An example tree is shown below.

![viz](./example_tree.png)


mode_choice.py contains python code for the series of if -else statements corresponding to the calibrated decision tree. This script is created by running the nhts_simple.py script. 


## Running

###### Download the NHTS Data

This data is large and excluded from the repository via an entry in the `.gitignore`.

- Create a folder to hold the data `./nhts/`
- Download the NHTS 2017 v1.1 csv files from https://nhts.ornl.gov/ and save `./nhts/trippub.csv` and `./nhts/perpub.csv`


###### Download the Blocks Data

Download the blocks data from the [spreadsheet](https://docs.google.com/spreadsheets/d/1aKRPp83EWWdBri6MwjuGcZVSki2xQuxd_OjMAdrrpS4) to `blocks.csv`.

###### Recompute

`python nhts_simple.py`
