# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:10:00 2018

@author: YC
"""

import glob
import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt


## ==============================================================================
## ==============================================================================

'''
  In the beginning, we may run models in different time and save as different
  names, so we need to find out what those files are.
'''
fileList = [f for f in glob.glob("*.csv")]
fileList.remove('Morgan_2_1024_10K.csv')  # Remove the dataset csv file
fileList.remove('Morgan_2_2048_10K.csv')
nFile = len(fileList)
indicator = []
for i in range(nFile):
  indicator.append(fileList[i].replace('modelsummary_', ''))
  indicator[i] = indicator[i].replace('.csv', '')

'''
  Synthesize the data together ...
'''
## ==============================================================================

def fileSynthesis(nFile, indicator, filename, filetype):
  '''
    Purpose: Synthesize the data together!!
    args:
      nFile: An integer. Number of the files to synthesize.
      indicator: An string that is different among all the files.
      filename: An string. Name of the file.
      filetype: An string. Ex: 'Dataframe', 'npy', ...
  '''
  if filetype == 'Dataframe':
    synthesisFile = pd.read_csv(filename+'_'+indicator[0]+'.csv')
    synthesisFile = synthesisFile.drop(synthesisFile.columns.values[0], axis=1)    
    for i in range(1, nFile):
      anothersummary = pd.read_csv(filename+'_'+indicator[i]+'.csv')
      anothersummary = anothersummary.drop(anothersummary.columns.values[0], axis=1)
      synthesisFile = synthesisFile.append(anothersummary, ignore_index=True)
      del anothersummary
    # Sort the dataframe by the column 'assayID'
    synthesisFile = synthesisFile.sort_values(by=['assayID']).reset_index()
  elif filetype == 'npy': 
    synthesisFile = {}
    tosynthesisFile = np.load(filename+'_'+indicator[0]+'.npy').item()
    for i in range(1, nFile):
      tosynthesisFile.update(np.load(filename+'_'+indicator[i]+'.npy').item()) 
    tosynthesisFile = sorted(tosynthesisFile.items())
    for i in range(len(tosynthesisFile)):
      synthesisFile[tosynthesisFile[i][0]] = tosynthesisFile[i][1]
  return synthesisFile 

## ==============================================================================

modelsummary = fileSynthesis(nFile, indicator, 'modelsummary', 'Dataframe')
testPrearray = fileSynthesis(nFile, indicator, 'testPrearray', 'npy')
testOutarray = fileSynthesis(nFile, indicator, 'testOutarray', 'npy')

## ==============================================================================
## ==============================================================================

''' 
  Evaluate the model given modelsummary, testPrearray, testOutarray ...
'''

scoreDict = {}

## ==============================================================================

def evaluateModel(assayID, testPrearray, testOutarray):
  '''
    Purpose: Evaluate the model performance
    args:
      assayID: A string. The ID of the assay.
      testPrearray: An array of Prediction from testing input.
      testOutarray: An array of Actual testing output.
  '''
  test_predictions = testPrearray[assayID]
  testOutput = testOutarray[assayID]
  
  finalOutput = []
  finalPredic = []
  performance = 0
  
  for i in range(len(testOutput)):
    # =================================================
    # Convert Z-score into 0 (inactive), 1 (active)
    if np.absolute(float(testOutput[i])) < 3 :
      finalOutput.append(0)
    else:
      finalOutput.append(1)
    if np.absolute(float(test_predictions[i])) < 3 :
      finalPredic.append(0)
    else:
      finalPredic.append(1)
    # =================================================
    if finalOutput[i] == finalPredic[i]:
      performance += 1
  
  finalScore = performance / len(testOutput)
  return finalScore

## ==============================================================================

for i in range(len(testOutarray)):
  finalScore = evaluateModel(str(modelsummary['assayID'][i]), testPrearray, testOutarray)
  scoreDict[modelsummary['assayID'][i]] = finalScore

np.save('Score_All.npy', scoreDict) 

## ==============================================================================
assayID = str(modelsummary['assayID'][0])

maxDict = []
minDict = []
medianDict = []
errorDict = []

## ==============================================================================

def getErrorStatist(assayID, testPrearray, testOutarray):
  '''
    Purpose: Get the statistic of the error list in (Z-score).
    args:
      assayID: A string. The ID of the assay.
      testPrearray: An array of Prediction from testing input.
      testOutarray: An array of Actual testing output.
  '''
  test_predictions = testPrearray[assayID]
  testOutput = testOutarray[assayID]  
  errorList = np.abs(np.subtract(testOutput, test_predictions))
  median = np.median(errorList)
  mini = errorList.min()
  maxi = errorList.max()
  return maxi, mini, median, errorList

## ==============================================================================

for i in range(len(testOutarray)):
  maxi, mini, median, error = getErrorStatist(str(modelsummary['assayID'][i]), testPrearray, testOutarray)
  maxDict.append(maxi)
  minDict.append(mini)
  medianDict.append(median)
  errorDict.extend(error)

## ==============================================================================
## ==============================================================================
  

def plotError(testOutarray, errorDict, whichtype):
  '''
    Purpose: define the function to visualize the performance ...
    args:
      testOutarray: Array of the testing output.
      errorDict: the list of the error.
      whichtype: which type (ex: MAximum, Median, Minimum, ...) 
  '''
  plt.figure(figsize=(9, 10))
  plt.plot(list(range(1,len(testOutarray)+1)),errorDict, 'ro')
  plt.ylabel(whichtype + ' absolute error')
  plt.xlabel('Assays')
  plt.title(whichtype + ' absolute error across all assays')
  # Save the figure
  plt.savefig(whichtype + 'MAE_All.png')
  plt.show()
   
# Plot the figure 
  
plotError(testOutarray, maxDict, 'Maximum')
plotError(testOutarray, minDict, 'Minimum')
plotError(testOutarray, medianDict, 'Median')

plt.figure()
bp = plt.boxplot(np.asarray(errorDict), showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="darkblue"),
            capprops=dict(color="blue"),
            whiskerprops=dict(color="blue"))
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
for median in bp['medians']:
    median.set(color='red', linewidth=2)
for box in bp['boxes']:
    # change outline color
    box.set(linewidth=2) 
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) 
plt.ylim([-.2, 3.5])
plt.ylabel('Mean Absolute Error')
plt.title('Distribution of mean absolute error among all the trained models')
plt.show()

