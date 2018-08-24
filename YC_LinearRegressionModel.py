# -*- coding: utf-8 -*-
"""
@author: YC
"""

''' Import the required module... ''' 
import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import pickle
import time



''' Read the csv file (dataset) '''
#df = pd.read_csv('Morgan_2_1024_100K.csv.gz', compression = 'gzip')
df = pd.read_csv('Morgan_2_1024_10K.csv')
print('\n')
print('Original dataset ....... ')
print(df.columns.values)
print('The shape of the dataset is {} ...'.format(df.shape))

## ====================================================================================================
## df = ... 
## |1024 or 2048 columns for each value of the bit vector | inchi_key | Smiles | Assay results | Mols |
## Example: inchi_key -> 'HENMVSORDOFOGZ-UHFFFAOYSA-N'
##          smiles    -> 'COC1=CC(=CC(=C1OC)OC)C(=O)NCCNC2=N[S](=O)(=O)C3=CC=CC=C23'
##          Mols      -> '<rdkit.Chem.rdchem.Mol object at 0x0000005C08F74670>'
## Size : (100000, 1270)
## ====================================================================================================

## -----------------------------------------------------------------------------

''' Discard unnecessary columns such as inchi_key, smiles, and Mols '''
df.pop('inchi_key')
df.pop('smiles')
df.pop('Mols')
print('\n')
print('Now dataset becomes after discarding unnecessary columns ....... ')
print(df.columns.values)
print('The shape of the dataset is {} ...'.format(df.shape))

## ====================================================================================================
## New df = ... 
## |1024 or 2048 columns for each value of the bit vector | Assay results |
## Example: inchi_key -> 'HENMVSORDOFOGZ-UHFFFAOYSA-N'
##          smiles    -> 'COC1=CC(=CC(=C1OC)OC)C(=O)NCCNC2=N[S](=O)(=O)C3=CC=CC=C23'
##          Mols      -> '<rdkit.Chem.rdchem.Mol object at 0x0000005C08F74670>'
## Size : (100000, 1267)
## ====================================================================================================

## -----------------------------------------------------------------------------

''' 
  Discard 10 selected assays due to the missing value ... 
  ('504621','485358','2563','588358','2221','2314','588334','1554','1662','651550')  
'''
discardList = ['504621','485358','2563','588358','2221','2314','588334','1554','1662','651550']
for ind, i in enumerate(discardList):
  df.pop(i)
print('\n')
print('Now dataset becomes after discarding selected assays ....... ')
print(df.columns.values)
print('The shape of the dataset is {} ...'.format(df.shape))

## ====================================================================================================
## New df = ... 
## |1024 or 2048 columns for each value of the bit vector | Assay results |
## Example: inchi_key -> 'HENMVSORDOFOGZ-UHFFFAOYSA-N'
##          smiles    -> 'COC1=CC(=CC(=C1OC)OC)C(=O)NCCNC2=N[S](=O)(=O)C3=CC=CC=C23'
##          Mols      -> '<rdkit.Chem.rdchem.Mol object at 0x0000005C08F74670>'
## Size : (100000, 1257)
## ====================================================================================================

## -----------------------------------------------------------------------------

'''
  Define the function to train the model iterately ...

'''
def trainModel(n):
  
  '''
    args: 
      n: A integer. To indicate which assay is analyzing.
      
  '''
  
  ## ==================================================================================================
  
  ''' Retrieve the training, testing data. '''
  #n = 1 # First model to run (ex: 1, 2, 3, ...)
  assayID    = df.columns.values[1023+n]
  newdf = df.dropna(subset = [assayID])  ## Drop missing value
  nDel = df.shape[0] - newdf.shape[0]
  dataInput  = newdf.loc[:, '0':'1023'].values
  dataOutput = np.reshape(newdf.iloc[:, 1023+n].values, [-1, 1])
  
  # Split the data into training and testing data (80%, 20%)
  trainInput, testInput, trainOutput, testOutput = sklearn.model_selection.train_test_split(dataInput,
                                                                                            dataOutput,
                                                                                            test_size=0.2,
                                                                                            random_state=5)
  
  print('\n')
  print('Print the shape of individual data ... ')
  print(60*'=')
  print('    Training Input shape : {}'.format(trainInput.shape))
  print('    Training Output shape : {}'.format(trainOutput.shape))
  print('    Testing Input shape : {}'.format(testInput.shape))
  print('    Testing Output shape : {}'.format(testOutput.shape))
  print(60*'=')
  
  print(60*'-')
  print('\n')
  print('Start training the model ...')
  
  ## Build the linear regression model ...
  model = LinearRegression()
  tic = time.clock() # Calculate time spent
  model.fit(trainInput, trainOutput)
  toc = time.clock() # Calculate time spent
  elapseTime = toc - tic
  
  print('\n')
  print('Training finished !!!')
  print('Time passed for training the model: {:2.4f} seconds... '.format(elapseTime))
  print(60*'-')
  
  ## Predict the Z-score based on the linear regression model
  trainPrediction = model.predict(trainInput)
  testPrediction = model.predict(testInput)
  
  print('\n')
  print('Training error (mean squared error) : {:2.4f}'.format(np.mean(np.subtract(trainOutput, trainPrediction) ** 2)))
  print('Testing error (mean squared error) : {:2.4f}'.format(np.mean(np.subtract(testOutput, testPrediction) ** 2)))
  
  pickle.dump(model, open(assayID + '_LinearRegression.h5', 'wb'))
 
  # some time later...
   
  ## load the model from disk
  # loaded_model = pickle.load(open(assayID + '_LinearRegression.h5', 'rb'))

  return nDel, elapseTime, model, trainPrediction, trainOutput, testPrediction, testOutput

n = 1 
nDel, elapseTime, model, trainPrediction, trainOutput, testPrediction, testOutput = trainModel(n)

  
  
  
  




