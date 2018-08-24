# -*- coding: utf-8 -*-
"""
@author: YC
"""

''' Import the required module... ''' 
import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


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
## Size : (10000, 1270)
## ====================================================================================================

## -----------------------------------------------------------------------------

''' Discard unnecessary columns (.pop) such as inchi_key, smiles, and Mols '''
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
## Size : (10000, 1267)
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
## Size : (10000, 1257)
## ====================================================================================================

## -----------------------------------------------------------------------------

modelsummary  = pd.DataFrame(columns=['assayID', 'nDel', 'elapseTime', 'test_loss', 'test_mae'])
trainidxarray = {}
testidxarray  = {}
testPrearray  = {}
testOutarray  = {}

'''
  Define the function to train the model iterately ...

'''
def trainModel(assayIDs):
  
  '''
    args: 
      n: A list. To indicate how many targets (assay) is analyzing.
        (ex: ['1511', '2363', or any assayID]) 
      
  '''
  
  ## ==================================================================================================
  
  ''' Retrieve the training, testing data. '''
  newdf = df.dropna(subset = assayIDs)  ## Drop missing value
  nDel = df.shape[0] - newdf.shape[0]
  dataInput  = newdf.loc[:, '0':'1023'].values
  dataOutput = np.reshape(newdf.loc[:, assayIDs].values, [-1, len(assayIDs)])
  
  # Split the data into training and testing data (80%, 20%)
  indices = np.random.permutation(dataInput.shape[0])
  nInputs = np.int(np.round(dataInput.shape[0]*0.8))
  training_idx , test_idx = indices[:nInputs], indices[nInputs:]
  trainInput, testInput = dataInput[training_idx,:], dataInput[test_idx,:]
  trainOutput, testOutput = dataOutput[training_idx,:], dataOutput[test_idx,:]
  
#  ''' Normalize the features (Only use train data to calculate the mean and std) '''
#  mean = trainInput.mean(axis=0)
#  std = trainInput.std(axis=0)
#  trainInput = (trainInput - mean) / std
#  testInput  = (testInput - mean) / std
  
  ## -----------------------------------------------------------------------------
  
  '''
    Build the NNet model using keras ...
    
  '''
  def build_model():
    model = keras.Sequential([
      keras.layers.Dense(64, activation=tf.tanh, 
                         input_shape=(trainInput.shape[1],)),
      keras.layers.Dense(len(assayIDs))
    ])
    
    ## You can visit Keras website to add more complex layers into the model
    ## (https://keras.io/)
  
    optimizer = keras.optimizers.Adadelta()
  
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model
  
  model = build_model()
  print('\n')
  print('The summary of the model ..........')
  model.summary()
  
  ## Display training progress by printing a single dot for each completed epoch.
  #class PrintDot(keras.callbacks.Callback):
  #  def on_epoch_end(self,epoch,logs):
  #    if epoch % 100 == 0: print('')
  #    print('', end='')
  
  def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0,2])    
  
  # The patience parameter is the amount of epochs to check for improvement.
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
  
  EPOCHS = 5000
  
  print(60*'-')
  print('\n')
  print('Start training the model ...')
  
  # Store training stats
  tic = time.clock() # Calculate time spent
  history = model.fit(trainInput, trainOutput, epochs=EPOCHS,
                      validation_split=0.2, verbose=1,
                      callbacks=[early_stop])
  toc = time.clock() # Calculate time spent
  elapseTime = toc - tic
  
  
  print('\n')
  print('Training finished !!!')
  print('Time passed for training the model: {} seconds... '.format(elapseTime))
  print(60*'-')
  
  plot_history(history)
  
  print('\n')
  print('Evaluate the model ... ')
  
  [loss, mae] = model.evaluate(testInput, testOutput, verbose=1)
  
  print('\n')
  print('Evaluation: mean squared error = {:1.4f}, mean absolute error = {:1.4f}'.format(loss, mae))
  
  test_predictions = model.predict(testInput)
  model.save('Last10' + '_multiTaskNNet.h5')
  
  dataFrame = pd.DataFrame(columns=['assayID', 'nDel', 'elapseTime', 'test_loss', 'test_mae'])
  for i in range(len(assayIDs)):
    newframe = pd.DataFrame({'assayID': assayIDs[i], 'nDel': nDel, 'elapseTime': elapseTime,
                             'test_loss': np.mean(np.square(np.subtract(test_predictions[:, i], testOutput[:, i]))),
                             'test_mae': np.mean(np.absolute(np.subtract(test_predictions[:, i], testOutput[:, i])))},
                            index = [i])
    dataFrame = pd.concat([dataFrame, newframe])

  ## Load model
  # model = load_model(assayID + '_NNet.h5')
  
  return assayIDs, nDel, training_idx , test_idx, model, elapseTime, history, loss, mae, test_predictions, testOutput, dataFrame


assayIDs = ['1511', '743269', '743279']  # ...... Put the selected assays ...........
assayIDs, nDel, training_idx , test_idx, model, elapseTime, history, loss, mae, test_predictions, testOutput, dataFrame = trainModel(assayIDs)
modelsummary = modelsummary.append(dataFrame, ignore_index=True)
#  trainidxarray[assayID] = training_idx
#  testidxarray[assayID] = test_idx
#  testPrearray[assayID] = test_predictions
#  testOutarray[assayID] = testOutput


## Save the summary to the csv file
modelsummary.to_csv('multiTaskmodelsummary.csv')
np.save('multiTasktrainidx.npy', training_idx) 
np.save('multiTasktestidx.npy', test_idx) 
np.save('multiTasktestPre.npy', test_predictions) 
np.save('multiTasktestOut.npy', testOutput) 

#plt.figure()
#plt.boxplot(np.abs(np.subtract(testOutput, test_predictions)))



