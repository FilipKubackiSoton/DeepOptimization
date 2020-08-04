import numpy as np
import os 

Direct = os.getcwd()

# Encoder Weights (transpose is the decoder)
Filename = '/Train_Weights_L1_TrainLevel_1.npy'
Weights_Layer1 = np.load(Direct+ Filename)
print(Weights_Layer1.shape)
#print(Weights_Layer1)

Filename = '/Train_BiasEncoder_L1_TrainLevel_1.npy'
BiasEncoder_Layer1 = np.load(Direct+ Filename)
print(BiasEncoder_Layer1.shape)
#print(BiasEncoder_Layer1)

Filename = '/Train_BiasDecoder_L1_TrainLevel_1.npy'
BiasDecoder_Layer1 = np.load(Direct+ Filename)
print(BiasDecoder_Layer1.shape)
#print(BiasDecoder_Layer1)
Filename = '/TrainingData_Layer_1_Evo1.npy'
TrainingData_Layer1 = np.load(Direct+ Filename)
print(TrainingData_Layer1.shape)
#print(TrainingData)