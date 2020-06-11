To train the net type in the working directory: 
python train_shallowNet.py 

Model is saved in the direcotry saved_model
Options:
--plot: directory of resulting plot 
--input: size of the input array 
--latent: size of the latend net 
--size: size of the training set 
--epochs: number of epochs 
--batch: batch size

To generate plots type in the same direcotry: 
python netInterpret.py 

Plots are based on the model from the directory: saved_model
Options:
--aplot: directory to save plot of activation latent layer 
--wplot: dierctory to save plot of weights to latent layer
--size: set size to evaluate model
