
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import copy
import matplotlib.cm as cm
from KnapSack import KnapSack
from shallowNet.shallowNet import DenseTranspose


class UtilsPlot:

    def __init__(self, utilsGeneral, utilsModel):
        self.utg = utilsGeneral
        self.utm = utilsModel
        self.fitness_function = self.utg.fitness_function
        self.search = self.utg.flip

    def plot_weights_model(self, model, plot_name="weight_plot_default_name.png", number_of_deep_layers_to_show = None):
        """
        Visualize weights of model. 
        If added hidden layers is more than 1, then the function shows
        number of hidden layers, starting from the deepest one,
        defined by the param: number_of_deep_layers_to_show

        Params:
            model - model to show 
            path (default: weight_plot_default_name.png)- path where the plot with weights shall be ploted 
        Optional params: 
            number_of_deepl_layers_to_show (defult: None - all layers)- number of deep layers to print

        Possible Erros:
            number_of_deepl_layers_to_show cannot be greater than the actual 
            number of added layers
        """
        number_of_convoluted_layers = 0
        transpose_layer_type = str(type(DenseTranspose(None))).split(".")[-1]
        layers_to_show = []
        for l in model.layers:
            if str(type(l)).split(".")[-1] == transpose_layer_type: # compare types of convoluted layers
                number_of_convoluted_layers += 1
                layers_to_show.append(l)

        if number_of_deep_layers_to_show == None: # if param is None then show all convoluted layers 
            number_of_deep_layers_to_show = number_of_convoluted_layers
            
        # check if number of layers to show is fine
        assert number_of_convoluted_layers + 1 > number_of_deep_layers_to_show, "number of layers to visualize cannot exceded the real number of convoluted layers"  

        print("[INFO]: number of convoluted layers is equal to: ", number_of_convoluted_layers)
        index = 0

        if number_of_deep_layers_to_show == 1: # show encoder and decoder weights 
            fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout = True)
            fig.suptitle('Encoder/Decoder Weights', fontsize=16)
            im =axes[0].imshow(layers_to_show[0].get_weights()[1], interpolation='nearest', cmap=cm.Greys_r)
            axes[0].set_title("Encoder")
            axes[0].set_ylabel("Visible")
            axes[0].set_xlabel("Hidden")
            im =axes[1].imshow(layers_to_show[0].get_weights()[1], interpolation='nearest', cmap=cm.Greys_r)
            axes[1].set_title("Decoder")
            axes[1].set_ylabel("Visible")
            axes[1].set_xlabel("Hidden")
        else: # show hidden deep layers weights 
            fig, axes = plt.subplots(nrows=1, ncols=number_of_deep_layers_to_show, constrained_layout = True)
            fig.suptitle('Weights Matrix Image Reconstruction', fontsize=16)
            for ax in axes.flat:
                
                im = ax.imshow(layers_to_show[index].get_weights()[1], interpolation='nearest', cmap=cm.Greys_r)
                ax.set_title("Conv Layer : " + str(index))
                ax.set_xlabel("Hidden")
                ax.set_ylabel("Vissible")
                index+=1
        fig.colorbar(im, ax=axes.ravel().tolist())
        path = self.utg.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Weights of model were saved in the directory: ", path)


    def plot_model_loss(self, model_fit, plot_name, epochs):
        """
        Plot model loss to epochs

        Parameters: 
            model_fit - result of the model training 
            plot_name - name of the saving file 
        """
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epochs), model_fit.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), model_fit.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy\n")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        path = self.utg.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Loss plot was saved in the directory: ", path)

    


    def plot_fitness_development_phase(self, model, array, search = None, plot_name = None):
        """
        Plot fitnes development phase. 

        Parameters: 
            model - tf's model based on which we will evaluate 
            array - set of samples from which a one will be choosen 
            and will be passed to the transfer_sample_latent_flip 
            to see how does the model develop 
        """
        index = np.random.randint(self.utg.knapSack.Size)  
        encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        if search == None:
            search = self.search # modify encoded representation using default search function
        progress_set_evidence = self.utm.transfer_sample_latent_flip(
            array = array[index], 
            encoder = encoder, 
            decoder = decoder, 
            search = search)[-1]

        # construct a plot that plots and saves the training history
        N = np.shape(progress_set_evidence)[0] # size of the array 
        X = np.arange(0, N) # x range 
        plt.figure()
        plt.plot(X, progress_set_evidence, 'o', label="AutoEncoder")
        plt.title("Fitness of Solution after each Development Phase\n")
        plt.xlabel("Evolution Step")
        plt.ylabel("Fitness")
        plt.legend(loc="lower left")
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Fitness development phase plot was saved in the directory: ", path)


    def plot_evolution_model(self, model, array, plot_name = None, search = None, learning_steps = 50, debuge_variation=False):
        """
        Generate and save evolution plot of the model. 

        Parameters: 
            model - model to test 
            array - set of sample from which one will be choosen and evaluated
            plot_name - name of the saving plot 
        
        Optional parameters: 
            learning_steps - number of steps of sample evaluation 
        """
        if search == None:
            search = self.search # modify encoded representation using default search function
        encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        N = np.shape(array)[0] # size of the array 
        index = np.random.randint(N) #choose random index to flip 
        candidate_solution = array[index]# pick up random sample 
        sol_evol = [] # list to store steps of evolution 
        sol_evol.append(candidate_solution)
        current_fittnes = self.fitness_function(candidate_solution)
        for i in range(learning_steps-1):
            new_candidate_sol = copy.copy(candidate_solution)
            output_array, new_fitness = self.utm.code_flip_decode(
                        array = new_candidate_sol, 
                        encoder = encoder, 
                        decoder = decoder, 
                        search = search,
                        debuge_variation= debuge_variation
                        )
            if new_fitness >= current_fittnes:
                candidate_solution = output_array 
                current_fittnes = new_fitness
            sol_evol.append(candidate_solution)

        tmp = np.array(sol_evol)
        plt.figure()
        plt.imshow(tmp, interpolation='nearest', cmap=cm.Greys_r)
        plt.title("Solution Development at Evolution Step 1")
        plt.xlabel("Solution variable")
        plt.ylabel("Development Step")
        plt.colorbar()
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Evolution model plot was saved in the directory: ", path)

    def plot_latent_acitvation(self, model, plot_name, validation_set_size = 50):
        """
        Plot latent activation 

        Parameters: 
            model - model on which we are working 
            plot_name - name of the saving file 
        """
        # generate the val set 
        print("[INFO] generating validating dataset...")
        valY = self.utg.generate_training_sat(self.utg.knapSack.Size, validation_set_size)

        features_list = [layer.output for layer in model.layers[:4]]
        new_model = tf.keras.Model(inputs = model.input, outputs = features_list)
        predict = new_model.predict(valY)
        N = np.arange(0, len(predict[3][0]))

        plt.figure()
        for i in range(20):
            index = np.random.randint(len(predict[3][0]))
            plt.plot(N, predict[3][index], 'o',color = 'black')
        plt.title("L1 activation")
        plt.xlabel("Node #")
        plt.ylabel("Activation value")
        path = self.utg.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Latent activation plot was saved in the directory: ", path)

    def plot_trajectory_evolution(self, sample_size, plot_name = None, search = None, sample_number=10, learning_steps=50, model=None, debuge_variation=False):
        """
        Generate and save trajectory plot of the model. 

        Parameters: 
            model - model on which base the evolution will be calculated 
            array - set of sample from which  will be choosen and evaluated
            plot_name - name of the saving plot 
            model - model based on which samples will be changed 
        """
        global_history = []
        final_solutions = []
        max_fitness = 0
        if model != None: 
            encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        
        if search == None:
            search = self.search # modify encoded representation using default search function

        global_history = []
        for i in range(sample_number):
            
            arr = self.utg.initialize_solution(sample_size)
            current_solution = np.copy(arr)
            sample_history = [self.fitness_function(current_solution)]

            for k in range(learning_steps-1):
                if model == None:
                    current_solution = self.utg.flip_and_update(
                        current_solution = current_solution, 
                        search = search, 
                        debuge_variation = debuge_variation)
                else:
                    current_fitness = self.fitness_function(current_solution)
                    new_solution, new_fitness = self.utm.code_flip_decode(
                        array = current_solution, 
                        encoder = encoder, 
                        decoder = decoder, 
                        search = search,
                        debuge_variation= debuge_variation
                        )
                    if new_fitness >= current_fitness: 
                        current_solution = new_solution
                    if k == learning_steps - 2:
                        final_solutions.append(new_solution)
                fitness_to_append = self.fitness_function(current_solution)

                if max_fitness <= fitness_to_append:
                    max_fitness = fitness_to_append

                sample_history.append(fitness_to_append)
            global_history.append(np.asarray(sample_history))
        

        plt.figure()
        plt.title("Example Solution Trajectory")
        X = np.arange(learning_steps)
        for j in range(sample_number):
            plt.plot(X, global_history[j])
        plt.xlabel("epoch")
        plt.ylabel("fitness \ max_fitness")
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Trajectory evoultion plot was saved in the directory: ", path)
        else:
            plt.show()
        return max_fitness, final_solutions

    """
    def plot_global_trajectory(self, encoder, decoder, array,plot_name, epochs = 20, 
                                        learning_steps = 50, target_size = 10 ,
                                        debuge_variation = False, threshold = 5):

        loop_done = False
        fitness_history = [] 
        iteration = 0 
        while iteration <= epochs and not loop_done:
            array, trajectory_samples = self.utm.generate_trajectory_plot(
                                            encoder=encoder, decoder = decoder,
                                            array = array,
                                            learning_steps = learning_steps, 
                                            target_size = target_size,
                                            )
            fitness_history.append(trajectory_samples[:,-1])
            
            if debuge_variation:
                print(trajectory_samples[:,-1], " at iteration: ", iteration)

            if iteration > threshold and (fitness_history[-threshold] ==fitness_history[-1]).all():
                loop_done = True
                print("[INFO]: Loop was terminated after: ", iteration, " iterations, due to the lack of improvement!!!")

            iteration += 1
            

        plt.figure()
        plt.title("History Solution Trajectory after "+ str(epochs)+ " Epochs | epoch = " + str(learning_steps))
        fitness_history = np.asarray(fitness_history)
        X = np.arange(iteration)
        for i in range(iteration+1):
            plt.plot(X, fitness_history[:,i])
        plt.xlabel("epoch")
        plt.ylabel("fitness \ max_fitness")
        path = self.utg.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Global trajectory plot was saved in the directory: ", path)

        return trajectory_samples

    """
    """
    def plot_trajectory_evolution(self, model, array, plot_name, normalization_factor = 1
                                    ,target_size = 10, learning_steps = 30):  
        
        Still to improve
        Generate and save trajectory plot of the model. 

        Parameters: 
            model - model on which base the evolution will be calculated 
            array - set of sample from which  will be choosen and evaluated
            plot_name - name of the saving plot 

        Optional parameters: 
            target_size - number of tracking samples - default 10
            learning_steps - number of steps in evaluation - default 30
            
         
        encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        X = np.arange(learning_steps)
        #normalization_factor = ut.hiff_fitness(np.ones((np.shape(array)[-1],)))
        trajectory_samples = []
        plt.figure()
        plt.title("Example Solution Trajectory 2 at Evolution Step 1")
        for k in range(target_size):
            current_array = array[k]
            current_fitness = self.fitness_function(current_array)
            current_target_trajectory = []
            current_target_trajectory.append(current_fitness/normalization_factor)
            for i in range(learning_steps-1):
                output_tensor, output_array, new_fitness = self.utm.code_flip_decode(current_array, encoder, decoder)
                if new_fitness >= current_fitness:
                    current_fitness = new_fitness
                    current_array = output_array
                    current_target_trajectory.append(new_fitness/normalization_factor)
                else:
                    current_target_trajectory.append(current_target_trajectory[-1])
            trajectory_samples.append(current_target_trajectory)
            plt.plot(X, np.asarray(current_target_trajectory))
        plt.xlabel("learning step")
        plt.ylabel("fitness \ max_fitness")
        path = self.utg.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Trajectory evoultion plot was saved in the directory: ", path)
    """
    """
    def plot_evolution(encoder, decoder, array, plot_name, learning_steps = 50):
        N = np.shape(array)[-1] # size of the array 
        index = np.random.randint(N) #choose random index to flip 
        candidate_solution = array[index]# pick up random sample 
        sol_evol = [] # list to store steps of evolution 
        sol_evol.append(candidate_solution)
        current_fittnes = ut.hiff_fitness(candidate_solution)
        print(candidate_solution)
        for i in range(learning_steps-1):
            new_candidate_sol = copy.copy(candidate_solution)
            output_tensor, output_array, new_fitness = ut.code_flip_decode(new_candidate_sol, encoder, decoder)
            if new_fitness >= current_fittnes:
                candidate_solution = output_array 
                current_fittnes = new_fitness
            sol_evol.append(candidate_solution)

        tmp = np.array(sol_evol)
        plt.figure()
        plt.imshow(tmp, interpolation='nearest', cmap=cm.Greys_r)
        plt.title("Solution Development at Evolution Step 1")
        plt.xlabel("Solution variable")
        plt.ylabel("Development Step")
        plt.colorbar()
        path = ut.create_plot_path(plot_name)
        plt.savefig(path)
        print("[INFO]: Evolution plot was saved in the directory: ", path)
    """