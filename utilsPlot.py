
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import copy
import matplotlib.cm as cm
from KnapSack import KnapSack
from shallowNet.shallowNet import DenseTranspose
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import transforms


class UtilsPlot:

    def __init__(self, utilsGeneral, utilsModel):
        self.utg = utilsGeneral
        self.utm = utilsModel
        self.fitness_function = self.utg.fitness_function
        self.search = self.utg.flip

    
    def plot_weights_model(self, model,*args, plot_name=None,  show = True, decoder = False):
        """
        Visualize weights of model (based on the encoder). 
        If added hidden layers is more than 1, then the function shows
        number of hidden layers, starting from the deepest one,
        defined by the param: number_of_deep_layers_to_show

        Parameters:
            model - model which weights will be visualized 
            *args (()) - number of layers to visualize (if nothing is passed show all dense layers from encoder)
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot
            decoder (False) - visualize weights from decoder 

        Possible Erros:
            number_of_deepl_layers_to_show cannot be greater than the actual 
            number of added layers
        """
        e,d = self.utm.split_model_into_encoder_decoder(model, show = False)
        if decoder: 
            e = d
        encoder = []
        for layer in e.layers:
            if type(layer) ==type(tf.keras.layers.Dense(1)):
                encoder.append(layer)

        if len(encoder)==1:
            plt.figure(figsize=(10,10))
            plt.imshow(encoder[0].get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
            plt.xlabel("hidden")
            plt.ylabel("visible")
            plt.title("layer 0")
            plt.colorbar()
            
        else:
            if args == ():
                args = list(np.arange(len(encoder)))

            fig, axes = plt.subplots(nrows=1, ncols=len(args), constrained_layout = True, figsize=(15,15))
            index = 0
            
            for i in args:
                ax = axes[index]
                im = ax.imshow(encoder[i].get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
                ax.set_title("Layer : " + str(i))
                ax.set_xlabel("Hidden")
                ax.set_ylabel("Vissible")
                index+=1
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='5%', pad=0.45)
                fig.colorbar(im, cax = cax, orientation='horizontal')
        
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Trajectory evoultion plot was saved in the directory: ", path)
        if show:
            plt.show()


    def plot_model_loss(self, model_fit, plot_name = None, show = True):
        """
        Plot model loss history based on the tensor flow training history

        Parameters: 
            model_fit - history of model training
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot

        """
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        epochs = len(model_fit.history[list(model_fit.history.keys())[0]])
        for metrics in list(model_fit.history.keys())[1:]:
            plt.plot(np.arange(0, epochs), model_fit.history[metrics], label=metrics)
        
        plt.title("Training Loss and Accuracy\n")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        if show:
            plt.show()
        if not plot_name == None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Loss plot was saved in the directory: ", path)
        return

    


    def plot_fitness_development_phase(self, model, array, search = None, plot_name = None, show = True):
        """
        Plot fitnes development phase. 

        Parameters: 
            model - tf's model based on which we will evaluate 
            array - set of samples from which a one will be choosen and will be passed to the transfer_sample_latent_flip 
                    to see how does the model develop 
            search (None) - searching function in encoded representation (None - 2 random bits flip)
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot
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
        if show:
            plt.show()
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Fitness development phase plot was saved in the directory: ", path)


    def plot_evolution_model(self, model, sample_set, plot_name = None, search = None, learning_steps = 50,  debuge_variation = False, show = True,):
        """
        Run model evolution based on the randomly choosen solution from sample_set.
        Plot shows how solution is changed over many evaluations.  

        Parameters: 
            model - model based on which latent space search sample is modified
            sample_set - set of samples from whcih one will be choosen randomly
            search (None) - searching function in encoded representation (None - 2 random bits flip)
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            learning_steps (50) - number of sample evaluations
            show (True) - show generated plot
            learning_steps (False) - turns on debuge mode 
            debuge_variation (False) - turn on debuge mode
        """
        if search == None:
            search = self.search # modify encoded representation using default search function
        encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        N = np.shape(sample_set)[0] # size of the array 
        index = np.random.randint(N) #choose random index to flip 
        candidate_solution = sample_set[index]# pick up random sample 
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
        if show: 
            plt.show()
        
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Evolution model plot was saved in the directory: ", path)

    def plot_latent_acitvation(self, model, sample_set, plot_name = None, validation_set_size = 20, show = True):
        """
        Plot latent activation based on many samples: 

        Parameters: 
            model - model which latent distribution will be displayed 
            sample_set - set of sample to evaluate 
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            validation_set_size (20) - number of samples used to create plot
        """
        e = self.utm.split_model_into_encoder_decoder(model)[0]
        predict = e.predict(sample_set[:validation_set_size])
        N = np.shape(predict)[1]
        for i in range(validation_set_size):
            plt.plot(np.arange(N), predict[np.random.randint(validation_set_size)], 'o',color = 'black')
        plt.title("Latent Activation")
        plt.xlabel("Node #")
        plt.ylabel("Activation value")
        
        if show:
            plt.show()

        if not plot_name == None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Latent activation plot was saved in the directory: ", path)

    def plot_trajectory_evolution(self, sample_size, plot_name = None, search = None, sample_number=10, learning_steps=50, model=None, debuge_variation=False, show = True):
        """
        Generate and save trajectory plot of the model. 

        Parameters:
            szmple_size - size of the sample (lenght of the knapsack solution) 
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            search (None) - searching function in encoded representation (None - 2 random bits flip)
            sample_numbers (10) - numbers of samples to visualize on plot
            learning_steps (50) - number of evolution steps
            model (None) - model based on which samples are evaluated (if None - optimize non compressed samples)
            debug_variation (False) - turn on debuge mode
        Returns: 
            maximal_possible fitness (int), 
            list of improved solutions
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
                        final_solutions.append(current_solution)
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
        if show:
            plt.show()
        return max_fitness, final_solutions


    def plot_correlation_matrix(self, model, activation=1, background_activation =-1, threshold = None, plot_name = None,show = True):
        """
        Visualize "correlation" matrix (masure of overlapping of encoded
        representation comming from separate neuron activarions)

        Correlation matrix -  It compares encoded representation comming from 
            different single bits activation (sample with i-th bit set to activation's value and 
            all others set to back_ground's value). "Correlation" is calculated as RMSE with 
            some threshodl value to alliviate noise influence. 

        Parameters:
            model - model based on which we get encoded form 
            activation (1) - value of separate bit activation 
            background_activation (-1) - value of activation of background neurons in encoded phase
            threshold (None) - threshold (int) saying above which level we count overlaping if(None - take std as threshold)
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot
        """
        plt.figure()
        res = self.utm.correlation_matrix(model, activation=activation, background_activation = background_activation, threshold = threshold)
        plt.imshow(res, interpolation='nearest')
        if threshold==None:
            threshold = "std"
        plt.title("Correlation Matrix:\nlatent space overlapping: RMSE threshold {}".format(threshold))
        plt.xlabel("Bit position")
        plt.ylabel("Bit position")
        plt.colorbar()

        if show:
            plt.show()

        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Trajectory evoultion plot was saved in the directory: ", path)

        return plt

    def plot_set_probability_and_values(self, sample_set, sort = False, plot_name = None, show = True):
        """
        Visualize bit's probability distribution. Calculate i-th bit probability of activation. 
        
        Parameters:
            sample_set - sample set based on which we calculate distribution 
            sort (False) - sort distribution according to the probability of activation 
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot

        """
        sample_size = np.shape(sample_set)[-1]
        arr = np.zeros(sample_size)
        set_size = np.shape(sample_set)[0]
        glob = {}
        index = 0
        for i in sample_set:
            arr +=np.where(i<0,0,1)
            
        arr = arr/set_size

        for i in range(len(arr)):
            glob[i] = arr[index]
            index +=1
        
        if sort:
            arr.sort()
        
        pos = {k : v for k, v in sorted(glob.items(), key = lambda item : item[1])}
        
        arr2 = []
        for i in pos:
            tmp = np.zeros(100)
            tmp[i] = 1
            arr2.append(np.dot(self.utg.knapSack.P, tmp))
        
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout = True)
        fig.suptitle('Training Set Anomaly', fontsize=16)
        axes[0].bar(np.arange(sample_size), arr, alpha = 0.5)
        axes[0].set_title("Input bit\n activation distribution")
        axes[0].set_ylabel("Probability")
        axes[0].set_xlabel("Index")
        axes[1].bar(np.arange(sample_size), arr2, alpha = 0.5) 
        axes[1].set_title("Single bit contribution:\n sorted by index probability")
        axes[1].set_ylabel("Contribution")
        axes[1].set_xlabel("Index")
        if show:
            plt.show()
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
            print("[INFO]: Trajectory evoultion plot was saved in the directory: ", path)

    def plot_latent_activation_distribution(self, sample_set, model, probability = True, sort = False, model_name = "", plot_name = None, show = True):
        """
        Plot distribution of encoded bits' activation being positive or negative. 
        Add mean and std on the plot. 

        Paramters:
            sample_set - sample set based on which we calculate distribution 
            model - tf's model based on which samples from sample_set are encoded 
            probability (True) - False - show positive and negative porbabilities separetely, True - stack both positive and negative probability
            sort (False) - sort distribution according to the probability of activation 
            model_name ("") - name of the model which will be added to the plot title
            plot_name (None) - save plot under the given name in the directory saved plots (if None - not save)
            show (True) - show generated plot 
        Returns: 
            negative distribution, 
            positive distribution, 
            partitioned samples belonging to each subset (result of partition funtion from UtilsGeneral)
        """

        def AllPositivesToZero(sol):
            ConvertSol = np.copy(sol)
            if probability:
                return  np.where(sol > 0.0, 1, 0)
            ConvertSol[sol >= 0] = 0
            return ConvertSol
        def AllNegativesToZero(sol):
            ConvertSol = np.copy(sol)
            if probability:
                return np.where(sol > 0.0, 0, 1)
            ConvertSol[sol < 0] = 0
            return ConvertSol
        
        
        e, d = self.utm.split_model_into_encoder_decoder(model)
        res_pos = []
        res_neg = []
        latent_size = np.shape(e.layers[-1].get_weights()[0])[-1]
        final_pos = np.zeros(latent_size)
        final_neg = np.zeros(latent_size)
        if len(np.shape(sample_set))== 3: 
            sample_set  = sample_set[0]
        for i in sample_set:
            res_pos.append(AllNegativesToZero(self.utm.code(i,e)))
            res_neg.append(AllPositivesToZero(self.utm.code(i,e)))

        for i in res_pos:
            final_pos += i
        for i in res_neg:
            final_neg += i

        final_pos = final_pos/len(sample_set)
        if probability:
            final_neg = final_neg/len(sample_set) 
        else: 
            final_neg = final_neg/len(sample_set) *(-1)

        if sort: 
            final_neg.sort()
            final_pos[::-1].sort()

        partition_, mean, std = self.utg.partition(final_pos, numbers_of_partition=2)
        partition_size = len(partition_)
        std_lines = []
        set_lines = []
        for arr in partition_:
            set_lines.append(arr[-1])
        set_lines.pop(0)

        for i in range(1,int(partition_size/2)):
            std_lines.append(i*std + mean)
            std_lines.append(-i*std + mean)


        plt.figure()
        if probability:
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(90)
            plt.bar(np.arange(len(final_pos)),-final_pos, color='blue', alpha = 0.7, label = "+p",transform = rot + base)
            plt.bar(range(len(final_neg)), -final_neg, bottom = -np.array(final_pos), color = "red", alpha = 0.7, label = "-p", transform = rot + base)
            if sort:
                for xc in std_lines:
                    plt.axvline(x = xc , color ="black", linestyle = '--')
                plt.axvline(x = mean, color = "black", linestyle = '--')
                for yc in set_lines: 
                    plt.axhline(y = yc, color = "black", linestyle = '-')
                plt.title("+/-probability distributioin {}: stacked + sorted".format(model_name))
            else:
                plt.title("+/-probability distributioin {}: stacked".format(model_name))
            plt.xlabel("Probability")
            plt.ylabel("Bit positions")
            plt.legend()
        else:
            plt.bar(np.arange(len(final_pos)),final_pos,label = "Positive values", alpha = 0.5, color = "red")
            plt.bar(np.arange(len(final_neg)),final_neg, label = "Negative values * (-1)", alpha = 0.5, color = "blue")
            plt.title("+/- average bit values distributioin {}:".format(model_name))
            plt.ylabel("Average Value")
            plt.xlabel("Bit positions")
            plt.legend()
        if plot_name != None:
            path = self.utg.create_plot_path(plot_name)
            plt.savefig(path)
        if show: 
            plt.show()
        return final_pos, final_neg, partition_
            
    def plot_latent_activation(self, model, index_to_split = None, title ="", activation=1, background_activation =-1, column_sort = True, unit_sort=True, index_sort = False, log_conversion = True, show = True):
        """
        Show plots associated with weight matrix measurements. 

        Parameters: 
            model - tf's model to evaluate 
            index_to_split (None) - index of layer around which model will be splited (None - around bottle neck)
            Filters:
                column_sort (True) -  sort weight by columns (True) | sort weights by rows (False)
                unit_sort (True) - sort by the magnitude (absolut value) of weights (True) | else sort by signs (False)
                index_sort (False) - shuffle columns according to the probability of encoded neuron actiavtion (True) | no sort (False)
                log_conversion (False) - convert values of weights to discrete log values (True) | work on the linear scale (False)
            title ("") - title to show on the plot title (advice - model name) + parameters 
            activation (1) - activation of a single bit 
            background_activation (-1) - activation of the remaining bits
            show (True) - show generated sample
        
        """
        bit_activation, weights_matrix, *masks  = self.utm.weights_matrix_sorting(model, index_to_split, activation, background_activation,column_sort = column_sort, unit_sort = unit_sort, index_sort = index_sort, log_conversion = log_conversion)

        title = title + " - activation: " +str(activation)+" background: "+ str(background_activation) + "\n "
        if column_sort:
            title += "column sort, "
        else:
            title += "row sort, "
        if index_sort:
            title += "index_sort"
        else:
            title += "no index sort"
        if log_conversion:
            title += " log scale"
        else:
            title += " linear scale"
        
        if index_sort:
            fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout = True, figsize=(10,10))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout = True, figsize=(10,10))
        fig.suptitle(title, fontsize=16)
        if unit_sort:
            pa = axes[0].imshow(masks[0],interpolation='nearest',cmap=cm.Greys_r)
            cba = fig.colorbar(pa, ax = axes[0], location = "left")
            cba.set_label('Magnitude of Absolute Activation')
            axes[0].set_title("Unit Sort")
            axes[0].set_xlabel("Hidden")
        else:
            pa = axes[0].imshow(masks[0],interpolation='nearest',cmap=cm.Blues)
            cba = fig.colorbar(pa, ax = axes[0], location = "left")
            pb = axes[0].imshow(masks[1],interpolation='nearest',cmap=cm.Reds)
            cbb = fig.colorbar(pb, ax = axes[0], location = "right")
            cba.set_label('Negative')
            cbb.set_label('Positive')
            axes[0].set_title("Split Sort")
            axes[0].set_xlabel("Hidden")
        axes[0].set_aspect('auto')

        
        axes[1].imshow(weights_matrix, interpolation='nearest', cmap=cm.Greys_r)
        axes[1].set_title("Weights")
        axes[1].set_ylabel("Visible")
        axes[1].set_xlabel("Hidden")
        axes[1].set_aspect('auto')
        
        if index_sort:
            axes[2].bar(np.arange(len(bit_activation)), bit_activation, align = "center", alpha = 0.5)
            axes[2].set_title("Bit Activation")
            axes[2].set_ylabel("Activation Probability")
            axes[2].set_xlabel("Bit index")
            axes[2].set_aspect('auto')
        if show:
            plt.show()

        return
    




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