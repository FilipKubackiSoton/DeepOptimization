import numpy as np

# from . import Helper


class KnapSack:
    def __init__(self, InstanceName, Debug=True):

        self.Name = "KnapSack"
        self.InstanceName = InstanceName
        Beasly = False
        if InstanceName[0:3] == "sen":
            ProblemFolder = "MKP/sac94/sento/"
            FileName = InstanceName + ".dat"
        else:
            Beasly = True
            ProblemFolder = "MKP/chubeas/"
            ResFile = ProblemFolder + "BestKnownResults.txt"
            InstanceName = InstanceName.split("_")

            ProblemFolder += "OR{}x{}/".format(InstanceName[1], InstanceName[0])
            FileName = "OR{}x{}-0.{}_{}.dat".format(
                InstanceName[1], InstanceName[0], InstanceName[2], InstanceName[3]
            )
        print(ProblemFolder)
        print(FileName)

        f = open(ProblemFolder + FileName)
        nextLine = f.readline().strip().split()
        self.Size = int(nextLine[0])
        self.nCoefs = int(nextLine[1])
        self.GlobalOptimumFitness = float(nextLine[2])
        # print('Size = {}, nCoefs = {}, Glbl = {}'.format(self.Size, self.nCoefs, self.GlobalOptimumFitness))
        self.P = np.zeros(self.Size)  # array of proffits - value of each item
        self.R = np.zeros((self.nCoefs, self.Size))  # matrix of constrains
        self.b = np.zeros(self.nCoefs)  # array of capacities for d dimensions array([11927., 13727., 11551., 13056., 13460.])



        CountV = 0

        # read data and feed profit array
        while CountV < self.Size:
            nextLine = f.readline().strip()  # read the line
            AddArray = np.fromstring(nextLine, dtype=float, sep=" ")  # axtract data
            self.P[
                CountV : (CountV + len(AddArray))
            ] = AddArray  # save date on the right possition
            CountV += len(AddArray)  # incremet index
        # print('Coefficients = {}'.format(self.P))

        # read data and feed constrains matrix
        for l in range(
            self.nCoefs
        ):  # go through the number of coefficients = go through rows
            CountV = 0  # reset the counter
            while CountV < self.Size:  # fill the one row
                nextLine = f.readline().strip()  # read the line
                AddArray = np.fromstring(
                    nextLine, dtype=float, sep=" "
                )  # split the line to extract data
                self.R[
                    l, CountV : (CountV + len(AddArray))
                ] = AddArray  # put the data in the right row
                CountV += len(AddArray)  # increment coursor

        self.R = np.transpose(
            self.R
        )  # transpose the constrain matrix, it is done to achieve the functionality to:
        # self.R * variables = vector of constrains values - we can check easily if
        # restrictions are fulfilled
        CountV = 0
        # read data and feed array of capacity
        while CountV < self.nCoefs:  # go through coeficients
            nextLine = f.readline().strip()  # read the line
            AddArray = np.fromstring(nextLine, dtype=float, sep=" ")  # extract data
            self.b[
                CountV : (CountV + len(AddArray))
            ] = AddArray  # put the data on the right place
            CountV += len(AddArray)  # increment counter
        f.close()  # clos the filse stream

        if Beasly:
            # Get Beasly best results.
           # f_res = open(ResFile)
            instance_indx = int(InstanceName[3]) - 1
            if int(InstanceName[2]) == 50:
                instance_indx += 10
            if int(InstanceName[2]) == 75:
                instance_indx += 20

            BestResult = 0
            res_indx = "{}.{}-{:02d}".format(
                InstanceName[1], InstanceName[0], instance_indx
            )
            """
            for l in range(272):
                nextLine = f_res.readline().strip().split()
                if nextLine[0] == res_indx:
                    BestResult = nextLine[1]
                    break
            # print(res_indx)
            """
        else:
            BestResult = self.CalcMax(np.ones(self.Size))

        self.GlobalOptimumFitness = int(BestResult)

        self.InitFitness = 0.0
        self.Debug = Debug
        self.ProblemDepth = 1
        self.SolutionShape = (1, self.Size)
        self.Properties = ("Binary", "1D", "MKP", "Constraints")
        self.FunctionEvaluations = 0

        self.maxp = np.max(self.P)
        self.Map = {}
        self.Map["R"] = np.copy(self.R)

        self.CalcUtility()
        self.GlobalOptimum()

        self.zero_out = False
        self.GlobalFoundIn = 0
        """
        print("self.P", self.P)
        print("self.b", self.b)
        print("self.GlobalOptimumFitness", self.GlobalOptimumFitness)
        print("selfInitFitness.", self.InitFitness)
        print("self.maxp", self.maxp)
        print("self.Map", self.Map)
        """
        return

    def SetSolution(self):
        # Initilise solution specific to this problem
        # Assignment = np.random.choice((-1,1),size=self.Size)
        Assignment = np.zeros(self.Size) - 1
        return Assignment

    def Search(self, Solution, indx):
        """
        I do not know what this method is used for. 
        I assume it performs random bit flip 
        """
        Solution.NewSol[indx] *= -1
        Other_indx = np.random.randint(self.Size)
        Solution.NewSol[Other_indx] *= -1
        return

    def FitnessEffect(self, newf, oldf):
        # return: -1 = negative change, 0 = neutral changne, +1 = beneficial change
        # Minimisation problem
        if newf > oldf:
            return 1
        if newf == oldf:
            return 0
        if newf < oldf:
            return -1

    def InterpretSample(self, Generated):
        # Generate a solution probabilistic (model-informed generation)
        if len(Generated) == 1:
            Generated = Generated[0]
        if self.zero_out:
            Sample = np.random.uniform(0, 1, size=Generated.shape)  # [0, 1]
        else:
            Sample = np.random.uniform(-1, 1, size=Generated.shape)  # [-1, 1]
        NewSol = (
            np.zeros_like(Generated) - 1
        )  # create array of -1 in the shape of Generated's array

        # create random minus binary array {-1, 1}
        NewSol[
            Generated >= Sample
        ] = 1  # if generated if >= Sample then set the filfilling element to 1
        return NewSol

    def Interpret(self, Generated):
        """ 
        Interperate the output from the model, specific for this problem

        In other words, extract solution via minus binarization:
        (like binarization around threshold but with -1 and 1) of the generated solution.
        
        """
        # Model-informed variation
        if len(Generated) == 1:
            Generated = Generated[0]

        if self.zero_out:
            Threshold = 0.5
        else:
            Threshold = 0.0
        Solution = np.zeros(self.Size)
        Solution[
            Generated > Threshold
        ] = 1  # discretize Generated around threshold to the [-1, 1]
        Solution[
            Generated < Threshold
        ] = -1  # discretize Generated around threshold to the [-1, 1]
        return Solution

    def CalcMax(self, st):
        """
        max possible value is returned when (st = array of ones)
        """
        Val = np.dot(self.P, st)
        return Val

    def CheckConstraints(self, st):
        # check constraint used by plotting routines
        return self.CalcOverfilled(st)

    def CalcOverfilled(self, st):
        Val = np.dot(st, self.R)  # values of constrains with given variables
        # print(Val > self.b)
        # print(Val)

        OverFilled = Val > self.b  # if all values of val > self.b then True
        # array[true] -> array, array[false] -> []
        sumOverFill = np.sum(OverFilled)  # calculate sum of the elements in the aray
        AbsOverFill = np.sum(
            np.abs(Val[OverFilled] - self.b[OverFilled])
        )  # calculate overfitt if constrains are not fulfilled
        # if constrains are filfilled then it's equal to 0

        return sumOverFill, AbsOverFill

    def Fitness(self, state):
        # Calculate the fitness of a solution
        # f_st = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,
        # 1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,
        # 1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,
        # -1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,
        # 1,-1,-1,-1])

        # print(self.b)

        st = np.copy(state)
        st[st == -1] = 0  # set all -1 to 0
        # st = np.copy(f_st)
        # f_st[f_st==-1] = 0

        # for i in range(self.Size):
        # Check constraints
        # st = np.copy(f_st)
        # st[i] = 0
        SubjectTo, AbsOver = self.CalcOverfilled(st)  # calculate overfill

        if SubjectTo > 0:  # if overfill occured
            # illegal
            total_fitness = -(AbsOver * SubjectTo)  # calculate loss = total overfill
        else:  # calculate sume of the used parameters
            total_fitness = self.CalcMax(st)  # calculae parameters * st

            # print(SubjectTo,AbsOver,total_fitness)
        return total_fitness

    def SolutionRepresentation(self, Solution):
        """
        Shows how much Solution and global Solution are ovelaped.

        E.g: 
        self.GlobalOptimumSolution = np.array([1, 2, 3, 4, 5])
        Solution = np.array([1, 3, 2, 4, 6])

        then: 

        Representation: {1: array([1., 0., 0., 1., 0.])}
        """
        Representation = {}
        Representation[1] = np.zeros(
            self.Size
        )  # fill the secound dimension (first non-zero dimension) with 0's
        Representation[1][
            Solution == self.GlobalOptimumSolution
        ] = 1  # map overlapping part of soultions with GlobalOptimumSolution as 1's in the 2nd dimension
        return Representation

    def interp_sol(self, sol_traj, single=False):
        """
        Interpret solution. 
        If single = False: return single array holding trajectory solution 
        if single = True: retun binary array with 1 at the positon of the global optimum 

        Params: 
            sol_traj - solution trajectory - array/ vector in the solution space 
        """

        if single:
            devo_steps = 1  # it will cose the loop to execute onece
            interp_trajectory = np.zeros(self.Size)  # create array of 0's
        else:
            devo_steps = sol_traj.shape[
                0
            ]  # execute loop as many times as potentail solutions
            interp_trajectory = np.zeros(
                (devo_steps, self.Size)
            )  # 2D array holding potential solutions
        for d in range(devo_steps):
            interp_trajectory[
                d, sol_traj == self.GlobalOptimumSolution
            ] = 1  # if single = True: return array[1,self.Size] holding solution
            # else: return array[devo_steps, self.Size] - only one row will hold solution
            # other rows will be empty

        return interp_trajectory

    def GlobalOptimum(self):
        """
        Calculate how much 
        """
        self.N_GlobalAnswers = 1
        self.GlobalAnswers = np.ones(self.Size)
        self.GlobalAnswers[
            self.GlobalAnswers == 0
        ] = -1  # i think this line does nothing because we always compare 1 to 0
        self.GlobalOptimumSolution = (
            self.GlobalAnswers
        )  # assign new GlobalOptimum Solution
        if self.Debug:
            print("Best Objective Value: {}".format(self.GlobalOptimumFitness))

        return

    def CalcUtility(self):
        """
        I do not understand why do we do that??? 
        """
        self.Utility = np.zeros((self.Size, 2))
        for i in range(self.Size):
            self.Utility[i, 0] = i  # fill column 0 with consequtive numbers =
            # = vector of size self.Size with consequtve numbers [0, self.Size -1 ]

            self.Utility[i, 1] = self.P[i] / np.sum(self.R[i, :])
            # fill column 1 with ration of param[i] / Sum(Restriction[i:])

        i = np.lexsort(
            (self.Utility[:, 0], self.Utility[:, 1])
        )  # sort by column 0, then by column 1
        self.Utility = np.copy(self.Utility[i])  # copy util in the right order

        return 

    def SolToTrain(self, Sol):
        """
        Solution are in the form of binary array, while in the
        training phase when we want to know contribution of 
        each components we want to convert 0's to -1's, to
        indicate negative contributon. 
        """
        ConvertSol = np.copy(Sol)
        ConvertSol[Sol == 0] = -1
        
        """ 
        Orginal code 
        if self.zero_out:
            ConvertSol[Sol == -1] = 0
            # ConvertSol[ConvertSol==0] = 0.499
        # else:
        # ConvertSol[ConvertSol==-1] = -0.001
        """
        return ConvertSol

    def CountChange(self, Old, New):
        change = np.sum(Old != New)

        return change

    def __del__(self):
        print("Destructor for Problem")
        return
