import numpy as np 
import matplotlib.pyplot as plt
import math 
import cv2

class HIFF(object):
    def __init__(self, array_size, step_size):
        if not math.log2(array_size).is_integer():
            raise ValueError("Array size must be power of 2.")
        self.levels = int(math.log2(array_size))
        self.size = array_size
        self.step_size = step_size

    def f_null_to_one(self, val):
        if val == 1:
            return 1
        else:
            return 0

    def f_null_to_zero(self, val):
        if val == 0:
            return 1
        else:
            return 0

    def t(self, left, right):
        if left == 1 and right == 1:
            return 1
        elif left == 0 and right == 0:
            return 0
        else:
            return None 

    def __bin_array(self, K, N):
        arr = np.zeros(N)
        arr[:K]  = 1
        return arr

    def value(self, array, fitnes_function):
        sum =0
        return self.val_recursive(array, 0,  sum, fitnes_function)

    def val_recursive(self, array, flor, sum, fitnes_function):
        if flor > self.levels:
            return sum
        arr = []
        power = 2 ** flor
        for i in range(0,2**(self.levels - flor)-1,2):
            arr.append(self.t(array[i], array[i+1]))
            sum += (fitnes_function(array[i]) + fitnes_function(array[i+1]))* power
        return self.val_recursive(arr, flor + 1, sum, fitnes_function)

    def array(self):
        X = []
        Y_null_to_one = []
        Y_null_to_zero = []
        for index in range(0, self.size + 1, self.step_size):
            X.append(index)
            Y_null_to_one.append(self.value(self.__bin_array(index, self.size), self.f_null_to_one))
            Y_null_to_zero.append(self.value(self.__bin_array(index, self.size), self.f_null_to_zero))
        return X, Y_null_to_one, Y_null_to_zero

    def plot(self):
        X, Y_null_to_one, Y_null_to_zero = self.array()       
        plt.plot(X, Y_null_to_one, 'r--')
        plt.plot(X, Y_null_to_zero, 'b--')
        plt.plot(X, np.add(Y_null_to_one, Y_null_to_zero), 'g-')
        plt.legend(("all null to all 1s", "all null to all 0s", "all 0s to all 1s"))
        plt.show()
        plt.savefig("Richard's plot H-IFF.png")
    
h = HIFF(64, 4)
h.plot()
            