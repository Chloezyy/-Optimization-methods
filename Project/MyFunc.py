import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

'''This is the definition area for the various functions'''
def Sphere(x):
    return np.sum(x*x)

def Rosenbrock(X):
    X_len = len(X)
    O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    return O 

def Schwefel(X):
    O=np.sum(np.abs(X))+np.prod(np.abs(X))
    return O 

def Schwefel2(X):
    O=0
    for i in range(len(X)):
        O=O+np.square(np.sum(X[0:i+1]))
    return O

def Ackley(X):
    dim=len(X)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_1 = -a * np.exp(-b * np.sqrt(np.sum(X ** 2) / dim))
    sum_2 = np.exp(np.sum(np.cos(c * X)) / dim)
    O= sum_1 - sum_2 + a + np.exp(1)
    return O 

def GRastrigin(X):
    dim=len(X)
    O=np.sum(X**2-10*np.cos(2*np.pi*X))+10*dim
    return O 

def Oned(X):
    return x*x

'''Here is the definition area of various functions (the minimum value of these functions is 0, can be determined uniformly)'''
'''The defined function is linked here with the dictionary name and function'''

func_dict = {'Sphere':Sphere,'Rosenbrock':Rosenbrock,'Schwefel':Schwefel,'Schwefel2':Schwefel2,'Ackley':Ackley,'GRastrigin':GRastrigin,'Oned':Oned}

class Func():
    def __init__(self,problem_name,dim,x_range):
        self.x_range = np.array(x_range)
        self.prb_dim = dim
        self.prb_name = problem_name
        self.prb_func = func_dict[self.prb_name]
        self.x_rem = []

    def get_y(self,x):
        '''With this function you can input x to get y'''
        x = np.array(x)
        #if len(x) != self.prb_dim:
         #   print('X with wrong dim')
          #  return 0
        self.x_rem.append(x)
        return self.prb_func(x)

    def draw_func(self,show_finding=False):
        '''This function displays the image. show_finding=True shows the search process'''
        '''Supports two-dimensional and one-dimensional fucntions'''
        if self.prb_dim == 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            x1 = np.arange(self.x_range[0][0],self.x_range[1][0],0.1)
            x2 = np.arange(self.x_range[0][1],self.x_range[1][1],0.1)
            X1,X2 = np.meshgrid(x1,x2)
            Y = np.zeros_like(X1)
            for i in range(len(X1)):
                for j in range(len(X1[0])):
                    y = self.prb_func(np.array([X1[i,j],X2[i,j]]))
                    Y[i,j] = y
            ax.plot_surface(X1,X2,Y,cmap='rainbow',alpha=0.6)
            if show_finding:
                rem_num = len(self.x_rem)
                X1 = np.zeros(rem_num)
                X2 = np.zeros(rem_num)
                Y = np.zeros(rem_num)
                for i in range(rem_num):
                    y = self.prb_func(np.array(self.x_rem[i]))
                    X1[i] = self.x_rem[i][0]
                    X2[i] = self.x_rem[i][1]
                    Y[i] = y
                ax.view_init(elev=70,azim=20)
                ax.scatter(X1,X2,Y,c='r')
        if self.prb_dim == 1:
            fig = plt.figure()
            X = np.arange(self.x_range[0][0],self.x_range[1][0],0.1)
            Y = np.zeros_like(X)
            for i in range(len(Y)):
                Y[i] = self.prb_func(X[i])
            plt.plot(X,Y,c='b')
            if show_finding:
                rem_num = len(self.x_rem)
                X = np.zeros(rem_num)
                Y = np.zeros(rem_num)
                for i in range(rem_num):
                    y = self.prb_func(np.array(self.x_rem[i]))
                    X[i] = self.x_rem[i]
                    Y[i] = y
                plt.scatter(X,Y,c='r')
        plt.show()

    def get_min(self):
        return 0

    def clean_rem(self):
        '''Clean up the search process and do this every time you do a new search'''
        self.x_rem = []


if __name__ == '__main__':
    x_range = [[-10,-10],[10,10]]
    test_fuc = Func(0,2,x_range)
    print(test_fuc.get_y([0.5,0.5]))
    print(test_fuc.get_y([0.9,0.5]))
    print(test_fuc.get_y([0.3,0.5]))
    test_fuc.draw_func(show_finding=True)
    plt.show()
