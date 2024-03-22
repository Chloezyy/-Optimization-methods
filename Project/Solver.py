import numpy as np
import matplotlib.pyplot as plt

class Solver:
    def __init__(self,solver_name,problem,max_step=1000,tol=1e-4) -> None:
        self.solver_name_list = ['Golden','Powell','DownHill','Gradient','SA']#check that if the input method name is our type
        self.solver_name = solver_name
        self.problem = problem
        self.max_step = max_step #Maximum number of iteration rounds
        self.tol = tol #Set the minimum value considered（If satisfied, the iteration can be terminated）
        self.show_process = False
        self.t = 0

    def solve(self):
        '''Call the method to solve the problem'''
        self.problem.clean_rem()
        if self.solver_name in self.solver_name_list:
            # Firstly, determine whether the method name we have
            if self.solver_name == 'Golden':
                ans = self.golden()
            elif self.solver_name == 'Powell':
                ans = self.powell()
            elif self.solver_name == 'DownHill':
                ans = self.downhill()
            elif self.solver_name == 'Gradient':
                ans = self.gradient()
            elif self.solver_name == 'SA':
                ans = self.SA()
            self.t = len(self.problem.x_rem)
            y = self.problem.get_y(ans)
            ans = str(ans)
            # Get the answers and output
            print('{} Iterations: {:d} times'.format(self.solver_name,self.t)) #Output the method name and the number of iterations
            print('Final x: {:s}'.format(ans)) #Output X
            print('Final f(x): {:.5f}'.format(y)) #output y
        else:
            print('Get a wrong solver name, please change into one of:')
            print(self.solver_name_list)

    def golden(self):
        if self.problem.prb_dim != 1:
            # The golden section solves only one dimension of the problem
            print('Problem Dim is greater than 1, Golden method can not solve it.')
        ratio = (3 - np.sqrt(5))/2
        a = 0
        b = 1
        x1 = a+ratio*(b-a)
        x2 = b-x1+a
        f1 = self.problem.get_y[(x1)]
        f2 = self.problem.get_y[(x2)]
        for t in range(self.max_step):
            if f1>f2:
                a = x1
                if b-a < self.tol:
                    break
                else:
                    x1 = x2
                    f1 = f2
                    x2 = a+(1-ratio)*(b-a)
                    f2 = self.problem.get_y(x2)
            else:
                b = x2
                if b-a < self.tol:
                    break
                else:
                    x2 = x1
                    f2 = f1
                    x1 = a+ratio*(b-a)
                    f1 = self.problem.get_y(x1)
        final_x = (a+b)/2
        return final_x

    def powell(self):
        x0 = 10*np.random.random(self.problem.prb_dim)
        dir_list = []
        for i in range(self.problem.prb_dim):
            dir_list.append(np.zeros(self.problem.prb_dim))
            dir_list[-1][i] = 1
        for t in range(self.max_step):
            dir = np.sum(dir_list,axis=0)
            '''determine the search interval'''
            h_step = 0.1*dir
            a = x0
            c = x0+h_step
            if self.problem.get_y(a) > self.problem.get_y(c):
                b = c+h_step
                for _ in range(20):
                    if self.problem.get_y(c) < self.problem.get_y(b):
                        break
                    else:
                        h_step = 2*h_step
                        a = c
                        c = b
                        b = c+h_step
            else:
                h_step = -h_step
                a,c = c,a
                b = c+h_step
                for _ in range(20):
                    if self.problem.get_y(c) < self.problem.get_y(b):
                        break
                    else:
                        h_step = 2*h_step
                        a = c
                        c = b
                        b = c+h_step
            '''The golden section method is used to calculate the optimal solution of the current direction'''
            ratio = (3 - np.sqrt(5))/2
            x1 = a+ratio*(b-a)
            x2 = b-x1+a
            f1 = self.problem.get_y(x1)
            f2 = self.problem.get_y(x2)
            for t in range(self.max_step):
                if f1>f2:
                    a = x1
                    if np.linalg.norm(b-a) < self.tol:
                        break
                    else:
                        x1 = x2
                        f1 = f2
                        x2 = a+(1-ratio)*(b-a)
                        f2 = self.problem.get_y(x2)
                else:
                    b = x2
                    if np.linalg.norm(b-a) < self.tol:
                        break
                    else:
                        x2 = x1
                        f2 = f1
                        x1 = a+ratio*(b-a)
                        f1 = self.problem.get_y(x1)
            new_dir = (a+b)/2-x0
            x0 = (a+b)/2
            old_dir = dir_list.pop(0)
            if np.linalg.norm(new_dir-old_dir)<1e-10:
                break
            dir_list.append(new_dir)
        return x0

    def downhill(self):
        vertice = np.zeros((self.problem.prb_dim+1,self.problem.prb_dim))
        vertice_y = np.zeros(self.problem.prb_dim+1)
        x0 = 10*np.random.random(self.problem.prb_dim)
        vertice[0] = x0
        vertice_y[0] = self.problem.get_y(vertice[0])
        for i in range(self.problem.prb_dim):
            vertice[i+1] = x0
            vertice[i+1][i] += 3
            vertice_y[i+1] = self.problem.get_y(vertice[i])
        for t in range(self.max_step):
            v_mean = np.mean(vertice,axis=0)
            h = np.argmax(vertice_y)
            vh = vertice[h]
            vr = v_mean+1.5*(v_mean-vh)
            if np.linalg.norm(vr-v_mean) < 1e-10:
                break
            fvr = self.problem.get_y(vr)
            if fvr < np.min(vertice_y):
                ve = v_mean+2*(vr-v_mean)
                fve = self.problem.get_y(ve)
                if fve < np.min(vertice_y):
                    vertice[h] = ve
                    vertice_y[h] = fve
                else:
                    vertice[h] = vr
                    vertice_y[h] = fvr
            else:
                if fvr >= np.sort(vertice_y)[1]:
                    vc = v_mean+0.5*(vr-v_mean)
                    fvc = self.problem.get_y(vc)
                    if fvc < np.max(vertice_y):
                        vertice[h] = vc
                        vertice_y[h] = fvc
                    else:
                        vl = vertice[np.argmin(vertice_y)]
                        for i in range(self.problem.prb_dim+1):
                            vertice[i] = (vertice[i]+vl)/2
                            vertice_y[i] = self.problem.get_y(vertice[i])
                else:
                    vc = v_mean+0.5*(vr-v_mean)
                    fvc = self.problem.get_y(vc)
                    if fvc < np.max(vertice_y):
                        vertice[h] = vc
                        vertice_y[h] = fvc
                    else:
                        vl = vertice[np.argmin(vertice_y)]
                        for i in range(self.problem.prb_dim+1):
                            vertice[i] = (vertice[i]+vl)/2
                            vertice_y[i] = self.problem.get_y(vertice[i])
        v_mean = np.mean(vertice,axis=0)
        return v_mean

    def gradient(self,learning_rate=0.1,delta=1e-8):
        x0 = 10*np.random.random(self.problem.prb_dim)
        y = self.problem.get_y(x0)
        for t in range(self.max_step):
            '''Solving the gradient'''
            g = np.zeros(self.problem.prb_dim)
            for i in range(self.problem.prb_dim):
                x1 = np.copy(x0)
                x1[i] += delta
                x2 = np.copy(x0)
                x2[i] -= delta
                g[i] = (self.problem.get_y(x1)-self.problem.get_y(x2))/(2*delta)
            if np.linalg.norm(g) < 1e-10:
                # If the gradient is small that means we're approaching the end point
                break
            x0 = x0-g*learning_rate
            y = self.problem.get_y(x0)
        return x0

    def SA(self):
        '''Simulated annealing method'''
        x0 = 10*np.random.random(self.problem.prb_dim)
        T = 100
        fx = self.problem.get_y(x0)
        while T > 0.01:
            '''Generate a new solution, in this case some dimension of x is updated'''
            x_new = np.copy(x0)
            x_new[np.random.randint(self.problem.prb_dim)] = 10*T/100*np.random.random()
            fx_new = self.problem.get_y(x_new)
            '''determine whether the new solution can be accepted'''
            dE = fx_new - fx
            if dE < 0:
                x0 = x_new
                fx = fx_new
            else:
                p = np.exp(-dE/T) 
                if p > np.random.random():
                    x0 = x_new
                    fx = fx_new
            T = T*0.99
        return x0

if __name__ == '__main__':
    pass
