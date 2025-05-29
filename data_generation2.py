#!pip install tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import colorednoise as cn

class Binary_Telegraph_Process:
    """
    Инициализация телеграфного процесса
    :param size: размер выборки
    :param f: функция ошибки
    :param p1: матожидание ошибки в случае np.random.normal, low в случае np.random.uniform, alpha в случае np.random.beta
    :param p2: дисперсия ошибки в случае np.random.normal, high в случае np.random.uniform, beta в случае np.random.beta
    :param RANDOM_SEED: "random" csv number
    :param equation_type: select equation type from given:
                          (Default value: ``0``)
        - 0: $x'(t) = sin(x(t)) + \alpha * f(x(t))$, where $f()$ is an error function.
        - 1: $x'' + x' = sin(x(t)) + f(x(t))$, where $f()$ is an error function.
    """
    
    def __init__(self, size = None, f=np.random.normal, p1=None, p2=None, p3=None, RANDOM_SEED=123, alpha=1, equation_type=0, *args, **kwargs):
        self.size = size
        self.f    = f
        self.p1   = p1
        self.p2   = p2
        self.p3   = p3
        self.alpha = alpha
        self.RANDOM_SEED   = RANDOM_SEED
        self.synth_cps = []
        self.data = None
        self.equation_type = equation_type
        self.FLOAT_LIMITER = 1e3
    
    def errors(self):
        if self.p1 in ['buy', 'sell']:
            dir_ = f'moex_data2/noises/{self.p1}/{self.p2}/{self.RANDOM_SEED}.csv'
            return pd.read_csv(dir_)['STANDARDIZED_NOISE']
        if self.p1 in ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'SBERP', 'TCSG', 'TRUR', 'VTBR', 'YNDX']:
            dir_ = f'moex_data3/{self.p1}/noises/{self.RANDOM_SEED}.csv'
            return pd.read_csv(dir_)['STANDARDIZED_NOISE']
        if self.p1 == 'ECG':
            dir_ = 'ecg_data/sample_'+str(self.RANDOM_SEED)+'.csv'
            return pd.read_csv(dir_)['X']
        elif self.p1 == 'Si':
            dir_ = 'si_ar.csv'
            err = pd.read_csv(dir_)
            err = err[self.p2].dropna().values
            return err
        elif self.p1 == 'el_nino':
            dir_ = 'el_nino_data/mer_wind_' + str(self.RANDOM_SEED)+'.csv'
            return pd.read_csv(dir_)['mer.winds']
        elif self.p1 == 'moex':
            dir_ = 'moex_data/SBER/noises/noise_b_' + str(self.RANDOM_SEED)+'.csv'
            df = pd.read_csv(dir_)['STANDARDIZED_NOISE']
#             self.size = min(self.size, df.shape[0]) # вместо лен df.shape[0]
#             print(df.shape, self.size)
            return df
        else:
            if self.p2 == None:
                return self.f(self.p1, size=self.size)
            elif self.p3 == None:
                return self.f(self.p1, self.p2, self.size)
            else:
                return self.f(self.p1, self.p2, self.p3, self.size)
    
    def get_data(self):
        np.random.seed(self.RANDOM_SEED)
        data = []
        
        errors = self.errors()
        errors -= np.mean(errors)
        errors /= np.std(errors)   
        
#         if self.size is None:
#             self.size = errors.shape[0]
            
#         print(errors.shape)    

        # 0: $x'(t) = sin(x(t)) + \alpha * f(x(t))$
        if self.equation_type == 0:
            data = []
            i = 0
            x_t = 0
            data.append(x_t)

            for i in range(min(errors.shape[0], int(1e4))):
#                 print(errors.shape)
                x_t = x_t + np.sin(x_t) + self.alpha * errors[i]
                data.append(x_t)
#                 i += 1
       
            self.data = data
#             print(np.array(data).shape)
            return data
        
        # 1: $x'' + x' = sin(x(t)) + f(x(t))$
        if self.equation_type == 1:
            N = self.size
            h = self.alpha

            errors1 = self.errors()
            errors1 -= np.mean(errors1)
            errors1 /= np.std(errors1)
            
            X = np.zeros(N)
            Y = np.zeros(N)
            X[0], Y[0] = 0.0, 0.0

            # Simulate one run
            for i in range(N - 1):
                # Generating colored noise
                xi = errors[i]
                xi1 = errors1[i]

                # Predictor step
                X1 = X[i] + h * Y[i]
                Y1 = Y[i] + h * (np.sin(X[i]) - Y[i] + xi)

                # Corrector step
                X[i + 1] = X[i] + 0.5 * h * (Y[i] + Y1)
                Y[i + 1] = Y[i] + 0.5 * h * (np.sin(X[i]) + np.sin(X1) - Y[i] - Y1 + xi + xi1)

                if abs(X[i + 1]) > self.FLOAT_LIMITER:
                    X[i + 1] = (X[i + 1]) / abs(X[i + 1]) * self.FLOAT_LIMITER # take with correct sign
            


            self.data = X
            data = X
            return data
        
        return data
    
    def data_to_pi(self):
        pi_table = pd.DataFrame()
        # data1 = self.get_data(dt, D)
        pi_list = [0]
        level_list = [0]
        # FLOAT_LIMITER = 10e9 # handle exploding error

        # if self.equation_type == 0:
        #     scaling_factor = math.pi
        # elif self.equation_type == 1:
        #     scaling_factor = math.pi * 2 + math.pi

        # self.data = self.data[~np.isnan(self.data)]

        scaling_factor = math.pi

        for i in range(1, len(self.data)):

            # print(f'data index {i}')
            
            error_value = self.data[i]

            # if abs(error_value) > FLOAT_LIMITER: # handle exploding error
            #     error_value = FLOAT_LIMITER if error_value > 0 else -FLOAT_LIMITER
            
            # temp = error_value/scaling_factor

            if self.equation_type == 0:
                temp = error_value / scaling_factor
            if self.equation_type == 1:
                temp = (error_value - scaling_factor) / (2 * scaling_factor)

            # print('prev level:', level_list[i-1])
            # print('abs(temp):', abs(temp))
            # print('error_value:', error_value)
            # print('temp: ', temp)
            # print()
            if (pi_list[i-1] > 0  and 0 > temp) or (pi_list[i-1] < 0  and 0 < temp):
                if abs(temp - level_list[i-1]) >= 1:
                    if temp > 0:
                        level_list.append(level_list[i-1] + math.floor(abs(temp - level_list[i-1])))
                    else:
                        level_list.append(level_list[i-1] - math.floor(abs(temp - level_list[i-1])))
                else:
                      level_list.append(level_list[i-1])
                
            else:
                if abs(temp) - abs(level_list[i-1]) >= 1:
                    if temp > 0:
                            level_list.append(level_list[i-1] + math.floor(abs(temp) - abs(level_list[i-1])))
                    else:
                            level_list.append(level_list[i-1] - math.floor(abs(temp) - abs(level_list[i-1])))
                elif abs(temp) - abs(level_list[i-1]) <= -1:
                    if temp > 0:
                        level_list.append(level_list[i-1] - math.floor(abs(abs(temp) - abs(level_list[i-1]))))
                    else:
                        level_list.append(level_list[i-1] + math.floor(abs(abs(temp) - abs(level_list[i-1]))))
                else:
                        level_list.append(level_list[i-1])

            pi_list.append(temp)

        pi_table['data_pi'] = pi_list
        pi_table['levels'] = level_list
        return pi_table
    
    def labels(self):
        labels = [] #разметка данных: 1 если был переход, 0 если не было
        labels.append(0) #первый элемент не с чем сравнивать, по умолчанию без перехода
        data = self.data_to_pi()
        for i in range(1, data.shape[0]):
            if data.iloc[i][1] != data.iloc[i-1][1]:
                labels.append(1)
            else:
                labels.append(0)
        self.labels_ = labels
        data['MARKS'] = labels
        return data
    
    def print_data(self, title='', width=50):
        fig = plt.figure(figsize=(30,10)) #визуализация
        font = {'size': 20}
        plt.rc('font', **font)
        table = self.labels()
        table['index'] = table.index
        for i in table['index']:
            x = table.iloc[i][3]
            y = table.iloc[i][0]
            level = table.iloc[i][1]
            mark = table.iloc[i][2]
            if mark == 1:
                if level == 0:
                    plt.scatter(x, y, marker = 'v', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'v', c = 'r', s=width)
            elif mark == 0:
                if level == 0:
                    plt.scatter(x, y, marker = 'o', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'o', c = 'r', s=width)


        ones = table['MARKS']
        plt.vlines(np.where(ones)[0], -2, 10)

        plt.title(title, fontweight='bold')
        plt.xlabel('Time', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.grid()
        return fig


class Multilable_Telegraph_Process:
    """
    Инициализация телеграфного процесса
    :param size: размер выборки
    :param f: функция ошибки
    :param p1: матожидание ошибки в случае np.random.normal, low в случае np.random.uniform, alpha в случае np.random.beta
    :param p2: дисперсия ошибки в случае np.random.normal, high в случае np.random.uniform, beta в случае np.random.beta
    """
    
    def __init__(self, size, f, p1, p2=None, p3=None, RANDOM_SEED=123):
        self.size = size
        self.f    = f
        self.p1   = p1
        self.p2   = p2
        self.p3   = p3
        self.RANDOM_SEED   = RANDOM_SEED
    
    def errors(self):
        if self.p1 == 'ECG':
            dir_ = 'ecg/sample_'+str(self.RANDOM_SEED)+'.csv'
            return pd.read_csv(dir_).values
        else:
            if self.p2 == None:
                return self.f(self.p1, size=self.size)
            elif self.p3 == None:
                return self.f(self.p1, self.p2, self.size)
            else:
                return self.f(self.p1, self.p2, self.p3, self.size)
    
    def get_data(self):
        np.random.seed(123)
        data = []
        i = 0
        x_t = 0
        data.append(x_t)
        errors = self.errors()
        errors -= np.mean(errors)
        errors /= np.std(errors)
        while i < self.size:
            x_t = x_t + math.sin(x_t) + errors[i]
            data.append(x_t)
            i += 1
        self.data = data
        return data
    
    def clusters(self):
        marked_table = pd.DataFrame()
        data = self.get_data()
        for i in range(0, len(data)): #приводим данные к новой шкале {-pi, 0, pi}
            if data[i] >= 0:
                if  data[i] >= 3*math.pi:
                    new_str = {'INIT': data[i], 'LABEL': 3*math.pi}
                    marked_table = marked_table.append(new_str, ignore_index=True)
                elif 3*math.pi >= data[i] >= 2*math.pi:
                    if np.linalg.norm(data[i]-3*math.pi) <= np.linalg.norm(data[i]-2*math.pi):
                        new_str = {'INIT': data[i], 'LABEL': 3*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': 2*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                elif 2*math.pi >= data[i] >= math.pi:
                    if np.linalg.norm(data[i]-2*math.pi) <= np.linalg.norm(data[i]-math.pi):
                        new_str = {'INIT': data[i], 'LABEL': 2*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                elif math.pi >= data[i] >= 0:
                    if np.linalg.norm(data[i]-math.pi) <= np.linalg.norm(data[i]-0):
                        new_str = {'INIT': data[i], 'LABEL': math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': 0}
                        marked_table = marked_table.append(new_str, ignore_index=True)
            elif data[i] < 0:
                if  data[i] <= -3*math.pi:
                    new_str = {'INIT': data[i], 'LABEL': -3*math.pi}
                    marked_table = marked_table.append(new_str, ignore_index=True)
                elif -3*math.pi <= data[i] <= -2*math.pi:
                    if np.linalg.norm(data[i]-(-3*math.pi)) <= np.linalg.norm(data[i]-(-2*math.pi)):
                        new_str = {'INIT': data[i], 'LABEL': -3*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': -2*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                elif -2*math.pi <= data[i] <= -math.pi:
                    if np.linalg.norm(data[i]-(-2*math.pi)) <= np.linalg.norm(data[i]-(-math.pi)):
                        new_str = {'INIT': data[i], 'LABEL': -2*math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': -math.pi}
                        marked_table = marked_table.append(new_str, ignore_index=True)
                elif -math.pi <= data[i] <= 0:
                    if np.linalg.norm(data[i]-(-math.pi)) <= np.linalg.norm(data[i]-0):
                        new_str = {'INIT': data[i], 'LABEL': -math.pi}
                        marked_table= marked_table.append(new_str, ignore_index=True)
                    else:
                        new_str = {'INIT': data[i], 'LABEL': 0}
                        marked_table = marked_table.append(new_str, ignore_index=True)
        return marked_table

    # def clusters(self):
    #     data = self.get_data()
    #     marked_table = pd.DataFrame({'INIT': np.zeros(len(data)), 'LABEL': np.zeros(len(data))})
    #     for i in range(0, len(data)): #приводим данные к новой шкале {-pi, 0, pi}
    #         if data[i] >= 0:
    #             if  data[i] >= 3*math.pi:
    #                 marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 3*math.pi}
    #             elif 3*math.pi >= data[i] >= 2*math.pi:
    #                 if np.linalg.norm(data[i]-3*math.pi) <= np.linalg.norm(data[i]-2*math.pi):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 3*math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 2*math.pi}
    #             elif 2*math.pi >= data[i] >= math.pi:
    #                 if np.linalg.norm(data[i]-2*math.pi) <= np.linalg.norm(data[i]-math.pi):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 2*math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': math.pi}
    #             elif math.pi >= data[i] >= 0:
    #                 if np.linalg.norm(data[i]-math.pi) <= np.linalg.norm(data[i]-0):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 0*math.pi}
    #         elif data[i] < 0:
    #             if  data[i] <= -3*math.pi:
    #                 marked_table[i] = {'INIT': data[i], 'LABEL': -3*math.pi}
    #             elif -3*math.pi <= data[i] <= -2*math.pi:
    #                 if np.linalg.norm(data[i]-(-3*math.pi)) <= np.linalg.norm(data[i]-(-2*math.pi)):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': -3*math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': -2*math.pi}
    #             elif -2*math.pi <= data[i] <= -math.pi:
    #                 if np.linalg.norm(data[i]-(-2*math.pi)) <= np.linalg.norm(data[i]-(-math.pi)):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': -2*math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': -math.pi}
    #             elif -math.pi <= data[i] <= 0:
    #                 if np.linalg.norm(data[i]-(-math.pi)) <= np.linalg.norm(data[i]-0):
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': -math.pi}
    #                 else:
    #                     marked_table.iloc[i, :] = {'INIT': data[i], 'LABEL': 0}
    #     return marked_table
    
    def labels(self):
        labels = [] #разметка данных: 1 если был переход, 0 если не было
        labels.append(0) #первый элемент не с чем сравнивать, по умолчанию без перехода
        data = self.clusters()
        for i in range(1, data.shape[0]):
            if data.iloc[i][1] > data.iloc[i-1][1]:
                labels.append(1)
            elif data.iloc[i][1] < data.iloc[i-1][1]:
                labels.append(-1)
            else:
                labels.append(0)
        data['MARKS'] = labels
        self.labels = labels
        return data
    
    def print_data(self, title='', width=50):
        fig = plt.figure(figsize=(30,10)) #визуализация
        font = {'weight': 'bold', 'size': 22}
        plt.rc('font', **font)
        table = self.labels()
        table['index'] = table.index
        for i in table['index']:
            x = table.iloc[i][3]
            y = table.iloc[i][0]
            level = table.iloc[i][1]
            mark = table.iloc[i][2]
            if mark == 0:
                if level == 0:
                    plt.scatter(x, y, marker = 'o', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'o', c = 'r', s=width)
            else:
                if level == 0:
                    plt.scatter(x, y, marker = 'v', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'v', c = 'r', s=width)
        plt.title(title, fontweight='bold')
        plt.xlabel('Time', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.grid()
        return fig