import pandas as pd
import numpy as np


def task1():
    # labels are 0 - 9
    # we need 5, 7, 9
    df = pd.read_csv('fashion-mnist_train.csv')
    dfa = df[(df['label'] == 5) | (df['label'] == 7) | (df['label'] == 9)]
    df_label = dfa['label']
    df_pixels = dfa.loc[:, dfa.columns != 'label']

    # print image of 1, 7, 21
    
    return df_label, df_pixels


#def task2():



#def task3():



#def task4():



#def task5():



#def task6():



#def task7():





def main():
    labels, pixels = task1()
    print(pixels)


    #task2()
    #task3()
    #task4()
    #task5()
    #task6()
    #task7()


main()
