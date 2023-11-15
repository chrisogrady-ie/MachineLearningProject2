import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def task1():
    # labels are 0 - 9
    # we need 5, 7, 9
    df = pd.read_csv('fashion-mnist_train.csv')
    dfa = df[(df['label'] == 5) | (df['label'] == 7) | (df['label'] == 9)]
    df_label = dfa['label']
    df_pixels_temp = dfa.loc[::28, dfa.columns != 'label']

    #result = pd.DataFrame(df.values.reshape(-1, 3),
    #                      index=df.index.repeat(2), columns=list('XYZ'))

    #df_pixels = pd.DataFrame(df_pixels_temp.values.reshape(-1, 28))

    # change shape of pixel dataframe to 28x28
    # change to series, reshape, turn back to dataframe
    # df_pixels.shape(28, 28)
    print(df_pixels_temp.shape)
    print(len(df_pixels_temp))

    # print image of 1, 7, 21
    # plt.figure()
    # plt.imshow(dfa[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    return df_label, df_pixels_temp


# def task2():


# def task3():


# def task4():


# def task5():


# def task6():


# def task7():


def main():
    labels, pixels = task1()
    # print(pixels)

    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()


main()
