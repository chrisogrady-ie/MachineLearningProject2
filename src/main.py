import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.linear_model import Perceptron


def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_time_series_plot(dict_in):
    key = sorted(dict_in.keys())
    values = [dict_in[time_value] for time_value in key]

    plt.figure(figsize=(10, 5))
    plt.plot(key, values, marker='o', linestyle='-')
    plt.xlabel('Input size')
    plt.ylabel('Time taken in µs')
    plt.title('Time Series Plot of Perceptron')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def task1():
    # labels are 0 - 9
    # Sandal, Sneaker, Ankle Boot
    #   5,       7,        9
    df = pd.read_csv('fashion-mnist_train.csv')
    df_set = df[(df['label'] == 5) | (df['label'] == 7) | (df['label'] == 9)]
    df_label = df_set['label']
    df_pixels = df_set.loc[:, df_set.columns != 'label']

    print("Label shape: ", df_label.shape, "Length ", len(df_label))
    print("Pixels shape: ", df_pixels.shape, "Length ", len(df_pixels))

    # Sandal
    show_img(df_pixels.iloc[0].values.reshape(28, 28))
    print(df_label.iloc[0])
    # Sneaker
    show_img(df_pixels.iloc[1].values.reshape(28, 28))
    print(df_label.iloc[1])
    # Ankle Boot
    show_img(df_pixels.iloc[4].values.reshape(28, 28))
    print(df_label.iloc[4])

    return df_set


def task2(data, sample_size):
    this_data = data.sample(n=sample_size, random_state=21)
    kf = KFold(n_splits=10, shuffle=False)

    np_label = this_data['label'].values
    np_pixels = this_data.loc[:, this_data.columns != 'label'].values

    # max, min and average of:
    total_training_time = []
    total_evaluation_time = []
    total_accuracy = []

    counter = 0
    # evaluation is an incrementing 10% chunk
    for training, evaluation in kf.split(np_pixels):
        counter += 1
        split_start = datetime.now()

        pixels_train, pixels_eval = np_pixels[training], np_pixels[evaluation]
        labels_train, labels_eval = np_label[training], np_label[evaluation]

        # train using pixels_train and labels_train
        model = Perceptron(max_iter=1000)
        model.fit(pixels_train, labels_train)
        split_trained = (datetime.now() - split_start).microseconds

        # evaluate using pixels_eval and labels_eval
        predictions = model.predict(pixels_eval)
        split_evaluated = (datetime.now() - split_start).microseconds

        accuracy = accuracy_score(labels_eval, predictions)

        # x and y co-ordinate for each class
        confusion = metrics.confusion_matrix(labels_eval, predictions)

        print('Fold {} trained in {} evaluated in {} with accuracy of {}'
              .format(counter, split_trained, split_evaluated, accuracy))
        total_training_time.append(split_trained)
        total_evaluation_time.append(split_evaluated)
        total_accuracy.append(accuracy)
        print(f'Confusion matrix:\n {confusion}\n')

    # Training time stats:
    total, maximum, minimum = 0, 0, 1000000
    for t in total_training_time:
        if t < minimum:
            minimum = t
        if t > maximum:
            maximum = t
        total += t
    average_training_time = total/counter
    print(f'Training time:\nAverage: {average_training_time}µs Max: {maximum}µs Min: {minimum}µs')

    # Eval time stats:
    total, maximum, minimum = 0, 0, 1000000
    for e in total_evaluation_time:
        if e < minimum:
            minimum = e
        if e > maximum:
            maximum = e
        total += e
    average_evaluation_time = total/counter
    print(f'Evaluation time:\nAverage {average_evaluation_time}µs Max: {maximum}µs Min: {minimum}µs')

    # Accuracy stats:
    total, maximum, minimum = 0, 0, 100
    for a in total_accuracy:
        if a < minimum:
            minimum = a
        if a > maximum:
            maximum = a
        total += a
    print(f'Accuracy:\nAverage:{total/counter} Max: {maximum} Min: {minimum}')

    return average_training_time

# def task3():


# def task4():


# def task5():


# def task6():


# def task7():


def main():
    data = task1()
    task3_times = {
        500: task2(data, 500),
        2500: task2(data, 2500),
        #5000: task2(data, 5000),
        #7500: task2(data, 7500),
        #10000: task2(data, 10000),
        #12500: task2(data, 12500),
        #15000: task2(data, 15000),
        #18000: task2(data, 18000)
    }
    show_time_series_plot(task3_times)

    # task3()
    # task4()
    # task5()
    # task6()
    # task7()


main()
