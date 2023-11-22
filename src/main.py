import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_time_series_plot(dict_in, model):
    acc_pop = 0
    acc = 0
    dictionary = {}
    for k in dict_in:
        dictionary[k] = [dict_in[k][0], dict_in[k][1]]
        acc += (dict_in[k][2])
        acc_pop += 1

    acc = acc/acc_pop

    key = sorted(dictionary.keys())
    values = [dictionary[time_value] for time_value in key]
    plt.figure(figsize=(10, 5))

    plt.plot(key, values, marker='o', linestyle='-')

    plt.xlabel('Input size  ///// Average accuracy = ' + str(acc))
    plt.ylabel('Time taken in µs')
    plt.title('Time Series Plot of ' + model)
    plt.xticks(rotation=30)
    plt.tight_layout()
    #plt.show()


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


def task2(data, sample_size, model_type):
    this_data = data.sample(n=sample_size, random_state=21)
    kf = KFold(n_splits=10, shuffle=False)

    np_label = this_data['label'].values
    np_pixels = this_data.loc[:, this_data.columns != 'label'].values

    # max, min and average of:
    total_training_time = []
    total_evaluation_time = []
    total_accuracy = []

    model = Perceptron(max_iter=1000)
    if model_type == 'decision':
        model = tree.DecisionTreeClassifier(max_depth=10)

    if model_type == 'neighbours':
        model = KNeighborsClassifier(n_neighbors=5)

    if model_type == 'svm':
        model = svm.SVC(kernel='rbf', gamma=0.27)

    counter = 0
    # evaluation is an incrementing 10% chunk
    for training, evaluation in kf.split(np_pixels):
        counter += 1
        split_start = datetime.now()

        pixels_train, pixels_eval = np_pixels[training], np_pixels[evaluation]
        labels_train, labels_eval = np_label[training], np_label[evaluation]

        # train using pixels_train and labels_train
        model.fit(pixels_train, labels_train)
        split_trained = (datetime.now() - split_start).microseconds
        split_start = datetime.now()

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
    average_accuracy = total/counter
    print(f'Accuracy:\nAverage:{average_accuracy} Max: {maximum} Min: {minimum}')

    return average_training_time, average_evaluation_time, average_accuracy


def main():
    data = task1()
    task3_times = {
        500: task2(data, 500, 'perceptron'),
        2500: task2(data, 2500, 'perceptron'),
        5000: task2(data, 5000, 'perceptron'),
        7500: task2(data, 7500, 'perceptron'),
        10000: task2(data, 10000, 'perceptron'),
        12500: task2(data, 12500, 'perceptron'),
        15000: task2(data, 15000, 'perceptron'),
        18000: task2(data, 18000, 'perceptron')
    }
    show_time_series_plot(task3_times, 'perceptron')

    task4_times = {
        500: task2(data, 500, 'decision'),
        2500: task2(data, 2500, 'decision'),
        5000: task2(data, 5000, 'decision'),
        7500: task2(data, 7500, 'decision'),
        10000: task2(data, 10000, 'decision'),
        12500: task2(data, 12500, 'decision'),
        15000: task2(data, 15000, 'decision'),
        18000: task2(data, 18000, 'decision')
    }
    show_time_series_plot(task4_times, 'decision')

    task5_times = {
        500: task2(data, 500, 'neighbours'),
        2500: task2(data, 2500, 'neighbours'),
        5000: task2(data, 5000, 'neighbours'),
        7500: task2(data, 7500, 'neighbours'),
        10000: task2(data, 10000, 'neighbours'),
        12500: task2(data, 12500, 'neighbours'),
        15000: task2(data, 15000, 'neighbours'),
        18000: task2(data, 18000, 'neighbours')
    }
    show_time_series_plot(task5_times, 'neighbours')

    task6_times = {
        500: task2(data, 500, 'svm'),
        2500: task2(data, 2500, 'svm'),
        #5000: task2(data, 5000, 'svm'),
        #7500: task2(data, 7500, 'svm'),
        #10000: task2(data, 10000, 'svm'),
        #12500: task2(data, 12500, 'svm'),
        #15000: task2(data, 15000, 'svm'),
        #18000: task2(data, 18000, 'svm')
    }
    show_time_series_plot(task6_times, 'svm')

    plt.show()


main()
print('\n\n')
print('Perceptron\'s training times scale linearly with input size '
      'and is quick with high accuracy in our dataset size. Evaluation time is almost non-existent.')
print('With decision trees training times input size does not directly effect training and evaluation time. '
      'Accuracy scores are high but slightly lower than Perceptron. Again, evaluation times are minimal')
print('K-Nearest neighbours does not have a training phase as data is just saved in memory. With evaluation'
      'times in our model, they remain in an acceptable range. However if this dataset were larger the time taken '
      'would reach an unacceptable level and our perceptron classifier may be more efficient as accuracy levels '
      'are similar, being high.')
print('Support vector machine takes by far the longest and would require changing our units of measurement'
      'from microseconds to seconds in order to account for the time taken. Depending on our y value '
      'the accuracy ranges can be quite low. A benefit of this classification is that time needed does '
      'not increase as much as the other methods for training and predictions. Meaning, with'
      ' a much larger dataset it would be more efficient but all y values tested here seem to have '
      'an accuracy of ~33 percent.')
print('\n\n')
print('Taking completion time and accuracy into account:\n'
      'K-nearest neighbours\n'
      'Perceptron\n'
      'Decision tree\n'
      'Support vector machine')

