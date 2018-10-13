from naive_utils import *
import matplotlib.pyplot as plt

input_filename='SMSSpamCollection'
alpha = 0.1
N = 20000
train_size = 0.8

all_data = load_data(input_filename)
data = preprocess_data(all_data)
# print(data)

train, test = split(data, train_size)
#
# print(train)
# print(test)

model = train_model(train)

x_axis = []
y_axis_accuracy_train = []
y_axis_accuracy_test = []
y_axis_fscore_train = []
y_axis_fscore_test = []

for i in range(-5, 1, 1):
    alpha = 2**i
    x_axis.append(i)

    # evaluate performance on training data
    accuracy, precision, recall, f_score = test_model(train, model, alpha, N)
    y_axis_fscore_train.append(f_score)
    y_axis_accuracy_train.append(accuracy)

    # evaluate performance on testing data
    accuracy, precision, recall, f_score = test_model(test, model, alpha, N)
    y_axis_accuracy_test.append(accuracy)
    y_axis_fscore_test.append(f_score)

# Plot everything
plt.title('Problem 4 - b')
plt.figure(1)
plt.subplot(221)
plt.plot(x_axis, y_axis_accuracy_train)
plt.yscale('linear')
plt.title('Training data Accuracy vs i')
plt.grid(True)

plt.subplot(222)
plt.plot(x_axis, y_axis_fscore_train)
plt.yscale('linear')
plt.title('Training data F-score vs i')
plt.grid(True)

plt.subplot(223)
plt.plot(x_axis, y_axis_accuracy_test)
plt.title('Testing data Accuracy vs i')
plt.grid(True)

plt.subplot(224)
plt.plot(x_axis, y_axis_fscore_test)
plt.yscale('linear')
plt.title('Testing data F-score vs i')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()