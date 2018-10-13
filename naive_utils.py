import csv
from numpy import random
import re


# loads file containing the SMS dataset and return a list
def load_data(f):
    input_data = csv.reader(open(f, 'r'), delimiter='\t')
    data = list(input_data)

    # for i in range(len(data)):
        # print("complete line:", data[i])
        # print("label", data[i][0])
        # print("SMS", data[i][1])


    return data

def clean_sms(sms):
    # everything to lowercase
    # remove everything except alphanumeric
    sms = sms.lower().replace(' ', '_')
    # .replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace(':', '').replace(' ', '')
    sms = re.sub(r'\W+', '', sms).replace('_', ' ')

    return sms

def preprocess_data(data):
    # 2x1 vector
    clean_data = []
    for i in range(len(data)):
        if data[i][0] == "ham":
            clean_data.append([0, clean_sms(data[i][1])])
        else:
            clean_data.append([1, clean_sms(data[i][1])])

    return clean_data

def split(data, train_fraction):
    train = []
    test = []

    randomized_indexes = random.permutation(range(0, len(data)))

    for i in range(len(data)):
        if i < round(len(data) * train_fraction):
            train.append(data[i])
        else:
            test.append(data[i])

    print("train set count:", len(train), "test set count:", len(test), "total:", len(data))

    return train, test

def train_model(data):
    ham = {}
    spam = {}
    # count_ham = 0
    # count_spam = 0

    for i in range(len(data)):
        # ham
        if data[i][0] == 0:
            for word in data[i][1].split():
                # print("word in data[i][1]", word)
                if word in ham:
                    ham[word] = ham[word] + 1
                else:
                    ham[word] = 1
        # spam
        else:
            for word in data[i][1].split():
                # print("word in data[i][1]", word)
                if word in spam:
                    spam[word] = spam[word] + 1
                else:
                    spam[word] = 1

    # count_ham = len(ham)
    # count_spam = len(spam)

    model = [ham, spam]

    return model


def classify_sms(sms, model, alpha, N):
    ham_dict = model[0]
    spam_dict = model[1]

    spam_class_probability = len(spam_dict) / (len(spam_dict) + len(ham_dict))
    ham_class_probability = 1 - spam_class_probability

    # print("spam prob", spam_class_probability, "ham prob", ham_class_probability)

    is_ham_probability = ham_class_probability
    is_spam_probability = spam_class_probability
    for word in sms.split():
        if word in ham_dict:
            word_count_in_ham = ham_dict[word]
        else:
            word_count_in_ham = 0

        if word in spam_dict:
            word_count_in_spam = spam_dict[word]
        else:
            word_count_in_spam = 0

        # probability of ham
        is_ham_probability = is_ham_probability * ((word_count_in_ham + alpha) / (len(ham_dict) + (N * alpha)))

        # probability of spam
        is_spam_probability = is_spam_probability * ((word_count_in_spam + alpha) / (len(spam_dict) + (N * alpha)))

        # print("word:",word, "isham", is_ham_probability, "is spam", is_spam_probability, word_count_in_ham, word_count_in_spam)

    if is_ham_probability >= is_spam_probability:
        c = 0
    else:
        c = 1

    # print("class", c)

    return c

# calculate accuracy, confusion matrix, precision, recall, and f-score
def test_model(test, model, alpha, N):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(test)):
        true_class = test[i][0]
        predicted_class = classify_sms(test[i][1], model, alpha, N)
        if true_class == 1:
            if predicted_class == 1:
                true_positive = true_positive + 1
            else:
                false_negative = false_negative + 1
        else:
            if predicted_class == 1:
                false_positive = false_positive + 1
            else:
                true_negative = true_negative + 1

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * precision * recall / (precision + recall)

    print("                   Confusion Matrix")
    print("                                      True condition")
    print("                                 Positive    |     Negative")
    print("Predicted Condition Positive","      TP", true_positive, "          FP", false_positive)
    print("Predicted Condition Negative","      FN", false_negative, "         TN", true_negative)
    print("")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-score:", f_score)

    return accuracy, precision, recall, f_score