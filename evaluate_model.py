from naive_utils import *

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

# print(model[2])
test_model(test, model, alpha, N)