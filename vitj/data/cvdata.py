import pickle
import joblib
import numpy

# airplane
# automobile
# bird
# cat
# deer
# dog
# frog
# horse
# ship
# truck

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_l = {
    'data':[],
    'labels':[],
    'filenames':[]
}
for i in range(1,6):
    file="data_batch_{}".format(i)
    numpy_array = unpickle(file)
    for key in ('data','labels','filenames'):
        train_l[key].extend(numpy_array[key.encode('UTF-8')])

joblib.dump(train_l,"train.data", compress=3)

test_l = {
    'data':[],
    'labels':[],
    'filenames':[]
}
test = unpickle("test_batch")
for key in ('data', 'labels', 'filenames'):
    test_l[key].extend(test[key.encode('UTF-8')])

joblib.dump(test_l,"test.data",compress=3)
print('DONE')