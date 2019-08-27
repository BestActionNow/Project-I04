# coding:utf-8
import pandas as pd
import pickle
import logging
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn import  metrics
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


def one_hot_representation(sample, fields_dict, isample):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param isample: sample index
    :return: sample index
    """
    index = []
    for field in fields_dict:
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        index.append([isample,ind])
    return index

def read_data(path):
    data = pd.read_csv(path,header=None,encoding='utf-8',delim_whitespace=True,
                         names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time"])
    return data

def train_sparse_data_generate(train_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in train_data:
        labels = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            click = sample['click']
            # get labels
            if click == 0:
                label = 0
            else:
                label = 1
            labels.append(label)
            # get indexes
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'labels':labels})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('../avazu_CTR/train_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)

def test_sparse_data_generate(test_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in test_data:
        ids = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            ids.append(sample['id'])
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'id':ids})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('../avazu_CTR/test_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)

def read_preprocess():
    data = pd.read_csv('~/ByteCamp/bytecamp.data', sep=',', header=0)
    sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id']
    dense_features = ['duration']
    target = ['finish', 'like']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    x_df_sparse = data[sparse_features]
    x_df_dense = data[dense_features]
    y_df = data[target]
    date = np.array(data['date'].tolist())

    x_sparse = []
    for i in range(len(x_df_sparse)) :
        value_list = x_df_sparse.iloc[i].tolist()
        x_sparse.append(value_list)
        if i % 100000 == 0:
            print("the {}th item laading!".format(i))

    # dense feature normalize processing
    mms = MinMaxScaler(feature_range=(0, 1))
    print("start normalize processing")
    x_dense = np.array(mms.fit_transform(x_df_dense))
    print("end normalize processing")

    # sparse feature onehot processing
    ohe = OneHotEncoder(handle_unknown='ignore', sparse = True)
    print("start one-hot processing")
    x_sparse = ohe.fit_transform(x_sparse)    
    print("end on-hot processing")
    x = hstack((x_sparse, x_dense)).tocsr()
    np.save("feature_array", x)
    # sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
    #                        for feat in sparse_features]
    # dense_feature_list = [SingleFeat(feat, 0)
    #                       for feat in dense_features]
    train_model_input = x[date <= 20190707]
    test_model_input = x[date > 20190707]
    train_labels = np.array(y_df.values.tolist())[date <= 20190707]
    test_labels = np.array(y_df.values.tolist())[date > 20190707]
    
    return np.array(train_model_input).T, np.array(train_labels).T, np.array(test_model_input).T, np.array(test_labels).T

# generate batch indexes
if __name__ == '__main__':

    train_features, train_labels, test_features, test_labels = read_preprocess()
    print("Data Loaded! train_features: " , train_features.shape,
          " train_labels: " , train_labels.shape,
          " test_features: " , test_features.shape,
          " test_labels: " , test_labels.shape)







