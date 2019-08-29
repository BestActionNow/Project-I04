import pandas as pd
import time, datetime
#from deepctr import SingleFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
from model_more import xDeepFM_MTL
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'

loss_weights = [1.0, 1.0, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def change_time(timeStamp):
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    otherStyleTime = dateArray.strftime("%Y%m%d")

    return int(otherStyleTime)

if __name__ == "__main__":
    data = pd.read_csv('./data/bytecamp.data', sep=',', header=0)
   # Index(['duration', 'generate_time', 'finish', 'like', 'date', 'uid',
   #        'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id'],
   #       dtype='object')
    # data['time'] = data['generate_time'].apply(change_time)

    data['gate'] = data['u_region_id']

    sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id']
    dense_features = ['duration']
    gate_features = ['gate']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
    #                        for feat in sparse_features]
    # dense_feature_list = [SingleFeat(feat, 0)
    #                       for feat in dense_features]

    train = data[data['date'] <= 20190707]
    test = data[data['date'] == 20190708]

    # train_labels = [train[target[0]].values, train[target[1]].values]
    # test_labels = [test[target[0]].values, test[target[1]].values]

    train_y_id = train['g_region_id'].values
    test_y_id = test['g_region_id'].values

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, 1)
                             for feat in dense_features]

    gate_feature_columns = [DenseFeat(feat, 1)
                             for feat in gate_features]

    # sparse_feature_columns = [SparseFeat(feat, dimension=int(1e6), use_hash=True) for feat in
    #                           sparse_features]  # The dimension can be set according to data
    # dense_feature_columns = [DenseFeat(feat, 1)
    #                          for feat in dense_features]

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    linear_feature_columns = sparse_feature_columns + dense_feature_columns

    feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns + gate_feature_columns)

    print(feature_names)
    train_model_input = [train[name] for name in feature_names]

    test_model_input = [test[name] for name in feature_names]

    print('PPPP')
    print()
    model = xDeepFM_MTL(linear_feature_columns, dnn_feature_columns, gate_feature_columns)
    model.compile("adagrad", loss={
                  'finish': "binary_crossentropy",
                  'like': "binary_crossentropy"},
                  loss_weights=loss_weights)

    print(train_labels)

    history = model.fit(train_model_input, train_labels,
                        batch_size=4096, epochs=2, verbose=1)
    pred_ans = model.predict(test_model_input, batch_size=2 ** 10)

    test['finish_probability'] = pred_ans[0]
    test['like_probability'] = pred_ans[1]

    test_finish_auc = metrics.roc_auc_score(test['finish'], test['finish_probability'])
    test_like_auc = metrics.roc_auc_score(test['like'], test['like_probability'])
    print('the auc of test finish')
    print(test_finish_auc)
    print('the auc of tes like')
    print(test_like_auc)

    #----------


    pred_ans = model.predict(train_model_input, batch_size=2 ** 10)

    train['finish_probability'] = pred_ans[0]
    train['like_probability'] = pred_ans[1]

    train_finish_auc = metrics.roc_auc_score(train['finish'], train['finish_probability'])
    train_like_auc = metrics.roc_auc_score(train['like'], train['like_probability'])
    print('the auc of train finish')
    print(train_finish_auc)
    print('the auc of train like')
    print(train_like_auc)