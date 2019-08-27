import pandas as pd
import time, datetime
from deepctr import SingleFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import cross_validation, metrics
from model import xDeepFM_MTL


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



    sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id']
    dense_features = ['duration']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    train = data[data['date'] <= 20190707]
    test = data[data['date'] == 20190708]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
                       [test[feat.name].values for feat in dense_feature_list]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights, )

    history = model.fit(train_model_input, train_labels,
                        batch_size=4096, epochs=1, verbose=1)
    pred_ans = model.predict(test_model_input, batch_size=2 ** 14)

    # test_auc = metrics.roc_auc_score(test[], prodict_prob_y)
    # print(test_auc)
    #
    # result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
    # result.rename(columns={'finish': 'finish_probability',
    #                        'like': 'like_probability'}, inplace=True)
    test['finish_probability'] = pred_ans[0]
    test['like_probability'] = pred_ans[1]

    test_finish_auc = metrics.roc_auc_score(test['finish'], test['finish_probability'])
    test_like_auc = metrics.roc_auc_score(test['like'], test['like_probability'])
    print('the auc of finish')
    print(test_finish_auc)
    print('the auc of like')
    print(test_like_auc)