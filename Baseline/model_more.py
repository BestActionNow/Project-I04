import tensorflow as tf
from deepctr.inputs import input_from_feature_columns, get_linear_logit,build_input_features,combined_dnn_input, get_dense_input
# from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun
import numpy as np


def xDeepFM_MTL(linear_feature_columns, dnn_feature_columns, gate_feature_columns, embedding_size=8, dnn_hidden_units=(256, 256), cin_layer_size=(256, 256,),
                cin_split_half=True, init_std=0.0001,l2_reg_dnn=0, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False,
                task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                seed=1024, ):
    # check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    features = build_input_features(linear_feature_columns + dnn_feature_columns + gate_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         embedding_size,
                                                                         l2_reg_embedding, init_std,
                                                                         seed)
    gate = get_dense_input(features, gate_feature_columns)[0]

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, 0, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   dnn_use_bn, seed)(dnn_input)

    finish_out1 = DNN(task_net_size)(deep_out)
    finish_logit1 = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out1)

    like_out1 = DNN(task_net_size)(deep_out)
    like_logit1 = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out1)

    finish_out2 = DNN(task_net_size)(deep_out)
    finish_logit2 = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out2)

    like_out2 = DNN(task_net_size)(deep_out)
    like_logit2 = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out2)

    # condition = tf.placeholder("float32", shape=[None, 1], name="condition")

    finish_logit = gate * finish_logit1 + (1.0-gate) * finish_logit2
    like_logit = gate * like_logit1 + (1.0-gate) * like_logit2

    print(np.shape(like_logit))

    finish_logit = tf.keras.layers.add(
        [linear_logit, finish_logit, exFM_logit])
    like_logit = tf.keras.layers.add(
        [linear_logit, like_logit, exFM_logit])

    output_finish = PredictionLayer('binary', name='finish')(finish_logit)
    output_like = PredictionLayer('binary', name='like')(like_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[
                                  output_finish, output_like])
    return model
