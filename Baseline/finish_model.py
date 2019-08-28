import tensorflow as tf
from deepctr.inputs import input_from_feature_columns, get_linear_logit,build_input_features,combined_dnn_input
# from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun


def xDeepFM_MTL(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(256, 256), cin_layer_size=(256, 256,),
                cin_split_half=True, init_std=0.0001,l2_reg_dnn=0, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False,
                task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                seed=1024, ):
    # check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         embedding_size,
                                                                         l2_reg_embedding, init_std,
                                                                         seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, 0, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)


    # if len(cin_layer_size) > 0:
    #     exFM_out = CIN(cin_layer_size, 'relu',
    #                    cin_split_half, 0, seed)(fm_input)
    #     exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)

    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   dnn_use_bn, seed)(dnn_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)

    if len(dnn_hidden_units) == 0 and len(cin_layer_size) == 0:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and len(cin_layer_size) > 0:  # linear + CIN
        final_logit = tf.keras.layers.add([linear_logit, exFM_logit])
    elif len(dnn_hidden_units) > 0 and len(cin_layer_size) == 0:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(dnn_hidden_units) > 0 and len(cin_layer_size) > 0:  # linear + CIN + Deep
        final_logit = tf.keras.layers.add(
            [linear_logit, deep_logit, exFM_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer('binary')(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model

    # if len(cin_layer_size) > 0:
    #     exFM_out = CIN(cin_layer_size, 'relu',
    #                    cin_split_half, 0, seed)(fm_input)
    #     exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
    #
    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    #
    # deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
    #                dnn_use_bn, seed)(dnn_input)
    #
    # deep_logit = tf.keras.layers.Dense(
    #     1, use_bias=False, activation=None)(deep_out)
    #
    # if len(dnn_hidden_units) == 0 and len(cin_layer_size) == 0:  # only linear
    #     final_logit = linear_logit
    # elif len(dnn_hidden_units) == 0 and len(cin_layer_size) > 0:  # linear + CIN
    #     final_logit = tf.keras.layers.add([linear_logit, exFM_logit])
    # elif len(dnn_hidden_units) > 0 and len(cin_layer_size) == 0:  # linear +　Deep
    #     final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    # elif len(dnn_hidden_units) > 0 and len(cin_layer_size) > 0:  # linear + CIN + Deep
    #     final_logit = tf.keras.layers.add(
    #         [linear_logit, deep_logit, exFM_logit])
    # else:
    #     raise NotImplementedError
    #
    # output = PredictionLayer('binary')(final_logit)
    # model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    # return model


#
# def xDeepFM_MTL(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(256, 256),
#             cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
#             l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
#             dnn_activation='relu', dnn_use_bn=False, task='binary'):
#     """Instantiates the xDeepFM architecture.
#     :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
#     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
#     :param embedding_size: positive integer,sparse feature embedding_size
#     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
#     :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
#     :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
#     :param cin_activation: activation function used on feature maps
#     :param l2_reg_linear: float. L2 regularizer strength applied to linear part
#     :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
#     :param l2_reg_dnn: L2 regularizer strength applied to deep net
#     :param l2_reg_cin: L2 regularizer strength applied to CIN.
#     :param init_std: float,to use as the initialize std of embedding vector
#     :param seed: integer ,to use as random seed.
#     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
#     :param dnn_activation: Activation function to use in DNN
#     :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
#     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
#     :return: A Keras model instance.
#     """
#
#
#     features = build_input_features(linear_feature_columns + dnn_feature_columns)
#
#     inputs_list = list(features.values())
#
#     sparse_embedding_list, dense_value_list = input_from_feature_columns(features,dnn_feature_columns,
#                                                                               embedding_size,
#                                                                               l2_reg_embedding,init_std,
#                                                                               seed)
#
#     linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
#                                     seed=seed, prefix='linear')
#
#     fm_input = concat_fun(sparse_embedding_list, axis=1)
#
#     if len(cin_layer_size) > 0:
#         exFM_out = CIN(cin_layer_size, cin_activation,
#                        cin_split_half, l2_reg_cin, seed)(fm_input)
#         exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
#
#     dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
#
#     deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
#                    dnn_use_bn, seed)(dnn_input)
#     deep_logit = tf.keras.layers.Dense(
#         1, use_bias=False, activation=None)(deep_out)
#
#     if len(dnn_hidden_units) == 0 and len(cin_layer_size) == 0:  # only linear
#         final_logit = linear_logit
#     elif len(dnn_hidden_units) == 0 and len(cin_layer_size) > 0:  # linear + CIN
#         final_logit = tf.keras.layers.add([linear_logit, exFM_logit])
#     elif len(dnn_hidden_units) > 0 and len(cin_layer_size) == 0:  # linear +　Deep
#         final_logit = tf.keras.layers.add([linear_logit, deep_logit])
#     elif len(dnn_hidden_units) > 0 and len(cin_layer_size) > 0:  # linear + CIN + Deep
#         final_logit = tf.keras.layers.add(
#             [linear_logit, deep_logit, exFM_logit])
#     else:
#         raise NotImplementedError
#
#     output = PredictionLayer(task)(final_logit)
#
#     model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
#     return model