import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K

tf.compat.v1.disable_v2_behavior()

# loading keras model
K.set_learning_phase(0)
model = load_model('ann_model.h5')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in
                                tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)
        return frozen_graph

# create a frozen-graph of the keras model
frozen_graph = freeze_session(K.get_session(),
                                output_names=[out.op.name for out in model.outputs])

# save model as .pb file
tf.compat.v1.train.write_graph(frozen_graph, "TF_model/", "tf_model.pb", as_text=False)
