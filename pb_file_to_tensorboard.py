# run the following command in terminal to get tensorboard:
# tensorboard --logdir = /logs/test/1/

import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # model_filename = "/Users/Mohamad/Sensifai/FaceNet/data/ms-celeb-1m/20170512-110547.pb"
    model_filename = "/Users/Mohamad/Sensifai/FaceNet/data/CASIA-WebFace/20170511-185253.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/logs/tests/1/'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
