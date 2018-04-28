"""
Export tensorflow checkpoint to pb model
"""

import tensorflow as tf

import DCSCN
from helper import args

FLAGS = args.get()


def main(_):
  model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
  model.build_graph()
  model.build_optimizer()
  model.build_summary_saver()

  model.init_all_variables()
  model.load_model()
  model.export_model()


if __name__ == '__main__':
  tf.app.run()
