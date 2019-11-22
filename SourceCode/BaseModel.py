import tensorflow as tf

class BaseModel():

    def __init__(self, name, save_dir):
        self.NAME = name
        self.SAVE_DIR = save_dir
        self.graph = tf.Graph()



