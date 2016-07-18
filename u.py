import tensorflow as tf

def get_dim(X,index):
     shape=X.get_shape()
     return int(shape[index])


class u:

    # data is a 3 dim matrix:
    # 1) layers 2) number of data points 3) point dimensionality
    def __init__(self, data):
        self.data = data
        self.layerNum = get_dim(data, 1)
        self.pointIndex = get_dim(data, 2)
        self.pointDim = get_dim(data, 3)
        self.q = self.calculateQ(data)

    def calculateQ(self, data):
        #TODO
        return None

    # indexing from 1
    def getLayer(self, layer):
        return tf.slice(self.data, [layer -1, 0, 0], [1, -1, -1])

    def getPoint(self, layer, index):
        return tf.slice(self.data, [layer - 1, index - 1, 0], [1, 1, -1])

    def getValue(self, layer, index, dimension):
        return tf.slice(self.data, [layer -1, index - 1, dimension -1], [1, 1, 1])

