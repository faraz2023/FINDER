# -*- coding: utf-8 -*-
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
from FINDER import FINDER

def main():

    print("GPU: ", tf.config.list_physical_devices('GPU'))
    dqn = FINDER()
    dqn.Train()


if __name__=="__main__":
    main()
