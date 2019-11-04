# -*-coding:utf-8-*-
import sys
sys.path.append('.')
import tensorflow as tf

import math
import numpy as np

from train_config import config as cfg

from lib.core.model.shufflenet.shufflenet import Shufflenet




def batch_norm():
    return tf.keras.layers.BatchNormalization(fused=True,
                                              momentum=0.997,
                                              epsilon=1e-5)
class SimpleFaceHeadKeypoints(tf.keras.Model):
    def __init__(self,
                 output_size=136,
                 kernel_initializer='glorot_normal'):
        super(SimpleFaceHeadKeypoints, self).__init__()

        self.output_size=output_size

        self.dense=tf.keras.layers.Dense(self.output_size,
                                         use_bias=True,
                                         kernel_initializer=kernel_initializer )

    def call(self, inputs):

        keypoints=self.dense(inputs)

        return keypoints


class SimpleFaceHeadCls(tf.keras.Model):
    def __init__(self,
                 output_size=4,
                 kernel_initializer='glorot_normal'):
        super(SimpleFaceHeadCls, self).__init__()

        self.output_size = output_size

        self.conv = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(256,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Conv2D(256,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Conv2D(self.output_size,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='valid',
                                    use_bias=True,
                                    kernel_initializer=kernel_initializer)

             ])

    def call(self, inputs):
        Cls = self.conv(inputs)
        Cls = tf.squeeze(Cls, axis=[1, 2])
        return Cls

class SimpleFaceHeadPose(tf.keras.Model):
    def __init__(self,
                 output_size=3,
                 kernel_initializer='glorot_normal'):
        super(SimpleFaceHeadPose, self).__init__()

        self.output_size = output_size



        self.conv=tf.keras.Sequential(
            [tf.keras.layers.Conv2D(256,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Conv2D(256,
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Conv2D(self.output_size,
                                kernel_size=(3, 3),
                                strides=1,
                                padding='valid',
                                use_bias=True,
                                kernel_initializer=kernel_initializer)

            ])



    def call(self, inputs):

        PRY = self.conv(inputs)


        PRY=tf.squeeze(PRY,axis=[1,2])
        return PRY


class SimpleFace(tf.keras.Model):

    def __init__(self,kernel_initializer='glorot_normal'):
        super(SimpleFace, self).__init__()

        model_size=cfg.MODEL.net_structure.split('_',1)[-1]
        self.backbone = Shufflenet(model_size=model_size,
                                   kernel_initializer=kernel_initializer)

        self.head_keypoints=SimpleFaceHeadKeypoints(kernel_initializer=kernel_initializer)
        self.head_pose = SimpleFaceHeadPose(kernel_initializer=kernel_initializer)
        self.head_cls = SimpleFaceHeadCls(kernel_initializer=kernel_initializer)

        self.pool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()


    @tf.function
    def call(self, inputs, training=False):
        inputs=self.preprocess(inputs)
        x1, x2, x3,x4 = self.backbone(inputs, training=training)


        s2 = self.pool1(x2)
        s3 = self.pool2(x3)
        s4 = self.pool3(x4)

        multi_scale = tf.concat([s2, s3, s4], 1)

        keypoints_predict=self.head_keypoints(multi_scale,training=training)

        head_pose_predict=self.head_pose(x2,training=training)

        head_cls_predict = self.head_cls(x2, training=training)

        res=tf.concat([keypoints_predict,head_pose_predict,head_cls_predict],axis=1)

        return res



    @tf.function(input_signature=[tf.TensorSpec([None,cfg.MODEL.hin,cfg.MODEL.win,3], tf.float32)])
    def inference(self,images):
        inputs = self.preprocess(images)
        x1,x2,x3,x4 = self.backbone(inputs, training=False)
        print(x4)
        s2 = self.pool1(x2)
        s3 = self.pool2(x3)
        s4 = self.pool3(x4)

        multi_scale = tf.concat([s2, s3, s4], 1)

        keypoints_predict = self.head_keypoints(multi_scale, training=False)


        return keypoints_predict



    def preprocess(self,image):

        mean = cfg.DATA.PIXEL_MEAN
        std =  cfg.DATA.PIXEL_STD

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_invstd

        return image




if __name__=='__main__':


    import time
    model = SimpleFace()

    image=np.zeros(shape=(1,160,160,3),dtype=np.float32)
    x=model.inference(image)
    #tf.saved_model.save(model,'./model/keypoints')
    start=time.time()
    for i in range(100):
        x = model.inference(image)

    print('xxxyyyy',(time.time()-start)/100.)








def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0,weights=1.):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = tf.abs(x)
    losses = tf.where(
        tf.greater(w, absolute_x),
        w * tf.math.log(1.0 + absolute_x / epsilon),
        absolute_x - c
    )
    losses=losses*cfg.DATA.weights
    loss = tf.reduce_sum(tf.reduce_mean(losses*weights, axis=[0]))

    return loss

def _mse(landmarks, labels,weights=1.):

    return tf.reduce_mean(0.5*tf.square(landmarks - labels)*weights)

def l1(landmarks, labels):
    return tf.reduce_mean(landmarks - labels)

def calculate_loss(predict_keypoints, label_keypoints):
    

    landmark_label =      label_keypoints[:, 0:136]
    pose_label =          label_keypoints[:, 136:139]
    leye_cls_label =      label_keypoints[:, 139]
    reye_cls_label =      label_keypoints[:, 140]
    mouth_cls_label =     label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]


    landmark_predict =     predict_keypoints[:, 0:136]
    pose_predict =         predict_keypoints[:, 136:139]
    leye_cls_predict =     predict_keypoints[:, 139]
    reye_cls_predict =     predict_keypoints[:, 140]
    mouth_cls_predict =     predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]


    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)

    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cls_predict,
                                                                      labels=leye_cls_label) )
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cls_predict,
                                                                      labels=reye_cls_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cls_predict,
                                                                       labels=mouth_cls_label))
    mouth_loss_big = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=big_mouth_cls_predict,
                                                                        labels=big_mouth_cls_label))
    mouth_loss=mouth_loss+mouth_loss_big

    return loss+loss_pose+leye_loss+reye_loss+mouth_loss


