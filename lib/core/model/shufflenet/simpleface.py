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
                 output_size=136-22-22,
                 kernel_initializer='glorot_normal'):
        super(SimpleFaceHeadKeypoints, self).__init__()

        self.output_size=output_size

        self.profile=tf.keras.layers.Dense(34,
                                         use_bias=True,
                                         kernel_initializer=kernel_initializer )

        self.noses = tf.keras.layers.Dense(18,
                                           use_bias=True,
                                           kernel_initializer=kernel_initializer)

        self.mouth = tf.keras.layers.Dense(40,
                                           use_bias=True,
                                           kernel_initializer=kernel_initializer)
        self.cls=tf.keras.layers.Dense(2,
                                           use_bias=True,
                                           kernel_initializer=kernel_initializer)


    def call(self, inputs,training):

        kp_profile=self.profile(inputs,training=training)
        kp_nose = self.noses(inputs,training=training)
        kp_mouth = self.mouth(inputs, training=training)
        cls = self.cls(inputs,training=training)
        return kp_profile,kp_nose,kp_mouth,cls


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



    def call(self, inputs,training):

        PRY = self.conv(inputs,training=training)

        PRY=tf.squeeze(PRY,axis=[1,2])
        return PRY

class OneEye(tf.keras.Model):
    def __init__(self,
                 output_size=23,
                 kernel_initializer='glorot_normal'):
        super(OneEye, self).__init__()

        self.output_size=output_size

        self.conv = tf.keras.Sequential(
            [tf.keras.layers.SeparableConv2D(256,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.SeparableConv2D(256,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=kernel_initializer),
             batch_norm(),
             tf.keras.layers.ReLU(),


             ])

        self.kp_eyebow=tf.keras.layers.Conv2D(10,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='valid',
                               use_bias=True,
                               kernel_initializer=kernel_initializer)
        self.kp_eye = tf.keras.layers.Conv2D(12,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding='valid',
                                                use_bias=True,
                                                kernel_initializer=kernel_initializer)

        self.cls = tf.keras.layers.Conv2D(1,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='valid',
                                         use_bias=True,
                                         kernel_initializer=kernel_initializer)

    def call(self, inputs,training):

        fm=self.conv(inputs,training=training)

        eyebow=tf.squeeze(self.kp_eyebow(fm,training=training),axis=[1,2])
        eye = tf.squeeze(self.kp_eye(fm, training=training),axis=[1,2])
        cls=tf.squeeze(self.cls(fm,training=training),axis=[1,2])
        return eyebow,eye,cls




class SimpleFace(tf.keras.Model):

    def __init__(self,kernel_initializer='glorot_normal'):
        super(SimpleFace, self).__init__()

        model_size=cfg.MODEL.net_structure.split('_',1)[-1]
        self.backbone = Shufflenet(model_size=model_size,
                                   kernel_initializer=kernel_initializer)



        self.pool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()

        self.head_keypoints = SimpleFaceHeadKeypoints(kernel_initializer=kernel_initializer)

        self.head_pose = SimpleFaceHeadPose(kernel_initializer=kernel_initializer)


        self.left_eye=OneEye(kernel_initializer=kernel_initializer)

        self.right_eye = OneEye(kernel_initializer=kernel_initializer)


    @tf.function
    def call(self, inputs, training=False):
        inputs=self.preprocess(inputs)
        x1, x2, x3 = self.backbone(inputs, training=training)


        s1 = self.pool1(x1)
        s2 = self.pool2(x2)
        s3 = self.pool3(x3)

        multi_scale = tf.concat([s1, s2, s3], 1)

        kp_profile, kp_nose,kp_mouth, mouthcls=self.head_keypoints(multi_scale,training=training)

        head_pose_predict=self.head_pose(x2,training=training)



        leyebow,leye,leyecls=self.left_eye(x2, training=training)
        reyebow, reye, reyecls = self.left_eye(x2, training=training)





        keypoints=tf.concat([kp_profile,leyebow,reyebow,kp_nose,leye,reye,kp_mouth,head_pose_predict,leyecls,reyecls,mouthcls],axis=1)



        return keypoints



    @tf.function(input_signature=[tf.TensorSpec([None,cfg.MODEL.hin,cfg.MODEL.win,3], tf.float32)])
    def inference(self,inputs):
        inputs = self.preprocess(inputs)

        x1, x2, x3 = self.backbone(inputs, training=False)

        s1 = self.pool1(x1)
        s2 = self.pool2(x2)
        s3 = self.pool3(x3)

        multi_scale = tf.concat([s1, s2, s3], 1)

        kp_profile, kp_nose, kp_mouth, mouthcls = self.head_keypoints(multi_scale, training=False)

        head_pose_predict = self.head_pose(x2, training=False)

        leyebow, leye, leyecls = self.left_eye(x2, training=False)
        reyebow, reye, reyecls = self.left_eye(x2, training=False)

        keypoints = tf.concat(
            [kp_profile, leyebow, reyebow, kp_nose, leye, reye, kp_mouth, head_pose_predict, leyecls, reyecls,
             mouthcls], axis=1)


        print(keypoints.shape)
        return keypoints

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


