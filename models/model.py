import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from loss_smooth import *
import cfsm

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class,FLAGS, GRAPH, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    # l0_xyz [12 4096 3]
    l0_points = point_cloud[:, :, 3:]
    # l0_points [12 4096 6]
    end_points['l0_xyz'] = l0_xyz

    l1_xyz_sem, l1_points_sem, l1_indices_sem = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1,
                                                                   nsample=32,
                                                                   mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer1_sem')
    l1_xyz_ins, l1_points_ins, l1_indices_ins = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1,
                                                                   nsample=32,
                                                                   mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer1_ins')
    l1_points_ins, l1_points_sem = cfsm.twin_cfsm(l1_points_ins, l1_points_sem, 64, 1024, 1,FLAGS, GRAPH)

    # [12 1024 3]  [12 1024 64] [12 1024 32]
    l2_xyz_sem, l2_points_sem, l2_indices_sem = pointnet_sa_module(l1_xyz_sem, l1_points_sem, npoint=256, radius=0.2,
                                                                   nsample=32,
                                                                   mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer2_sem')
    l2_xyz_ins, l2_points_ins, l2_indices_ins = pointnet_sa_module(l1_xyz_ins, l1_points_ins, npoint=256, radius=0.2,
                                                                   nsample=32,
                                                                   mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer2_ins')
    l2_points_ins, l2_points_sem = cfsm.twin_cfsm(l2_points_ins, l2_points_sem, 128, 256, 2,FLAGS, GRAPH)

    # [12 256 3]   [12 256 128] [12 256 32]
    l3_xyz_sem, l3_points_sem, l3_indices_sem = pointnet_sa_module(l2_xyz_sem, l2_points_sem, npoint=64, radius=0.4,
                                                                   nsample=32,
                                                                   mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer3_sem')
    l3_xyz_ins, l3_points_ins, l3_indices_ins = pointnet_sa_module(l2_xyz_ins, l2_points_ins, npoint=64, radius=0.4,
                                                                   nsample=32,
                                                                   mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer3_ins')
    l3_points_ins, l3_points_sem = cfsm.twin_cfsm(l3_points_ins, l3_points_sem, 256, 64, 3,FLAGS, GRAPH)

    # [12 64 3]    [12 64 256]  [12 64 32]
    l4_xyz_sem, l4_points_sem, l4_indices_sem = pointnet_sa_module(l3_xyz_sem, l3_points_sem, npoint=16, radius=0.8,
                                                                   nsample=32,
                                                                   mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer4_sem')
    l4_xyz_ins, l4_points_ins, l4_indices_ins = pointnet_sa_module(l3_xyz_ins, l3_points_ins, npoint=16, radius=0.8,
                                                                   nsample=32,
                                                                   mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='layer4_ins')
    l4_points_ins, l4_points_sem = cfsm.twin_cfsm(l4_points_ins, l4_points_sem, 512, 16, 4,FLAGS, GRAPH)
    # [12 16 3]    [12 16 512]  [12 16 32]

    # l3_points_sem [12 64 256]
    l3_points_sem_dec = pointnet_fp_module(l3_xyz_sem, l4_xyz_sem, l3_points_sem, l4_points_sem, [256, 256],
                                           is_training, bn_decay,
                                           scope='sem_fa_layer1')
    l3_points_ins_dec = pointnet_fp_module(l3_xyz_ins, l4_xyz_ins, l3_points_ins, l4_points_ins, [256, 256],
                                           is_training, bn_decay,
                                           scope='ins_fa_layer1')

    # l2_points_sem [12 256 256]
    l2_points_sem_dec = pointnet_fp_module(l2_xyz_sem, l3_xyz_sem, l2_points_sem, l3_points_sem_dec, [256, 256],
                                           is_training, bn_decay,
                                           scope='sem_fa_layer2')
    l2_points_ins_dec = pointnet_fp_module(l2_xyz_ins, l3_xyz_ins, l2_points_ins, l3_points_ins_dec, [256, 256],
                                           is_training, bn_decay,
                                           scope='ins_fa_layer2')

    # l1_points_sem [12 1024 128]
    l1_points_sem_dec = pointnet_fp_module(l1_xyz_sem, l2_xyz_sem, l1_points_sem, l2_points_sem_dec, [256, 128],
                                           is_training, bn_decay,
                                           scope='sem_fa_layer3')
    l1_points_ins_dec = pointnet_fp_module(l1_xyz_ins, l2_xyz_ins, l1_points_ins, l2_points_ins_dec, [256, 128],
                                           is_training, bn_decay,
                                           scope='ins_fa_layer3')

    # l0_points_sem [12 4096 128]
    l0_points_sem_dec = pointnet_fp_module(l0_xyz, l1_xyz_sem, l0_points, l1_points_sem_dec, [128, 128, 128],
                                           is_training, bn_decay,
                                           scope='sem_fa_layer4')
    l0_points_ins_dec = pointnet_fp_module(l0_xyz, l1_xyz_ins, l0_points, l1_points_ins_dec, [128, 128, 128],
                                           is_training, bn_decay,
                                           scope='ins_fa_layer4')
    net_ins = tf_util.conv1d(l0_points_ins_dec, 128, 1, padding='VALID', bn=True, is_training=is_training,
                             scope='ins_fc5',
                             bn_decay=bn_decay)
    net_sem = tf_util.conv1d(l0_points_sem_dec, 128, 1, padding='VALID', bn=True, is_training=is_training,
                             scope='sem_fc5',
                             bn_decay=bn_decay)

    net_ins = tf_util.dropout(net_ins, keep_prob=0.5, is_training=is_training, scope='ins_dp1')
    net_ins = tf_util.conv1d(net_ins, 5, 1, padding='VALID', activation_fn=None, scope='ins_fc6')
    net_sem = tf_util.dropout(net_sem, keep_prob=0.5, is_training=is_training, scope='sem_dp1')
    net_sem = tf_util.conv1d(net_sem, num_class, 1, padding='VALID', activation_fn=None, scope='sem_fc6')

    return net_sem, net_ins  # net_sem [12 4096 13]  net_ins [12 4096 5]


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label,num_class):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    delta_sem = 0.95
    K = 100
    sem_label = tf.one_hot(sem_label, num_class)
    logits = tf.nn.softmax(pred_sem)
    logits_log = tf.log(tf.clip_by_value(logits, 1e-12, 1.0))
    classify_loss = -tf.reduce_mean(sem_label * logits_log * tf.sigmoid(K * (delta_sem - logits))) * num_class
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape().as_list()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.001

    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, feature_dim,
                                                          delta_v, delta_d, param_var, param_dist, param_reg)

    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist, l_reg


