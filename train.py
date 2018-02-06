from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys

sys.path.append('./models/research/slim/')
from nets import nets_factory

from utils.custom_preprocessing import preprocessing_factory
from utils.dataset import CDiscountDataset


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'crop_and_resize', 'The name of the preprocessing to use. If left '
                                             'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        dataset = CDiscountDataset(FLAGS.dataset_dir)

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=dataset.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        train_preprocessing_fn = preprocessing_factory(FLAGS.preprocessing_name,
                                                       output_height=FLAGS.train_image_size,
                                                       output_width=FLAGS.train_image_size,
                                                       is_training=True)

        train_iterator = dataset.get_train_iterator(epochs=-1, batch_size=FLAGS.batch_size,
                                                    preprocessing_function=train_preprocessing_fn, shuffle=True)

        images, labels = train_iterator.get_next()
        logits, end_points = network_fn(images)

        if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], labels,
                label_smoothing=FLAGS.label_smoothing, weights=0.4,
                scope='aux_loss')

        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for end_points.
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, tf.get_variable_scope().name):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # Configure the optimization procedure.
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', FLAGS.learning_rate))

        total_loss = tf.losses.get_total_loss()

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            variables_to_train=_get_variables_to_train())
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('training/train_loss', total_loss))

        # Train accuracy
        def calculate_accuracy(images, labels):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                logits, _ = network_fn(images)
                predictions = tf.nn.softmax(logits)
                predicted_labels = tf.to_int32(tf.argmax(predictions, axis=1))
                accuracy = tf.reduce_mean(tf.to_float(tf.equal(predicted_labels, labels)))
                return accuracy

        summaries.add(tf.summary.scalar('training/train_accuracy', calculate_accuracy(*train_iterator.get_next())))

        val_preprocessing_fn = preprocessing_factory(FLAGS.preprocessing_name,
                                                     output_height=FLAGS.train_image_size,
                                                     output_width=FLAGS.train_image_size,
                                                     is_training=False)
        val_iterator = dataset.get_val_iterator(epochs=-1, batch_size=3 * FLAGS.batch_size,
                                                preprocessing_function=val_preprocessing_fn,
                                                shuffle=True)

        summaries.add(tf.summary.scalar('training/val_accuracy', calculate_accuracy(*val_iterator.get_next())))

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, tf.get_variable_scope().name))

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Kicks off the training.
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            local_init_op=tf.group(train_iterator.initializer, val_iterator.initializer),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            summary_writer=summary_writer,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=tfconfig)


if __name__ == '__main__':
    tf.app.run()
