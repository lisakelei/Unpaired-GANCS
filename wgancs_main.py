"""
train on GAN-CS 
example
export CUDA_VISIBLE_DEVICES=0 

python3 main.py  --run train \
                      --dataset_train /mnt/raid5/morteza/datasets/Abdominal-DCE-616cases/train\   
                      --dataset_test /mnt/raid5/morteza/datasets/Abdominal-DCE-616cases/test\   
                      --sample_size 256 \  
                      --sample_size_y 128 \
                      --batch_size 8  \ 
                      --summary_period  1000 \
                      --sample_test 128 \
                      --sample_train -1 \
                      --subsample_test 8 \
                      --subsample_train 10000 \
                      --train_time 6000 \ 
                      --R_seed -1 \
                      --R_alpha 2 \ 
                      --R_factor 10 \
                      --train_dir /mnt/raid5/morteza/GANCS-MRI/train_save_all
"""

import wgancs_input
import wgancs_model
import wgancs_train
import os.path
import random
import numpy as np
import numpy.random
import math
import tensorflow as tf
import shutil, os, errno # utils handling file manipulation
from scipy import io as sio # .mat I/O


FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)

tf.app.flags.DEFINE_string('activation','relu',
                            "activation to use for disc")

tf.app.flags.DEFINE_string('activation_G','relu',
                            "activation to use for gene")

tf.app.flags.DEFINE_string('architecture','resnet',
                            "model arch used for generator, ex: resnet, aec, pool")

tf.app.flags.DEFINE_integer('axis_undersample', 1,
                            "which axis to undersample")

tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_integer('starting_batch', 0,
                            "Starting batch count, use when resume from ckpt.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 999,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset_label', '',
                           "Path to the train dataset label directory.")

tf.app.flags.DEFINE_string('dataset_train', '',
                           "Path to the train dataset input directory.")

tf.app.flags.DEFINE_string('dataset_test', '',
                           "Path to the test dataset directory.")

tf.app.flags.DEFINE_string('disc_opti', 'adam',
                            "optimizer to use for discriminator")

tf.app.flags.DEFINE_float('disc_dropp', 0.0,
                          "drop prob for disc dropout layer: 0 is no dropout")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_bool('FM', False,
                         "Whether to use feature matching.")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")   #demo

tf.app.flags.DEFINE_float('gene_log_factor', 0,
                          "Multiplier for generator fool loss term, weighting log-loss vs LS loss")

tf.app.flags.DEFINE_float('gene_dc_factor', 0,
                          "Multiplier for generator data-consistency L2 loss term for data consistency, weighting Data-Consistency with GD-loss for GAN-loss")

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.97,
                            "specified the max gpu fraction used per device")

tf.app.flags.DEFINE_integer('hybrid_disc', 0,
                            "whether/level to augment discriminator input to image+kspace hybrid space.")

tf.app.flags.DEFINE_float('learning_beta1', 0.9,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.000001,
                          "Starting learning rate used for AdamOptimizer")  #0.000001

tf.app.flags.DEFINE_integer('learning_rate_half_life', 100000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('mse_batch', -200,
                            "Number of batches to run with pure mse loss.")

tf.app.flags.DEFINE_integer('number_of_copies', 3,
                            "Number of repeatitions for the generator network.")

tf.app.flags.DEFINE_integer('sample_size', 256,
                            "Image sample height in pixels.")

tf.app.flags.DEFINE_integer('sample_size_y', 320,
                            "Image sample width in pixels)

tf.app.flags.DEFINE_integer('label_size', -1,
                            "Good Image height in pixels. by default same as sample_size")

tf.app.flags.DEFINE_integer('label_size_x', -1,
                            "Good Image width in pixels. by default same as sample_size_y")

tf.app.flags.DEFINE_integer('summary_period', 2000,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('summary_train_period', 50,
                            "Number of batches between train data dumps")

tf.app.flags.DEFINE_bool('permutation_split', False,
                         "Whether to randomly permutate order of input and label.")

tf.app.flags.DEFINE_bool('permutation_train', True,
                         "Whether to randomly permutate order for training sub-samples.")

tf.app.flags.DEFINE_bool('permutation_test', False,
                         "Whether to randomly permutate order for testing sub-samples.")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('sample_test', -1,
                            "Number of features to use for testing.")

tf.app.flags.DEFINE_integer('sample_train', -1,
                            "Number of features to use for train. default value is -1 for use all samples except testing samples")
                         
tf.app.flags.DEFINE_string('sampling_pattern', '',
                            "specifed file path for undersampling")

tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 1500,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_float('R_factor', 4,
                            "desired reducton/undersampling factor")

tf.app.flags.DEFINE_float('R_alpha', 2,
                            "desired variable density parameter x^alpha")

tf.app.flags.DEFINE_integer('R_seed', -1,
                            "specifed sampling seed to generate undersampling, -1 for randomized sampling")

tf.app.flags.DEFINE_bool('use_patches', False,
                            "whether to patch generator output when feeding to disc")

tf.app.flags.DEFINE_bool('use_phase', True,
                            "whether to use two channels for both magnitude and phase")

tf.app.flags.DEFINE_bool('wgan_gp', True, 
                         "whether to use WGAN-GP instead of LSGAN")

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess , _oldwriter = setup_tensorflow()

    # image_size
    image_size = [FLAGS.sample_size, FLAGS.sample_size_y]

    # label_size
    if FLAGS.label_size<0:
        label_size = [FLAGS.sample_size, FLAGS.sample_size_y]
    else:
        label_size = [FLAGS.label_size, FLAGS.label_size_x]

    # Prepare train and test directories (SEPARATE FOLDER)
    prepare_dirs(delete_train_dir=False, shuffle_filename=False)
    # if not specify use the same as input
    if FLAGS.dataset_label == '':
        FLAGS.dataset_label = FLAGS.dataset_train
    filenames_input_train = get_filenames(dir_file=FLAGS.dataset_train, shuffle_filename=False)
    filenames_output_train = get_filenames(dir_file=FLAGS.dataset_label, shuffle_filename=False)
    num_filenames_input,num_filenames_output = len(filenames_input_train),len(filenames_output_train)
    filenames_output_train *= math.ceil(num_filenames_input/num_filenames_output)
    filenames_input_test = get_filenames(dir_file=FLAGS.dataset_test, shuffle_filename=False)
    filenames_output_test = get_filenames(dir_file=FLAGS.dataset_test, shuffle_filename=False)

    # check input and output sample number matches (SEPARATE FOLDER)
    assert(num_filenames_input<=len(filenames_output_train))
    num_filename_train = len(filenames_input_train)
    assert(len(filenames_input_test)==len(filenames_output_test))
    num_filename_test = len(filenames_input_test)

    # Permutate train and test split (SEPARATE FOLDERS)
    index_permutation_split = random.sample(range(num_filename_train), num_filename_train)
    filenames_input_train = [filenames_input_train[x] for x in index_permutation_split]
    if FLAGS.dataset_label != FLAGS.dataset_train:  
        index_permutation_split = random.sample(range(len(filenames_output_train)), num_filename_train)
    elif FLAGS.permutation_split:
        index_permutation_split = random.sample(range(num_filename_train), num_filename_train)      
    filenames_output_train = [filenames_output_train[x] for x in index_permutation_split]

    # Permutate test split (SAME FOLDERS)
    '''if FLAGS.permutation_split: # do not permutate test for now
        index_permutation_split = random.sample(range(num_filename_test), num_filename_test)
        filenames_input_test = [filenames_input_test[x] for x in index_permutation_split]
        filenames_output_test = [filenames_output_test[x] for x in index_permutation_split]'''
    print("First three filenames_output_Test",filenames_output_test[0:3])
    print("First three filenames_Input_train",filenames_input_train[0:3])
    print("First three filenames_Output_train",filenames_output_train[0:3])

    # Sample training and test sets (SEPARATE FOLDERS)
    train_filenames_input = filenames_input_train[:FLAGS.sample_train]    
    train_filenames_output = filenames_output_train[:FLAGS.sample_train]            
    test_filenames_input  = filenames_input_test[:FLAGS.sample_test]
    test_filenames_output  = filenames_output_test[:FLAGS.sample_test]

    # get undersample mask
    from scipy import io as sio
    try:
        content_mask = sio.loadmat(FLAGS.sampling_pattern)
        key_mask = [x for x in content_mask.keys() if not x.startswith('_')]
        mask = content_mask[key_mask[0]]
    except:
        mask = None
        print("[warining] NO MASK PATTERN!!!")
    # Setup async input queues
    train_features, train_labels, train_masks = wgancs_input.setup_inputs_one_sources(sess, train_filenames_input, 
                                                                                      train_filenames_output, 
                                                                        image_size=image_size, 
                                                                        label_size=label_size,
                                                                        axis_undersample=FLAGS.axis_undersample, 
                                                                        r_factor=FLAGS.R_factor,
                                                                        r_alpha=FLAGS.R_alpha,
                                                                        r_seed=FLAGS.R_seed,
                                                                        sampling_mask=mask
                                                                        )
    test_features,  test_labels, test_masks = wgancs_input.setup_inputs_one_sources(sess, test_filenames_input, 
                                                                                    test_filenames_output,
                                                                        image_size=image_size, 
                                                                        label_size=label_size,
                                                                        # undersampling
                                                                        axis_undersample=FLAGS.axis_undersample, 
                                                                        r_factor=FLAGS.R_factor,
                                                                        r_alpha=FLAGS.R_alpha,
                                                                        r_seed=FLAGS.R_seed,
                                                                        sampling_mask=mask
                                                                        )
    
    print('train_features_queue', train_features.get_shape())
    print('train_labels_queue', train_labels.get_shape())
    print('train_masks_queue', train_masks.get_shape())
    num_sample_train = len(train_filenames_input)
    num_sample_test = len(test_filenames_input)
    print('train on {0} input, {1} label, test on {2} samples'.format(num_filenames_input,num_filenames_output, num_sample_test))

    # Add some noise during training (think denoising autoencoders)
    noise_level = .00
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput, gene_output, gene_var_list, gene_layers, gene_mlayers, disc_real_output, disc_fake_output, disc_var_list, disc_layers_X, disc_layers_Z] = \
            wgancs_model.create_model(sess, noisy_train_features, train_labels, train_masks, architecture=FLAGS.architecture)

    gene_loss, gene_dc_loss, gene_ls_loss, list_gene_losses, gene_mse_factor = \
                     wgancs_model.create_generator_loss(disc_fake_output, gene_output, train_features, train_labels, train_masks,disc_layers_X, disc_layers_Z)
    
    # WGAN-GP
    disc_loss,disc_fake_loss,disc_real_loss = wgancs_model.create_discriminator_loss(disc_real_output, disc_fake_output, \
                                                    real_data = tf.identity(train_labels), fake_data = tf.abs(gene_output))

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            wgancs_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)
    #summary_op=tf.summary.merge_all()
    # Restore variables from checkpoint
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    metafile=filename+'.meta'
    if tf.gfile.Exists(metafile):
        saver = tf.train.Saver()
        print("Loading checkpoint from file `%s'" % (filename,))
        saver.restore(sess, filename)
    else:
        print("No checkpoint `%s', train from scratch" % (filename,))
        sess.run(tf.global_variables_initializer())
    # Train model
    train_data = TrainData(locals())
    wgancs_train.train_model(train_data, FLAGS.starting_batch, num_sample_train, num_sample_test)



def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def prepare_dirs(delete_train_dir=False, shuffle_filename=True):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        try:
            if tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.DeleteRecursively(FLAGS.train_dir)
            tf.gfile.MakeDirs(FLAGS.train_dir)
        except:
            try:
                shutil.rmtree(FLAGS.train_dir)
            except:
                print('fail to delete train dir {0} using tf.gfile, will use shutil'.format(FLAGS.train_dir))
            mkdirp(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset_train) or \
       not tf.gfile.IsDirectory(FLAGS.dataset_train):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset_train,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset_train)
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset_train, f) for f in filenames]

    return filenames

def get_filenames(dir_file='', shuffle_filename=False):
    try:
        filenames = tf.gfile.ListDirectory(dir_file)
    except:
        print('cannot get files from {0}'.format(dir_file))
        return []
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    else:
        filenames = sorted(filenames)
    filenames = [os.path.join(dir_file, f) for f in filenames if f.endswith('.jpg')]
    return filenames

def setup_tensorflow(gpu_memory_fraction=1.0):
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = min(gpu_memory_fraction, FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=config)
    #print('TF session setup for gpu usage cap of {0}'.format(config.gpu_options.per_process_gpu_memory_fraction))

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

# SummaryWriter is deprecated
# tf.summary.FileWriter.
    #summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess ,None # summary_writer   

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, _oldwriter = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = wgancs_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            wgancs_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    wgancs_demo.demo1(sess)
    
def main(argv=None):
    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()

