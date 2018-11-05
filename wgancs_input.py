import tensorflow as tf
import numpy as np
import math as math
FLAGS = tf.app.flags.FLAGS

# generate mask based on .mat mask
def generate_mask_mat(mask=[], mute=0):
    # shift
    mask = np.fft.ifftshift(mask)
    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('load mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
    return mask, r_factor

def setup_inputs_one_sources(sess, filenames_input, filenames_output, image_size=None, label_size=None, 
                             test=False,axis_undersample=1, capacity_factor=1, num_threads=1):
    # image size
    if image_size is None:
        if FLAGS.sample_size_y>0:
            image_size = [FLAGS.sample_size, FLAGS.sample_size_y]
        else:
            image_size = [FLAGS.sample_size, FLAGS.sample_size]
    # Read each np file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input, shuffle=False)
    key, _value_input = reader_input.read(filename_queue_input)
    
    if test is False:
        MYims= tf.py_func(read_npy,[key],tf.complex64,stateful=False)
        MYims.set_shape([17,256,320])
        image_input=MYims[8,:,:]
        MY=MYims[0:8,:,:]
        s=MYims[9::,:,:]
        reader_output = tf.WholeFileReader()
        filename_queue_output = tf.train.string_input_producer(filenames_output, shuffle=False)
        key_out, _value_output = reader_output.read(filename_queue_output)
        image_output = tf.py_func(read_npy,[key_out],tf.complex64)
        image_output.set_shape([256,320])
    else:
        MYimst = tf.py_func(read_npy,[key],tf.complex64)
        MYimst.set_shape([18,256,320])
        image_output=MYimst[-1,:,:]
        image_input=MYimst[8,:,:]
        MY=MYimst[0:8,:,:]
        s=MYimst[9:-1,:,:]

    if (FLAGS.use_phase == True):    
        pass
    else:
        image_input=abs(image_input)
        image_output=abs(image_output)
    #image_input = tf.cast(image_input, tf.complex64)
    #image_output = tf.cast(image_output, tf.complex64)
    # normalized to max 1
    ##image_input = tf.cast(tf.reduce_max(tf.abs(image_input)), tf.complex64)
    ##image_output = tf.cast(tf.reduce_max(tf.abs(image_output)), tf.complex64)

    print('image_input_complex size', image_input.get_shape())
    print('image_output_complex size', image_output.get_shape())
    print('sens map size',s.get_shape())
    print('MY size',MY.get_shape())
    image_zpad_real = tf.real(image_input)
    image_zpad_real = tf.reshape(image_zpad_real, [image_size[0], image_size[1], 1])
    image_zpad_imag = tf.imag(image_input)
    image_zpad_imag = tf.reshape(image_zpad_imag, [image_size[0], image_size[1], 1])    
    # concat to input, 2 channel for real and imag value
    image_zpad_concat = tf.concat(axis=2, values=[image_zpad_real, image_zpad_imag])
    feature = tf.reshape(image_zpad_concat, [image_size[0], image_size[1], 2])
    
    image_output_real = tf.real(image_output)
    image_output_real = tf.reshape(image_output_real, [label_size[0], label_size[1], 1]) 
    # split the complex label into real and imaginary channels
    if (FLAGS.use_phase == True):
      image_output_complex = tf.imag(image_output)
      image_output_complex = tf.reshape(image_output_complex, [label_size[0], label_size[1], 1])
      image_output_concat = tf.concat(axis=2, values=[image_output_real, image_output_complex])
      label   = tf.reshape(image_output_concat, [label_size[0], label_size[1], 2])
    else:  # use only real part
      label   = abs(image_output)
      feature = abs(image_input)
    
    # -> mask = tf.reshape(DEFAULT_MAKS_TF_c, [image_size[0], image_size[1]])

    # Using asynchronous queues
    image_in, MY, s, image_labels = tf.train.batch([feature,MY,s, label],
                                      batch_size = FLAGS.batch_size,
                                      num_threads = num_threads,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name = 'labels_and_features')

    tf.train.start_queue_runners(sess=sess)
    
    return image_in, MY, s, image_labels 

def read_npy(name):
    data=np.load(name.decode())
    return data
