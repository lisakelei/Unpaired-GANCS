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
                             axis_undersample=1, capacity_factor=1, 
                             r_factor=4, r_alpha=0, r_seed=0,
                             sampling_mask=None, num_threads=1):

    # image size
    if image_size is None:
        if FLAGS.sample_size_y>0:
            image_size = [FLAGS.sample_size, FLAGS.sample_size_y]
        else:
            image_size = [FLAGS.sample_size, FLAGS.sample_size]

    # generate default mask
    if sampling_mask is None:
        print('nomask!!')
    else:
        # get input mask
        DEFAULT_MASK, _ = generate_mask_mat(sampling_mask)

    # convert to complex tf tensor
    DEFAULT_MAKS_TF = tf.cast(tf.constant(DEFAULT_MASK), tf.float32)
    DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)

    # Read each np file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input, shuffle=False)
    _key, value_input = reader_input.read(filename_queue_input)

    reader_output = tf.WholeFileReader()
    filename_queue_output = tf.train.string_input_producer(filenames_output, shuffle=False)
    _key, value_output = reader_output.read(filename_queue_output)
    channels = 3

    image_input = tf.image.decode_jpeg(value_input, channels=channels, name="input_image")
    image_input.set_shape([image_size[0], 2*image_size[0], channels])
    image_output = tf.image.decode_jpeg(value_output, channels=channels, name="output_image")
    image_output.set_shape([label_size[0], 2*label_size[0], channels])
    print('size I/O_image', image_input.get_shape(), image_output.get_shape())

    image_input = image_input[:,:,-1]   
    image_output = image_output[:,:,-1]

    #choose the complex-valued image
    image_input_mag = tf.cast(image_input[0:image_size[0],0:image_size[1]], tf.complex64)
    image_output_mag = tf.cast(image_output[0:label_size[0],0:label_size[1]], tf.complex64)
    if (FLAGS.use_phase == True):    
      image_input_phase = tf.cast(8*tf.constant(math.pi), tf.complex64)*tf.cast(image_input[0:image_size[0],image_size[1]:2*image_size[1]], tf.complex64)
      image_input = tf.multiply(image_input_mag, tf.exp(tf.sqrt(tf.cast(-1,tf.complex64))*image_input_phase))
      image_output_phase = tf.cast(8*tf.constant(math.pi), tf.complex64)*tf.cast(image_output[0:label_size[0],label_size[1]:2*label_size[1]], tf.complex64)
      image_output = tf.multiply(image_output_mag, tf.exp(tf.sqrt(tf.cast(-1,tf.complex64))*image_output_phase))
    else:
      image_input=image_input_mag
      image_output=image_output_mag
    image_input = tf.cast(image_input, tf.complex64)
    image_output = tf.cast(image_output, tf.complex64)
# output, gold-standard
    image_input = image_input / 255.0     #tf.cast(tf.reduce_max(tf.abs(image_input)), tf.complex64)
    image_output = image_output / 255.0

    print('image_input_complex size', image_input.get_shape())
    print('image_output_complex size', image_output.get_shape())

    # apply undersampling mask
    kspace_input = tf.fft2d(tf.cast(image_input,tf.complex64))
    if (FLAGS.sampling_pattern!="nomask"):
      kspace_zpad = kspace_input * DEFAULT_MAKS_TF_c
    else:
      kspace_zpad = kspace_input
    # zpad undersampled image for input
    image_zpad = tf.ifft2d(kspace_zpad)
    image_zpad_real = tf.real(image_zpad)
    image_zpad_real = tf.reshape(image_zpad_real, [image_size[0], image_size[1], 1])
    image_zpad_imag = tf.imag(image_zpad)
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

      # The feature is zpad image with 2 channel, label is the ground-truth real-valued image
      label   = tf.reshape(image_output_concat, [label_size[0], label_size[1], 2])
    else:  # use only real part
      label   = image_output_real
    
    mask = tf.reshape(DEFAULT_MAKS_TF_c, [image_size[0], image_size[1]])

    # Using asynchronous queues
    features, labels, masks = tf.train.batch([feature, label, mask],
                                      batch_size = FLAGS.batch_size,
                                      num_threads = num_threads,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name = 'labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels, masks    

