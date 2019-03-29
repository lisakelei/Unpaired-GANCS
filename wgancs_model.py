import numpy as np
import tensorflow as tf
import mri_model

FLAGS = tf.app.flags.FLAGS

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        
        return '%s_L%03d' % (self.name, layer+1)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str()):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('weight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str()):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keepdims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self      

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            if (FLAGS.activation_G=='lrelu'):
                self.add_lrelu()
            else:
                self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        if (FLAGS.activation_G=='lrelu'):
            self.add_lrelu()
        else:
            self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)
        self.add_batch_norm()
        if (FLAGS.activation_G=='lrelu'):
            self.add_lrelu()
        else:
            self.add_relu()
            
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        if (FLAGS.activation_G=='lrelu'):
            self.add_lrelu()
        else:
            self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("prev %s term %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self, size=None):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        if size is None:
            size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self    

    def add_concat(self, layer_add):
        last_layer = self.get_output()
        prev_shape = last_layer.get_shape()
        try:
            out = tf.concat(axis = 3, values = [last_layer, layer_add])
            self.outputs.append(out)
        except:
            print('fail to concat {0} and {1}'.format(last_layer, layer_add))
        return self    

    def add_layer(self, layer_add):
        self.outputs.append(layer_add)
        return self 

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def _discriminator_model(sess, features, disc_input, layer_output_skip=5, hybrid_disc=0):

    # update 05092017, hybrid_disc consider whether to use hybrid space for discriminator
    # to study the kspace distribution/smoothness properties

    # Fully convolutional model
    mapsize = 3
    layers  = [4, 8, 16, 32]#[8,8, 16,32, 64]#[4, 8, 16, 32]#[8, 16, 32, 64]#[64, 128, 256, 512]
    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    
    # apply dropout to inputs to the disc
    if FLAGS.disc_dropp > 0.0: 
        disc_input=tf.nn.dropout(disc_input, 1.0-FLAGS.disc_dropp)

    # augment data to hybrid domain = image+kspace
    if hybrid_disc>0:
        disc_size = tf.shape(disc_input)#disc_input.get_shape()
        #print("DISCINPUT",disc_size)        
        disc_kspace = Fourier(disc_input, separate_complex=True)
        disc_kspace_real = tf.cast(tf.real(disc_kspace), tf.float32)
        #print(disc_kspace_real)
        disc_kspace_real = tf.reshape(disc_kspace_real, [disc_size[0],disc_size[1],disc_size[2],1])
        disc_kspace_imag = tf.cast(tf.imag(disc_kspace), tf.float32)
        #print(disc_kspace_imag)        
        disc_kspace_imag = tf.reshape(disc_kspace_imag, [disc_size[0],disc_size[1],disc_size[2],1])
        disc_kspace_mag = tf.cast(tf.abs(disc_kspace), tf.float32)
        #print(disc_kspace_mag)
        disc_kspace_mag = tf.log(disc_kspace_mag)
        disc_kspace_mag = tf.reshape(disc_kspace_mag/tf.reduce_max(disc_kspace_mag), [disc_size[0],disc_size[1],disc_size[2],1])
        if hybrid_disc == 1:
            disc_hybird = tf.concat(axis = 3, values = [disc_input , disc_kspace_mag])
        else:
            disc_hybird = tf.concat(axis = 3, values = [disc_input * 2-1, disc_kspace_imag, disc_kspace_real, disc_kspace_imag])
    else:
        disc_hybird = disc_input
    print('discriminator input dimensions: {0}'.format(disc_hybird.get_shape()))
    model = Model('DIS', disc_hybird)        

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        if not FLAGS.wgan_gp:  # bn not compatible with wgan, can use layer norm
            model.add_batch_norm()
        if (FLAGS.activation=='lrelu'):
            model.add_lrelu()
        else:
            model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    if not FLAGS.wgan_gp: 
        model.add_batch_norm()
    if (FLAGS.activation=='lrelu'):
        model.add_lrelu()
    else:
        model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    if not FLAGS.wgan_gp: 
        model.add_batch_norm()
    if (FLAGS.activation=='lrelu'):
        model.add_lrelu()
    else:
        model.add_relu()

    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()
    output_layers = [model.outputs[0]] + model.outputs[1:-1][::layer_output_skip] + [model.outputs[-1]]

    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    disc_vars = list(set(new_vars) - set(old_vars))
    

    return model.get_output(), disc_vars, output_layers

def conv(batch_input, out_channels, stride=2, size_kernel=4):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [size_kernel, size_kernel, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def deconv(batch_input, out_channels, size_kernel=3):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [size_kernel, size_kernel, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv        

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized      

def Fourier(x, separate_complex=True):    
    x = tf.cast(x, tf.complex64)
    if separate_complex:
        x_complex = x[:,:,:,0]+1j*x[:,:,:,1]
    else:
        x_complex = x
    x_complex = tf.reshape(x_complex,x_complex.get_shape()[:3])
    y_complex = tf.fft2d(x_complex)
    #print('using Fourier, input dim {0}, output dim {1}'.format(x.get_shape(), y_complex.get_shape()))
    # x = tf.cast(x, tf.complex64)
    # y = tf.fft3d(x)
    # y = y[:,:,:,-1]
    return y_complex


def _generator_model_with_scale(sess, features, masks, MY,s, layer_output_skip=5, num_dc_layers=0):
    # Upside-down all-convolutional resnet
    channels = 2
    #image_size = tf.shape(features)
    mapsize = 3
    res_units  = [64,64,64,64,64] #[128]*5 #[64, 32, 16]#[256, 128, 96]
    scale_changes = [0]*5
    print('use resnet without pooling:', res_units)
    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated

    # See Arxiv 1603.05027
    model = Model('GEN', features)

    # loop different levels
    for ru in range(len(res_units)-1):
        nunits  = res_units[ru]

        for j in range(2): 
            model.add_residual_block(nunits, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        if scale_changes[ru]>0:
            model.add_upscale()

        #-->model.add_batch_norm()
        if (FLAGS.activation_G=='lrelu'):
            model.add_lrelu()
        else:
            model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

    # Finalization a la "all convolutional net"
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    if (FLAGS.activation_G=='lrelu'):
        model.add_lrelu()
    else:
        model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    if (FLAGS.activation_G=='lrelu'):
        model.add_lrelu()
    else:
        model.add_relu()

    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    #model.add_sigmoid()
    
    # add dc connection for each block
    if num_dc_layers >= 0:
        # parameters
        threshold_zero = 1./255.
        mix_DC = 1 #0.95

        #mask_kspace = tf.cast(masks, dtype=tf.float32) #tf.greater(tf.abs(feature_kspace),threshold_zero)  
        mask_kspace = tf.cast(masks, tf.complex64) * mix_DC

        # add dc layers
        # get output and input
        if 0!=1:
            last_layer = tf.cast(model.outputs[-1],tf.complex64)
            last_layer = tf.reshape(last_layer[:,:,:,0]+1j*last_layer[:,:,:,1],[FLAGS.batch_size,1,256,320]) 
        # compute kspace
            gene_kspace = fftshift(tf.fft2d(fftshift(last_layer*s)))    
            last_layer =tf.reshape(last_layer,[FLAGS.batch_size,256,320,1])
        # affine projection
            corrected_kspace =  mask_kspace*MY + gene_kspace * (1.0 - mask_kspace)
        # inverse fft
            corrected_complex = fftshift(tf.ifft2d(fftshift(corrected_kspace)))*tf.conj(s)
            corrected_complex = tf.reduce_sum(corrected_complex,axis=1)

        else:
            feature_kspace = Fourier(features, separate_complex=True)        
            projected_kspace = feature_kspace #* mask_kspace
            last_layer = model.outputs[-1]
            gene_kspace = Fourier(last_layer, separate_complex=True) 
            corrected_kspace =  projected_kspace + 0*gene_kspace * (1.0 - mask_kspace)
            corrected_complex = tf.ifft2d(corrected_kspace)

        image_size = tf.shape(corrected_complex)

        # get abs
        #corrected_mag = tf.cast(tf.abs(corrected_complex), tf.float32)
          
        #get real and imaginary parts
        corrected_real = tf.reshape(tf.real(corrected_complex), [FLAGS.batch_size, image_size[1], image_size[2], 1])
        corrected_imag = tf.reshape(tf.imag(corrected_complex), [FLAGS.batch_size, image_size[1], image_size[2], 1])
           
        #print('size_corrected_real', corrected_real.get_shape())
        corrected_real_concat = tf.concat([corrected_real, corrected_imag], axis=3)
        #print('corrected_concat', corrected_real_concat.get_shape())
        #print('channels', channels)

        model.add_layer(corrected_real_concat)
  
        # model.add_sigmoid()

        #print('variational network with DC correction', model.outputs)

    
    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    gene_vars = list(set(new_vars) - set(old_vars))

    # select subset of layers
    output_layers = last_layer#[model.outputs[0]] + model.outputs[1:-1][::layer_output_skip] + [model.outputs[-1]]

    return model.get_output(), gene_vars, output_layers



def create_model(sess, features, labels, masks, MY, s, architecture='resnet'):
    # sess: TF sesson
    # features: input, for SR/CS it is the input image
    # labels: output, for SR/CS it is the groundtruth image
    # architecture: aec for encode-decoder, resnet for upside down 
    # Generator
    rows      = int(features.get_shape()[1])
    cols      = int(features.get_shape()[2])
    channels  = int(features.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels])
    gene_mMY = tf.placeholder(tf.complex64, shape=[FLAGS.batch_size, 8, rows, cols])
    gene_ms = tf.placeholder(tf.complex64, shape=[FLAGS.batch_size, 8, rows, cols])

    if (FLAGS.sampling_pattern!="nomask"):
        function_generator = lambda x,y,z,w: _generator_model_with_scale(sess,x,y,z,w,
                                                num_dc_layers=0, layer_output_skip=7)
    else:  # with unmasked input, remove dc
        function_generator = lambda x,y,z,w: _generator_model_with_scale(sess,x,y,z,w,
                                                num_dc_layers=-1, layer_output_skip=7)

    rbs=2
    with tf.variable_scope('gene') as scope:
        if FLAGS.unrolled>0:
            gene_output_1 = mri_model.unroll_fista(MY, s, num_grad_steps=FLAGS.unrolled,resblock_num_features=64, resblock_num_blocks=rbs, is_training=True, scope="MRI",mask_output=1, window=None, do_hardproj=True,num_summary_image=0,mask=masks, verbose=False)
            gene_var_list, gene_layers_1 =None,['empty']
        else:
            gene_output_1, gene_var_list, gene_layers_1 = function_generator(features, masks, MY, s)                      
        scope.reuse_variables()

        gene_output_real = gene_output_1
        gene_output = tf.reshape(gene_output_real, [FLAGS.batch_size, rows, cols, 2])
        gene_layers = gene_layers_1
        #tf.summary.image('gene_train_last',abs(gene_layers),2)
        #print('gene_output_train', gene_output.get_shape()) 

        # for testing input
        if FLAGS.unrolled>0:
            gene_moutput_1 = mri_model.unroll_fista(gene_mMY, gene_ms, num_grad_steps=FLAGS.unrolled,resblock_num_features=64, resblock_num_blocks=rbs, is_training=False, scope="MRI",mask_output=1, window=None, do_hardproj=True,num_summary_image=0, mask=masks, verbose=False)
            gene_mlayers_1 =None
        else:
            gene_moutput_1, _ , gene_mlayers_1 = function_generator(gene_minput, masks, gene_mMY,gene_ms)
        scope.reuse_variables()

        
        
        gene_moutput_real = gene_moutput_1
        #gene_moutput_complex = tf.complex(gene_moutput_real[:,:,:,0], gene_moutput_real[:,:,:,1])
        #gene_moutput = tf.abs(gene_moutput_complex)
        #print('gene_moutput_test', gene_moutput.get_shape())i
        gene_moutput = tf.reshape(gene_moutput_real, [FLAGS.batch_size, rows, cols, 2])
        gene_mlayers = gene_mlayers_1


    # Discriminator with real data
    if FLAGS.use_phase==True:
        disc_real_input = tf.identity(labels, name='disc_real_input')
    else:
        disc_real_input = tf.sqrt(labels[:,:,:,0]**2+labels[:,:,:,1]**2)
        disc_real_input =tf.expand_dims(disc_real_input,-1)

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc',reuse=tf.AUTO_REUSE) as scope:
        disc_real_output, disc_var_list, disc_layers_X = \
                _discriminator_model(sess, features, disc_real_input, hybrid_disc=FLAGS.hybrid_disc)

        scope.reuse_variables()
        if FLAGS.use_phase==True:
            gene_output_abs = gene_output
        else:
            gene_output_abs=tf.sqrt(gene_output[:,:,:,0]**2+gene_output[:,:,:,1]**2)
            gene_output_abs =tf.expand_dims(gene_output_abs,-1)

        disc_fake_output, _, disc_layers_Z = _discriminator_model(sess, features, gene_output_abs, hybrid_disc=FLAGS.hybrid_disc)

    return [gene_minput,gene_mMY,gene_ms, gene_moutput, gene_output_abs, gene_var_list, gene_layers, gene_mlayers, disc_real_output, disc_fake_output, disc_var_list,disc_real_input]    


# SSIM
def keras_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    # Returns
        A tensor with the variance of elements of `x`.
    """
    # axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    m = tf.reduce_mean(x, reduction_indices=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          reduction_indices=axis,
                          keepdims=keepdims)


def keras_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(keras_var(x, axis=axis, keepdims=keepdims))


def keras_mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.
    # Returns
        A tensor with the mean of elements of `x`.
    """
    # axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, reduction_indices=axis, keepdims=keepdims)

def loss_DSSIS_tf11(y_true, y_pred, patch_size=5, batch_size=-1):
    # get batch size
    if batch_size<0:
        batch_size = int(y_true.get_shape()[0])
    else:
        y_true = tf.reshape(y_true, [batch_size] + get_shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [batch_size] + get_shape(y_pred)[1:])

    patches_true = tf.extract_image_patches(y_true, [1, patch_size, patch_size, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, patch_size, patch_size, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    #print(patches_true, patches_pred)
    u_true = keras_mean(patches_true, axis=3)
    u_pred = keras_mean(patches_pred, axis=3)
    #print(u_true, u_pred)
    var_true = keras_var(patches_true, axis=3)
    var_pred = keras_var(patches_pred, axis=3)
    std_true = tf.sqrt(var_true)
    std_pred = tf.sqrt(var_pred)
    #print(std_true, std_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    #print(ssim)
    # ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return tf.reduce_mean(ssim, name='ssim') #tf.reduce_mean(((1.0 - ssim) / 2), name='ssim_loss')

def create_generator_loss(disc_output, gene_output, features, labels, masks):#, X,Z):
    # I.e. did we fool the discriminator?
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_output, labels=tf.ones_like(disc_output))

    # LS cost
    ls_loss = tf.square(disc_output - tf.ones_like(disc_output))
    gene_ls_loss = tf.reduce_mean(ls_loss, name='gene_ls_loss')
    #gene_ce_loss = tf.reduce_mean(cross_entropy, name='gene_ce_loss')
    gene_mse_factor  = tf.placeholder(dtype=tf.float32, name='gene_mse_factor')

    # mse loss
    if FLAGS.supervised==2:
        gene_mixmse_loss = tf.reduce_mean(tf.square(gene_output-labels),name='gene_l2_loss')
    else: 
        gene_mixmse_loss = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')

    '''gene_mse_loss = tf.add(FLAGS.gene_l1l2_factor * gene_l1_loss, 
                        (1.0 - FLAGS.gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    #ssim loss
    gene_ssim_loss = loss_DSSIS_tf11(labels, gene_output)     
    gene_mixmse_loss = tf.add(FLAGS.gene_ssim_factor * gene_ssim_loss,(1.0 - FLAGS.gene_ssim_factor) * gene_mse_loss, name='gene_mixmse_loss')'''

    
    if FLAGS.wgan_gp:
        gene_wgan_gp_loss = -tf.reduce_mean(disc_output)   # wgan gene loss
        gene_fool_loss = gene_wgan_gp_loss
    else:
        # generator fool descriminator loss: gan LS or log loss
        gene_fool_loss = (1.0 - FLAGS.gene_log_factor) * gene_ls_loss

    # non-mse loss = fool-loss + data consisntency loss
    gene_non_mse_l2     = gene_fool_loss
    

    #total loss = fool-loss + data consistency loss + mse forward-passing loss
    
    #gene_mse_factor as a parameter
    if FLAGS.supervised==3:
        gene_loss = (1-gene_mse_factor)*gene_non_mse_l2+gene_mse_factor*gene_mixmse_loss
    elif FLAGS.supervised>0:
        gene_loss = gene_mixmse_loss
    else:
        gene_loss     = gene_non_mse_l2
    # log to tensorboard
    tf.summary.scalar('gene_fool_loss', gene_non_mse_l2)
    tf.summary.scalar('gene_L1_loss', gene_mixmse_loss)
    #list of loss (dummy)
    list_gene_lose = None#[gene_mixmse_loss, gene_fool_loss, gene_non_mse_l2, gene_loss]

    return gene_loss, gene_mixmse_loss, list_gene_lose,gene_mse_factor
    

def create_discriminator_loss(disc_real_output, disc_fake_output, real_data = None, fake_data = None):
    # I.e. did we correctly identify the input as real or not?
    # cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
    # disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss') 
    # cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
    # disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
    
    if FLAGS.wgan_gp:
        disc_cost = tf.reduce_mean(disc_fake_output) - tf.reduce_mean(disc_real_output)  
        # generate noisy inputs 
        alpha = tf.random_uniform(shape=[FLAGS.batch_size, 1, 1, 1], minval=0.,maxval=1.) 
        interpolates = real_data + alpha*(fake_data - real_data) 
        
        with tf.variable_scope('disc', reuse=True) as scope:
            interpolates_disc_output, _, _ = _discriminator_model(None,None, interpolates, hybrid_disc=FLAGS.hybrid_disc)
            gradients = tf.gradients(interpolates_disc_output, [interpolates])[0] 
            gradients = tf.layers.flatten(gradients)  # [batch_size, -1] 
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))  
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        disc_loss = tf.add(disc_cost, 10. * gradient_penalty, name='disc_loss')

        tf.summary.scalar('disc_gp_loss',gradient_penalty)        
        tf.summary.scalar('disc_total_loss',disc_loss)
        tf.summary.scalar('disc_loss_wo_gp',disc_cost) 
        return disc_loss,gradient_penalty,disc_cost
  
    else:
        ls_loss_real = tf.square(disc_real_output - tf.ones_like(disc_real_output))
        # ls loss
        disc_real_loss = tf.reduce_mean(ls_loss_real, name='disc_real_loss')
        ls_loss_fake = tf.square(disc_fake_output)
        disc_fake_loss = tf.reduce_mean(ls_loss_fake, name='disc_fake_loss')

        # log to tensorboard
        tf.summary.scalar('disc_real_loss',disc_real_loss)
        tf.summary.scalar('disc_fake_loss',disc_fake_loss) 
        
        return disc_real_loss, disc_fake_loss

def create_optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    if FLAGS.wgan_gp:
        gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.5, beta2=0.9,
                                       name='gene_optimizer')
        disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.5, beta2=0.9,
                                       name='disc_optimizer')
      
    else:
        gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=FLAGS.learning_beta1,
                                           name='gene_optimizer')
        if FLAGS.disc_opti == 'adam':
          disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=FLAGS.learning_beta1,
                                           name='disc_optimizer')
        elif FLAGS.disc_opti == 'sgd':
          disc_opti = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                           name='disc_optimizer_sgd')
    
    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    
    disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)


def fftshift(im,axis=(2,3),name="fftshift"):
    
    with tf.name_scope(name):
        split0, split1 = tf.split(im, 2, axis=axis[0])
        output = tf.concat((split1, split0), axis=axis[0])
        split0, split1 = tf.split(output,2,axis=axis[1])
        output = tf.concat((split1, split0), axis=axis[1])
    return output
    '''axes=range(m.rank())
    shift=[dim // 2 for dim in 
    return tf.manip.roll()'''
