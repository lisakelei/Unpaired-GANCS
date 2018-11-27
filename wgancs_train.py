import numpy as np
from PIL import Image
import os.path
import scipy.misc
import tensorflow as tf
import time
import json
from scipy.io import savemat
import wgancs_model

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, 
                        batch, suffix, max_samples=8, gene_param=None):
    
    td = train_data

    size = [label.shape[1], label.shape[2]]

    # complex input zpad into r and channel
    complex_zpad = feature 

    # zpad magnitude
    if FLAGS.use_phase==True:
      mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2+complex_zpad[:,:,:,1]**2)
    else:
      mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2)
    
    # output image
    if FLAGS.use_phase==True:
      gene_output_complex = tf.complex(gene_output[:,:,:,0],gene_output[:,:,:,1])
    else:
      gene_output_complex = gene_output
    mag_output=tf.abs(gene_output_complex)#print('size_mag_output', mag)

    if FLAGS.use_phase==True:
      label_complex = tf.complex(label[:,:,:,0], label[:,:,:,1])
    else:
      label_complex = label
    mag_gt = tf.abs(label_complex)
    
    # calculate SSIM SNR and MSE for test images
    signal=mag_gt[:,20:size[0]-20,14:size[1]-14]    # crop out edges
    Gout=mag_output[:,20:size[0]-20,14:size[1]-14]
    SSIM=tf.convert_to_tensor(0)#wgancs_model.loss_DSSIS_tf11(signal, Gout)
    signal=tf.reshape(signal,(FLAGS.batch_size,-1))   # and flatten
    Gout=tf.reshape(Gout,(FLAGS.batch_size,-1))    
    s_G=tf.abs(signal-Gout)
    SNR_output = 10*tf.reduce_sum(tf.log(tf.reduce_sum(signal**2,axis=1)/tf.reduce_sum(s_G**2,axis=1)))/tf.log(10.0)/FLAGS.batch_size
    MSE=tf.reduce_mean(s_G)   
    
    # concate for visualize image
    if FLAGS.use_phase==True:
      image = tf.concat(axis=2, values=[mag_zpad, mag_output, mag_gt,50*abs(mag_output-mag_zpad),70*abs(mag_gt-mag_output)])
    else:
      image = tf.concat(axis=2, values=[mag_zpad, mag_output, mag_gt,abs(mag_gt-mag_zpad)])
    image = image[0:max_samples,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:] for i in range(int(max_samples))])
    image,snr,mse,ssim,igt = td.sess.run([image,SNR_output,MSE,SSIM,mag_gt])
    # save to image file
    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    try:
      scipy.misc.toimage(image,cmax=np.amax(igt),cmin=0).save(filename)
    except:
      import pilutil
      pilutil.toimage(image,cmax=np.amax(igt),cmin=0).save(filename)
    print("    Saved %s" % (filename,))

    #gene_output_abs = np.abs(gene_output)
    # save layers and var_list
    if gene_param is not None:
        #add feature 
        print('dimension for input, ref, output:',
              feature.shape, label.shape, gene_output.shape)
        gene_param['feature'] = feature.tolist()
        gene_param['label'] = label.tolist()
        gene_param['gene_output'] = gene_output.tolist()
        # add input arguments
        # print(FLAGS.__dict__['__flags'])
        # gene_param['FLAGS'] = FLAGS.__dict__['__flags']

        # save json
        '''
        filename = 'batch%06d_%s.json' % (batch, suffix)
        filename = os.path.join(FLAGS.train_dir, filename)
        with open(filename, 'w') as outfile:
            json.dump(gene_param, outfile)
        print("    Saved %s" % (filename,))
        '''
    return snr,mse,ssim

def _save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver(sharded=True)
    filename=saver.save(td.sess, newname)

    print("Checkpoint saved:",filename)

def train_model(train_data, batchcount, num_sample_train=16, num_sample_test=116):
    td = train_data
    #summary_op = td.summary_op

    #td.sess.run(tf.global_variables_initializer())

    #TODO: load data

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = batchcount
    # batch info    
    batch_size = FLAGS.batch_size
    num_batch_train = num_sample_train / batch_size
    num_batch_test = num_sample_test / batch_size            

    # Cache test features and labels (they are small)    
    # update: get all test features
    list_test_features = []
    list_test_labels = []
    list_test_s=[]
    list_test_MY=[]
    for batch_test in range(int(num_batch_test)):
        test_feature, test_label, test_s,test_MY = td.sess.run([td.test_features, td.test_labels,td.test_s,td.test_MY])
        list_test_features.append(test_feature)
        list_test_labels.append(test_label)
        list_test_s.append(test_s)
        list_test_MY.append(test_MY)
    print('prepare {0} test feature batches'.format(num_batch_test))
    # print([type(x) for x in list_test_features])
    # print([type(x) for x in list_test_labels])
    accumuated_err_loss=[]
    sum_writer=tf.summary.FileWriter(FLAGS.train_dir, td.sess.graph)
    summary_op=tf.summary.merge_all()
    snr_prev=0
    while not done:
        batch += 1
        gene_ls_loss = gene_dc_loss = gene_loss = disc_real_loss = disc_fake_loss = -1.234

        #first train based on MSE and then GAN
        feed_dict = {td.learning_rate : lrval }

	# train disc multiple times
        for disc_iter in range(0):
            td.sess.run([td.disc_minimize],feed_dict=feed_dict)
	# then train both disc and gene once
        ops = [td.gene_minimize, td.disc_minimize, summary_op, td.gene_loss, td.gene_mse_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss]                   
        _, _, fet_sum,gene_loss, gene_mse_loss, gene_dc_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        sum_writer.add_summary(fet_sum,batch)
        
        # verbose training progress
        if batch % 20 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            err_log = 'Elapsed[{0:3f}], Batch [{1:1f}], G_Loss[{2}], G_mse_Loss[{3:3.3f}], G_LS_Loss[{4:3.3f}], G_DC_Loss[{5:3.3f}], D_Real_Loss[{6:3.3f}], D_Fake_Loss[{7:3.3f}]'.format(elapsed, batch, gene_loss, gene_mse_loss, gene_ls_loss, gene_dc_loss, disc_real_loss, disc_fake_loss)
            print(err_log)
            # update err loss
            err_loss = [int(batch), float(gene_loss), float(gene_dc_loss), 
                        float(gene_ls_loss), float(disc_real_loss), float(disc_fake_loss)]
            accumuated_err_loss.append(err_loss)
            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if (current_progress >= 1.0) or (batch > FLAGS.train_time*200):
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        # export test batches
        if batch % FLAGS.summary_period == 0:
            # loop different test batch
            snr=mse=ssim=0
            for index_batch_test in range(int(num_batch_test)):
                # get test feature
                test_feature = list_test_features[index_batch_test]
                test_label = list_test_labels[index_batch_test]
                test_s=list_test_s[index_batch_test]
                test_MY=list_test_MY[index_batch_test]

                # Show progress with test features
                feed_dict = {td.gene_minput: test_feature, td.gene_ms:test_s, td.gene_mMY:test_MY}
                # not export var
                # ops = [td.gene_moutput, td.gene_mlayers, td.gene_var_list, td.disc_var_list, td.disc_layers]
                # gene_output, gene_layers, gene_var_list, disc_var_list, disc_layers= td.sess.run(ops, feed_dict=feed_dict)       
                
                ops = [td.gene_moutput, td.gene_mlayers]
                
                # get timing
                forward_passing_time = time.time()
                gene_output, gene_layers= td.sess.run(ops, feed_dict=feed_dict)       
                inference_time = time.time() - forward_passing_time
		
                # print('gene_var_list',[x.shape for x in gene_var_list])
                #print('gene_layers',[x.shape for x in gene_layers])
                #print("test time data consistency:", gene_dc_loss): add td.gene_dc_loss in ops
                # print('disc_var_list',[x.shape for x in disc_var_list])
                #print('disc_layers',[x.shape for x in disc_layers])

                # save record
                gene_param = {'train_log':err_log,
                              'train_loss':accumuated_err_loss,
                              'inference_time':inference_time}                
                # gene layers are too large
                if index_batch_test>0:
                    gene_param['gene_layers']=[]
                snr_b,mse_b,ssim_b=_summarize_progress(td, test_feature, test_label, gene_output, batch, 
                                    'test%03d'%(index_batch_test),                                     
                                    max_samples = batch_size,
                                    gene_param = gene_param)
                snr+=snr_b
                mse+=mse_b
                ssim+=ssim_b
                tbimage=tf.summary.image('testout',tf.abs(gene_layers),2)
                sum_writer.add_summary(td.sess.run(tbimage))
                # try to reduce mem
                gene_output = None
                gene_layers = None
                #disc_layers = None
                accumuated_err_loss = []
                Snr=snr/num_batch_test
            write_summary(Snr,'SNR',sum_writer,batch) 
            print('SNR: ',Snr,'MSE: ',mse/num_batch_test,'SSIM: ',ssim/num_batch_test)
        # export train batches
        if FLAGS.summary_train_period>0 and (batch % FLAGS.summary_train_period == 0):
            # get train data
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss, 
                   td.train_features, td.train_labels, td.gene_output]#, td.gene_var_list, td.gene_layers]
            _, _, gene_loss, gene_dc_loss, disc_real_loss, disc_fake_loss, train_feature, train_label, train_output = td.sess.run(ops, feed_dict=feed_dict)
            print('train sample size:',train_feature.shape, train_label.shape, train_output.shape)
            _summarize_progress(td, train_feature, train_label, train_output, batch%num_batch_train, 'train',max_samples=4)

        
        # export check points
        if batch % FLAGS.checkpoint_period == 0 and Snr>snr_prev:
            # Save checkpoint
            _save_checkpoint(td, batch)
            snr_prev=Snr

    _save_checkpoint(td, batch)
    print('Finished training!')

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
