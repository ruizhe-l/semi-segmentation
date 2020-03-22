from __future__ import absolute_import, division, print_function

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

import tf_eager.util as U

# from tf_eager._weight import balance_weight_map, feedback_weight_map, nan_to_zero

class Trainer:
    def __init__(self, model, learning_rate, training_iters, batch_size):
        self.model = model
        self.learning_rate = tfe.Variable(learning_rate)
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.checkpoint = None
        self.lr_step = 0
        def get_lr():
            if self.lr_step > 0 and self.lr_step % training_iters == 0:
                self.learning_rate.assign_sub(self.learning_rate * 0.005)
            self.lr_step += 1
            return self.learning_rate

        self.optimizer = tf.train.AdamOptimizer(get_lr)

    def train(self, data_provider, epochs, output_path, restore=False, train_summary=True, validation_summary=True, base_ckpt=None, eval_iter=5, save_best_ckpt=False, min_delta = None, max_patience = 3, min_epochs = 4):

        # make dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        summary_path = output_path+'/summary'
        prediction_path = output_path+'/prediction'
        checkpoint_path = output_path+'/checkpoint'

        if not restore:
            shutil.rmtree(summary_path, ignore_errors=True)
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            shutil.rmtree(prediction_path, ignore_errors=True)

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            if base_ckpt is not None:
                U.copytree(base_ckpt, checkpoint_path)

        # init global_step
        global_step = tf.train.get_or_create_global_step()
        global_step.assign(0)
        # restore
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        if restore:
            self.checkpoint.restore(checkpoint_path+'/best_ckpt')  
            # global_step.assign(0)

        # summary writer
        train_writer = tf.contrib.summary.create_file_writer(summary_path+'/train')
        validation_writer = tf.contrib.summary.create_file_writer(summary_path+'/validation')

        dataset_train = data_provider.get('train')
        dataset_validation = data_provider.get('validation')

        summary, imgs = self._store_prediction(dataset_validation, '%s/_init'%prediction_path)

        train_dice = []
        validation_dice = []

        best_loss = float('inf')
        best_epoch = 0
        patience = 0
        while global_step < epochs:
            train_evaluation = None
            for i in range(self.training_iters):
                batch_x, batch_y, batch_mask = dataset_train(self.batch_size)
                feed_dict = {'x':batch_x, 'y':batch_y, 'mask':batch_mask}
                grads = self.model.get_grads(feed_dict)
                self.optimizer.apply_gradients(zip(grads, self.model.net.variables))
                # if i % 5 == 0:
                #     print('  --iter %d'%i)
                sub_eval = self.model.evaluation(feed_dict)
                sub_eval.pop('prediction')
                train_evaluation = U.add_dict(train_evaluation, sub_eval)

                # print('step: %d'%i)
            print('epoch %d -- '%global_step, end='')
            
            train_evaluation = U.div_dict(train_evaluation, self.training_iters)
            train_evaluation['learning rate'] = self.learning_rate.numpy()
            output_str = U.eval_to_str(train_evaluation)
            print('train %s'%output_str)
            if train_summary:
                self.write_summary(train_evaluation, train_writer)

            train_dice.append(train_evaluation['dice'])

            if dataset_validation is not None and (global_step.numpy()+1) % eval_iter == 0:
                summary, imgs = self._store_prediction(dataset_validation, '%s/epoch_%d'%(prediction_path, global_step))
                validation_dice.append(summary['dice'])

                if validation_summary:
                    self.write_summary(summary, validation_writer)
                    self.write_image_summary(imgs, validation_writer)
            
                # early stopping
                curr_loss = summary['loss']
                if min_delta is not None:
                    if best_loss - curr_loss > min_delta:
                        patience = 0
                    else:
                        patience += 1

                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_epoch = global_step.numpy()
                    # save best result
                    if save_best_ckpt:
                        self.checkpoint.write(checkpoint_path+'/best_ckpt')
                        print('best ckpt saved')
                
                if global_step >= min_epochs - 1 and patience > max_patience:
                    break


            if not save_best_ckpt:
                self.checkpoint.write(checkpoint_path+'/best_ckpt')
                print('ckpt saved')
            # save model
            # if global_step.numpy() % 10 == 0:
            #     self.checkpoint.save(checkpoint_path+'/ckpt')

            global_step.assign_add(1)

        # finish
        # self.checkpoint.write(checkpoint_path+'/final_ckpt')
        # final_dice = validation_dice[-1]

        return best_epoch

    def pseudo_generation(self, dataset, checkpoint_path, weight=1.0, print_i=False):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)
        d_size = dataset.size()
        probs = None
        for i in range(d_size):
            if print_i and (i + 1) % 10 == 0:
                print(i)
            x, _, _ = dataset(1)
            sub_prob = self.model.predict(x)
            if probs is None:
                probs = sub_prob
            else:
                probs = np.concatenate((probs, sub_prob), axis=0)
        probs = probs * weight

        return probs


    def ckpt_restore(self, checkpoint_path):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)

    def pseudo_one_generation(self, data):
        prob = self.model.predict(data)

        return prob

    def results_eval(self, dataset_validation, checkpoint_path):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)
        v_size = dataset_validation.size()


        eval_results = {}
        eval_results.update({'dice': []})
        eval_results.update({'iou': []})
        eval_results.update({'acc': []})
        eval_results.update({'sensitivity': []})
        eval_results.update({'specificity': []})



        for i in range(v_size):
            sub_x, sub_y, sub_mask = dataset_validation(1)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            results = self.model.evaluation(feed_dict)
            eval_results['dice'].append(results['dice'])
            eval_results['iou'].append(results['iou'])
            eval_results['acc'].append(results['acc'])
            eval_results['sensitivity'].append(results['sensitivity'])
            eval_results['specificity'].append(results['specificity'])

        return eval_results


    def dice_eval(self, dataset_validation, checkpoint_path):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)
        v_size = dataset_validation.size()

        dice = []

        for i in range(v_size):
            sub_x, sub_y, sub_mask = dataset_validation(1)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            sub_dice = self.model.evaluation_dice(feed_dict)[1:]
            dice.append(sub_dice)
            
        return np.array(dice)

    def evaluate(self, dataset_test, batch_size, ckpt_path=None):
        if ckpt_path is not None:
            self.checkpoint.restore(ckpt_path)
        bk_batch_size = self.batch_size
        self.batch_size = batch_size
        evaluation, _ = self._store_prediction(dataset_test, None)
        self.batch_size = bk_batch_size
        return evaluation

    def save_weights(self, s_layer, ckpt_path, weights_path):
        if ckpt_path is not None:
            self.checkpoint.restore(ckpt_path)
        m = self.model.net
        for i in range(s_layer):
            m.get_layer('dw_%d'%(i)).save_weights('%s/dw_%d.h5'%(weights_path, (i)))
            if i > 0:
                m.get_layer('up_%d'%i).save_weights('%s/up_%d.h5'%(weights_path, (i)))

    def load_weights(self, l_layer, weights_path, dataset):
        self._store_prediction(dataset, None)
        m = self.model.net
        for i in range(l_layer):
            m.get_layer('dw_%d'%(i)).load_weights('%s/dw_%d.h5'%(weights_path, (i)))
            if i > 0:
                m.get_layer('up_%d'%i).load_weights('%s/up_%d.h5'%(weights_path, (i)))
        self._store_prediction(dataset, None)
    
    def restore(self, ckpt_path):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        if ckpt_path is not None:
            self.checkpoint.restore(ckpt_path)
        else:
            assert False, 'no ckpt!'

    def _store_prediction(self, dataset_validation, path):
        ndim = dataset_validation.ndim()
        if ndim == 4:
            return self._store_prediction_2d(dataset_validation, path)
        elif ndim == 5:
            return self._store_prediction_3d(dataset_validation, path)
        else:
            raise 'Unknow dimensions of prediction!'
        

    def _store_prediction_2d(self, dataset_validation, path):
        v_size = dataset_validation.size()

        evaluation = None
        imgs = None
        num_imgs = 0

        q = v_size // self.batch_size
        r = v_size % self.batch_size

        while q != 0 or r != 0:
            if q != 0:
                sub_size = self.batch_size
                q -= 1
            elif r != 0:
                sub_size = r
                r = 0    
            sub_x, sub_y, sub_mask = dataset_validation(sub_size)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            sub_eval = self.model.evaluation(feed_dict)
            sub_pred = sub_eval.pop('prediction')
            evaluation = U.add_dict(evaluation, U.multiply_dict(sub_eval, float(sub_size) / v_size))
            if path is not None:
                if imgs is None:
                    imgs = U.combine_images(sub_x, sub_y, sub_pred)
                    num_imgs += 1
                elif num_imgs < 10 / self.batch_size:
                    num_imgs += 1
                    imgs = np.concatenate((imgs, U.combine_images(sub_x, sub_y, sub_pred)), axis = 1)

        if imgs is not None:
            U.save_images(imgs, path)
            
        # print
        output_str = U.eval_to_str(evaluation)
        print('validation %s'%output_str)
        return evaluation, imgs

    def _store_prediction_3d(self, dataset_validation, path):
        evaluation_o = None
        imgs_o = []
        for i in range(dataset_validation.size()):
            v_x, v_y, v_mask = dataset_validation(1)
            feed_dict = {'x':v_x, 'y':v_y, 'mask':v_mask}
            evaluation = self.model.evaluation(feed_dict)
            pred = evaluation.pop('prediction')
            evaluation_o = U.add_dict(evaluation_o, evaluation)

            # save
            imgs = U.combine_images(v_x, v_y, pred, None)
            if len(imgs_o) < 3:
                U.save_images(imgs, '%s_batch_%d'%(path, i))
                imgs_o.append(imgs)

        evaluation_o = U.div_dict(evaluation_o, dataset_validation.size())
        
        # print
        output_str = U.eval_to_str(evaluation_o)
        print('validation %s'%output_str)

        return evaluation_o, imgs_o

    def write_summary(self, summary, writer):
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            for key in summary:
                value = summary.get(key)
                if value.size <= 1:
                    tf.contrib.summary.scalar(key, value)
                else:
                    for i, v in enumerate(value):
                        tf.contrib.summary.scalar('%s/class_%s'%(key, i), v)

    def write_image_summary(self, imgs, writer):
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            if type(imgs[0]) == list:
                for i, img in enumerate(imgs):
                    self.write_one_image_summary(img, 'output/sub_%s'%i)
            else:
                self.write_one_image_summary(imgs, 'output')

    def write_one_image_summary(self, imgs, path):
        for i in range(len(imgs)):
                img = imgs[i]
                img = img.reshape((1, img.shape[0], img.shape[1], 3))
                if i < len(imgs) - 1:
                    tf.contrib.summary.image('%s/class_%d'%(path, i), img)
                else:
                    tf.contrib.summary.image('%s/argmax'%path, img)
