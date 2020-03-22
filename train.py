import os
import time
import glob
import shutil
import numpy as np
import scipy.io as sio
import tensorflow as tf

import tf_eager.util as U

from PIL import Image

from tf_eager.util import copytree, load_prob_map, augmentaion, save_augmentation, copyfiles
from tf_eager.train import Trainer
from tf_eager.image_loder import ImageLoder
from model.unet_2d import Model

import platform

# enable eager execution
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
tf.enable_eager_execution(config=config)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=128, help='image size')
parser.add_argument('--sub_num', type=int, default=16, help='number of sub-models of M1')
parser.add_argument('--num_unlabeled', type=int, default=1944, help='number of unlabeled data (max = 1944)')
parser.add_argument('--scaled_weights', type=bool, default=True, help='scale the weight to 0.1 - 1 or not')
args = parser.parse_args()

img_size = str(args.img_size)
init_num_sub = args.sub_num
num_unlabeled = args.num_unlabeled
start_stage = 0

M0_ckpt_path = 'result_supervised_ISIC_' + img_size + '/checkpoint'
root_result_path = 'sub_models_ISIC_' + img_size + '_init_' + str(init_num_sub) + '_' + str(num_unlabeled)
data_path = 'data'
all_data = data_path + '/2D_ISIC_2018_' + img_size

unlabeled_set = root_result_path + '/_data/unlabeled'


div_saved_mat = sio.loadmat(data_path + '/ISBI_skin_100_500_50_' + img_size)
train_list = list(div_saved_mat['train_set'])
test_list = list(div_saved_mat['test_set'])
validation_list = list(div_saved_mat['validation_set'])
all_unlabeled_list = list(div_saved_mat['unlabeled_set'][:num_unlabeled])


if os.path.exists(root_result_path):
    shutil.rmtree(root_result_path, ignore_errors=True)
os.makedirs(root_result_path)
if os.path.exists(unlabeled_set):
    shutil.rmtree(unlabeled_set, ignore_errors=True)
os.makedirs(unlabeled_set)

copyfiles(all_unlabeled_list, unlabeled_set)
pseudo_lab_list = []
for ulab_path in all_unlabeled_list:
    pseudo_path = ulab_path.replace('2D_ISIC_2018_' + img_size, '2D_ISIC_2018_' + img_size + '_pseudo')
    pseodo_path = pseudo_path.replace('.jpg', 'prob.mat')
    pseudo_lab_list.append(pseodo_path)
copyfiles(pseudo_lab_list, unlabeled_set)

unlabeled_list = glob.glob(unlabeled_set+'/*.jpg')

total_data = len(unlabeled_list) 
end_stage = int(np.ceil(np.log2(init_num_sub)) + 2)
end_stage = max(end_stage, 4)
# end_stage = 6

time_cost = []
for stage in range(start_stage, end_stage):
    start_time = time.time()
    #=========================setting====================
    # min_c = min(stage, 3)
    # resize = (64*2**min_c, 64*2**min_c)
    match_ref = None

    # n_layer = min_c + 2
    # features_root = 128 // 2**min_c

    features_root = 16
    resize = None # (128, 128)
    n_layer = 5

    noise = False

    lr = 0.0001
    batch_size = 1
    restore = False
    save_best_ckpt = True
    keep_prob = 1.
    weight_type = None
    eval_iter = 1
    res = True

    #early stopping
    epochs = 50
    min_epochs = 10

    min_delta = 0.0005
    max_patience = 5


    #====================================================

    if stage == 1:
        num_sub = init_num_sub
    else:
        num_sub = np.max([int(init_num_sub // 2**(stage-1)), 1])  

    temp_unlab_list = unlabeled_list.copy()

    for curr_sub in range(num_sub):
        
    #====================================================
        stage_result_path = root_result_path + '/M%d'%stage

        sub_size = int(round(total_data / num_sub))
        np.random.shuffle(temp_unlab_list)
        pseudo_list = temp_unlab_list[0:sub_size]
        result_path = stage_result_path + '/sub_%d'%curr_sub

        prob_threshold = 0.5
        dataset_train = ImageLoder(filename_list=pseudo_list + train_list,
                                    data_suffix='.jpg',
                                    label_suffix='prob.mat',
                                    n_class=2,
                                    shuffle_data=True,
                                    resize=resize,
                                    match_ref=match_ref,
                                    prob_threshold=prob_threshold,
                                    noise=noise)

        # load validation data
        dataset_validation= ImageLoder(filename_list=validation_list,
                                data_suffix='.jpg',
                                label_suffix='_segmentation.png',
                                n_class=2,
                                shuffle_data=False,
                                resize=resize,
                                match_ref=match_ref)

        data_provider = {'train':dataset_train, 'validation':dataset_validation}

        # build model
        unet2d = Model(n_class=2, n_layer=n_layer, 
                        features_root=features_root, filter_size=3, pool_size=2, 
                        keep_prob=keep_prob,
                        weight_type=weight_type,
                        res=res,
                        concat_or_add='concat')
        

        train_size = dataset_train.size()
        assert train_size % batch_size == 0
        iters = train_size // batch_size

        print('dataset size: %d, batch size: %d, iters: %d'%(train_size, batch_size, iters))

        # init trainer
        trainer = Trainer(unet2d, learning_rate=lr, training_iters=iters, batch_size=batch_size)
        
        # load weights
        if stage > 0:
            if stage == 1:
                last_ckpt_path = M0_ckpt_path + '/best_ckpt'
            else:
                last_weights_path = root_result_path + '/M%d/sub_%d/weights'%(stage-1, curr_sub)
                last_ckpt_path = root_result_path + '/M%d/sub_%d/checkpoint/best_ckpt'%(stage-1, sub_rank[curr_sub])
                
            # trainer.load_weights(n_layer-2, last_weights_path, dataset_validation)
            trainer.restore(last_ckpt_path)

        trainer.train(data_provider, 
                        epochs=epochs, 
                        restore=restore, 
                        output_path=result_path, 
                        train_summary=False, 
                        validation_summary=False,
                        eval_iter=eval_iter,
                        save_best_ckpt=save_best_ckpt,
                        min_delta=min_delta,
                        max_patience=max_patience,
                        min_epochs=min_epochs)

        # save weights                
        ckpt_path = result_path+'/checkpoint/best_ckpt'
        weights_path = result_path + '/weights'
        if os.path.exists(weights_path):
            shutil.rmtree(weights_path, ignore_errors=True)
        os.makedirs(weights_path)
        trainer.save_weights(n_layer-1, ckpt_path, weights_path)

        # stage test
        dataset_unlabeled = ImageLoder(filename_list=unlabeled_list,
                                    data_suffix='.jpg',
                                    n_class=2,
                                    shuffle_data=False,
                                    resize=resize,
                                    match_ref=match_ref)

        dataset_test = ImageLoder(filename_list=test_list,
                                data_suffix='.jpg',
                                label_suffix='_segmentation.png',
                                n_class=2,
                                shuffle_data=False,
                                resize=resize,
                                match_ref=match_ref)
            
        prob = trainer.pseudo_generation(dataset_unlabeled, ckpt_path)
        eval_results = trainer.results_eval(dataset_test, ckpt_path)
        saved_mat = {'id':unlabeled_list, 'prob': prob}
        saved_mat.update(eval_results) 
        sio.savemat(result_path + '/prob', saved_mat)

    # combine pseudo labels
    prob = []
    mid_results = None
    for i in range(num_sub):
        print('sub: %d'%i, end='\t')
        if stage > 0:
            result_path = stage_result_path + '/sub_%d'%i
        sub_mat = sio.loadmat(result_path+'/prob.mat')
        prob.append(sub_mat.pop('prob')) 
        sub_mat.pop('id')
        sub_mat.pop('__globals__')
        sub_mat.pop('__header__')
        sub_mat.pop('__version__')

        # eval (labeled test data)
        if mid_results is None:
            mid_results = sub_mat
        else:
            mid_results = U.add_dict(mid_results, sub_mat)
        print('dice: %s'%np.mean(sub_mat['dice'])) # eval (labeled)
    mid_results = U.div_dict(mid_results, num_sub)

    prob_labeled = np.average(prob, 0) 
    prob_labeled = np.eye(prob_labeled.shape[-1])[np.argmax(prob_labeled, -1)]
    conf_weights = []
    for i in range(num_sub):
        sub_weight = np.sum(prob[i] * prob_labeled)
        conf_weights.append(sub_weight)

    if args.scaled_weights and num_sub > 1:
        scaled_max = 1.0
        scaled_min = 0.1
        conf_weights = (scaled_max - scaled_min) * ((conf_weights - np.min(conf_weights)) / (np.max(conf_weights) - np.min(conf_weights))) + scaled_min
        
    conf_weights = conf_weights / np.sum(conf_weights)


    prob = np.average(prob, 0, conf_weights)

    sub_rank = np.argsort(conf_weights)[::-1]

    # save
    sio.savemat(root_result_path+'/prob_'+str(stage), {'prob': prob})
    print('save')
    for i, f in enumerate(unlabeled_list):
        lab_name = f.replace('.jpg', 'lab.mat')
        sio.savemat(lab_name, {'prob':np.argmax(prob[i], -1)})
        # sio.savemat(lab_name, {'prob':prob[i]})
        prob_name = f.replace('.jpg', 'prob.mat')
        sio.savemat(prob_name, {'prob':prob[i]})  

    curr_time_cost = time.time() - start_time
    time_cost.append(curr_time_cost)

    with open(root_result_path+'/test_result.txt', 'a+') as f:
        f.write('M%d: '%stage)
        for key in mid_results:
            values = mid_results.get(key)
            values = np.mean(values)
            f.write('%s: %s  '%(key, str(values)))
        f.write('time cost: %.4f  '%curr_time_cost)
        f.write('\tthreshold: %f  '%prob_threshold)
        f.write('\tsub weight: ')
        for sw in conf_weights:
            f.write('%.4f '%sw)
        f.write('\n')

# time calculation
with open(root_result_path+'/test_result.txt', 'a+') as f:
    f.write('total time cost: %.4f  '%np.sum(time_cost))
    f.write('\n')

# test
ckpt_path = result_path+'/checkpoint/best_ckpt'
eval_results = trainer.results_eval(dataset_test, ckpt_path)

sio.savemat(root_result_path+'/final_dice', eval_results)

# save org label pred
vis_path = root_result_path+'/vis_results'
if os.path.exists(vis_path):
    shutil.rmtree(vis_path, ignore_errors=True)
os.makedirs(vis_path)
for i in range(dataset_test.size()):
    s_org, s_lab, _ = dataset_test(1)
    s_prob = trainer.model.predict(s_org)[0]
    s_pred = np.argmax(s_prob, -1)
    Image.fromarray(s_org[0, ..., 0]).save(vis_path+'/%d_org.tif'%i)
    Image.fromarray(np.array(np.argmax(s_lab[0], -1), np.uint8)).save(vis_path+'/%d_lab.tif'%i)
    Image.fromarray(np.array(s_pred, np.uint8)).save(vis_path+'/%d_pred.tif'%i)
    sio.savemat(vis_path+'/%d_lab.mat'%i, {'prob': s_prob})
