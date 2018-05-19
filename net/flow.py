import os
import sys
import csv
import time
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import shelve

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    step_plot = np.array([])
    lossmva_plot = np.array([], dtype = float)
    loss_plot = np.array([], dtype = float)

    step_no = 0
    s_file = shelve.open('steps_new')
    l_file = shelve.open('loss_new')
    s_file_ = shelve.open('steps_new2')
    l_file_ = shelve.open('loss_new2')
    step_plot = s_file['steps']
    loss_plot = l_file['loss']
    lossmva_plot = l_file['lossmva']
    step_no = step_plot[step_plot.shape[0] - 1]
    loss_mva = lossmva_plot[lossmva_plot.shape[0] - 1]

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1
        
        step_no += 1
        step_plot = np.append(step_plot, step_no)
        loss_plot = np.append(loss_plot, loss)
        lossmva_plot = np.append(lossmva_plot, loss_mva)
        
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('loss vs time')
        plt.plot(step_plot,loss_plot,'g')
        plt.savefig('lossplot6.png')
        
        s_file_['steps'] = step_plot
        l_file_['loss']  = loss_plot
        l_file_['lossmva'] = lossmva_plot



        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)
    #s_file.close()
    #l_file.close()
    s_file_.close()
    l_file_.close()


def predict(self):
    inp_path = self.FLAGS.test
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            print (os.path.join(inp_path, all_inp[i]))
            self.framework.postprocess(prediction,
                os.path.join(inp_path, all_inp[i]))
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))


def predict_params(self):
    boundingboxes = list()
    pick = self.meta['labels']

    images = list()
    ground_truth = dict()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_positives_cl = np.zeros((5,1),dtype = int)
    false_positives_cl = np.zeros((5,1),dtype = int)
    false_negatives_cl = np.zeros((5,1),dtype = int)
    precision_cl = np.zeros((5,1),dtype = float)
    recall_cl = np.zeros((5,1),dtype = float)
    sum_iou = 0
    total_boxes = 0
    sum_iou_cl = np.zeros((5,1),dtype = float)
    total_boxes_cl = np.zeros((5,1), dtype = int)
    C = self.meta['classes']

    results = shelve.open('results')  # for saving results


    csv_fname = os.path.join('/Users/Kush/Desktop/PyTorch-Python/udacity/darkflow/udacity_test.csv')
    boxes_in_ground_truth = 0

    # parsing the .csv file to obtain ground truth data for images, bounding boxes and their respective classes

    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )

        for row in spamreader:
            img_name = row[0]
            w = 1920
            h = 1200
            boxes_in_img = list()
            labels = row[1:]

            for i in range(0, len(labels), 5):
                boundbox = list()
                xmin = int(labels[i])
                ymin = int(labels[i + 1])
                xmax = int(labels[i + 2])
                ymax = int(labels[i + 3])
                class_idx = int(labels[i + 4])
                class_name = pick[class_idx]
                boundbox = [class_name, class_idx, xmin, ymin, xmax, ymax]
                boxes_in_ground_truth += 1
                boxes_in_img.append(boundbox)
            
            images.append(img_name)
            boundingboxes.append(boxes_in_img)
        ground_truth['imgs'] = images                           # ground_truth variable stores the parsed values
        ground_truth['bounding_boxes'] = boundingboxes


    

    inp_path = self.FLAGS.test
    batch = min(self.FLAGS.batch, len(ground_truth['imgs']))
    for j in range(len(ground_truth['imgs']) // batch):
        inp_feed = list(); new_all = list()
        all_inp = ground_truth['imgs'][j*batch: (j*batch+batch)]
        ground_truth_batch = ground_truth['bounding_boxes'][j*batch: (j*batch + batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        tp_batch = 0
        fp_batch = 0
        fn_batch = 0
        for i, prediction in enumerate(out):
            # results obtained from the postprocess function in test.py in yolovs directory
            # postprocess funtion has been modified to take additional input as ground truth bounding box info
            tp, fp, fn, tp_cl, fp_cl, totalp_cl, fn_cl, iou, iou_cl, tboxes, tboxes_cl = self.framework.postprocess(prediction, os.path.join(inp_path, all_inp[i]), ground_truth_batch[i])
            #storing the results
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            true_positives_cl += tp_cl
            false_positives_cl += fp_cl
            false_negatives_cl += fn_cl
            tp_batch += tp
            fp_batch += fp
            fn_batch += fn
            sum_iou += iou
            sum_iou_cl += iou_cl
            total_boxes += tboxes
            total_boxes_cl += tboxes_cl

        

        stop = time.time(); last = stop - start

        print ('precision for batch: ', float(tp_batch/(tp_batch + fp_batch)))
        print ('recall for batch: ', float(tp_batch/(tp_batch + fn_batch)))


  
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

    # computing precision, recall and iou 

    avg_iou = sum_iou/total_boxes
    avg_iou_cl = sum_iou_cl/total_boxes_cl
    precision = float(true_positives/(true_positives + false_positives))
    recall = float(true_positives/(true_positives + false_negatives))
    precision_cl = true_positives_cl/(true_positives_cl + false_positives_cl)
    recall_cl = true_positives_cl/(true_positives_cl + false_negatives_cl)
    

    # saving results
    
    results['tp'] = true_positives
    results['tp_cl'] = true_positives_cl
    results['fp'] = false_positives
    results['fp_cl'] =false_positives_cl
    results['fn'] = false_negatives
    results['fn_cl'] = false_negatives_cl
    results['pr'] = precision
    results['rc'] = recall
    results['pr_cl'] = precision_cl
    results['rc_cl'] = recall_cl
    results.close()

    # printing results
    
    print ('avg iou: ', avg_iou)
    print ('avg iou for classes: ', avg_iou_cl)
    print ('total boxes: ', total_boxes)
    print ('total boxes for classes: ', total_boxes_cl)
    print ('precision: ', float(true_positives/(true_positives + false_positives)))
    print ('recall: ', float(true_positives/(true_positives + false_negatives)))   
    print ('precision for class: ', precision_cl)
    print ('recall for class: ', recall_cl)
    print ('mean precision:', np.mean(precision_cl))
    print ('true positives: ', true_positives_cl)
    print ('false positives: ', false_positives_cl)
    print ('false negatives: ', false_negatives_cl)
    print ('true positives: ', true_positives)
    print ('false positives: ', false_positives)
    print ('false negatives: ', false_negatives)    
    print ('boxes in groundtruth: ', boxes_in_ground_truth)
