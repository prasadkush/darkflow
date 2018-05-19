import os
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
    #s_file = shelve.open('steps2')
    #l_file = shelve.open('loss2')
    s_file_ = shelve.open('steps_lr1e4')
    l_file_ = shelve.open('loss_lr1e4')
    #step_plot = s_file['steps']
    #loss_plot = l_file['loss']
    #lossmva_plot = l_file['lossmva']
    #step_no = step_plot[step_plot.shape[0] - 1]
    #loss_mva = lossmva_plot[lossmva_plot.shape[0] - 1]

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
        plt.savefig('lossplot4.png')
        
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
            # extract ground truth data for image from csv reader
            # make changes in postprocess function to compute true positives, false positives, false negatives and return precision, recall, IOU
            # use csv reader code in udacity_voc_csv.py to find ground truth data for an image name
            self.framework.postprocess(prediction,
                os.path.join(inp_path, all_inp[i]))
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

    # code added for detector performance

    # parsig the ground truth data
def predict_params(self):
    boundingboxes = list()
    images = list()
    ground_truth = dict()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    C = self.meta['classes']
    csv_fname = os.path.join('/Users/Kush/Desktop/PyTorch-Python/udacity/darkflow/udacity_valid.csv')
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )

        for row in spamreader:
            img_name = row[0]
            w = 1920
            h = 1200
            boxes_in_img = list()
             # for storing class label xmin ymin xmax ymax prob (1 if class present else 0)
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
                boxes_in_img.append(boundbox)
            
            images.append(img_name)
            boundingboxes.append(boxes_in_img)
        ground_truth['imgs'] = images
        ground_truth['bounding_boxes'] = boundingboxes

    inp_path = self.FLAGS.test
    batch = min(self.FLAGS.batch, len(ground_truth['images']))
    for j in range(len(ground_truth['images']) // batch):
        inp_feed = list(); new_all = list()
        all_inp = gound_truth['images'][j*batch: (j*batch+batch)]
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
        for i, prediction in enumerate(out):
            # make changes in postprocess function to compute true positives, false positives, false negatives and return precision, recall, IOU
            tp, fp, fn = self.framework.postprocess(prediction,
                os.path.join(inp_path, all_inp[i]), ground_truth_batch[j])
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        stop = time.time(); last = stop - start
  
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
        print 'precision: ', float(true_positives/(true_positives + false_positives))
        print 'recall: ', float(true_positives/(true_positives + false_negatives))
      

  