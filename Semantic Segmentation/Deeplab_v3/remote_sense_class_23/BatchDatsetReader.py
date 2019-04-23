"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import tensorflow as tf

class BatchDatset:
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, size, image_dir, annotation_dir):
        print("Initializing Batch Dataset Reader...")
        self.size = size
        self._read_images(image_dir, annotation_dir)

    def _read_images(self, image_dir, annotation_dir):
        self.images = np.load(image_dir).swapaxes(1, 2).swapaxes(2, 3)
        self.annotations = np.load(annotation_dir)[:, np.newaxis]

        # for i, image in enumerate(self.images):
        #     self.images[i] = image.swapaxes(0, 1).swapaxes(1, 2)

        # one_example = np.ones((self.size, self.size))*15
        # for i, annotation in enumerate(self.annotations):
        #     self.annotations[i] = np.equal(annotation, one_example).astype(float)
        self.annotations = self.annotations.swapaxes(1, 2).swapaxes(2, 3)
        # orishape = self.annotations.shape
        # self.annotations = self.annotations.reshape(-1)
        # unique_num = {x:i for i, x in enumerate(np.unique(self.annotations))}
        # for i, v in enumerate(self.annotations):
        #     self.annotations[i] = unique_num[v]
        # self.annotations = self.annotations.reshape(orishape)
        # [0,1,2,3]A
        print(self.images.shape)
        print(self.annotations.shape)

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
#             print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
#             print('self.annotations.shape', self.annotations.shape)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size



        end = self.batch_offset


        return self.images[start:end], self.get_annotations_to_23(self.annotations[start:end],start,end,batch_size)

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def get_annotations_to_23(self,batch_array,start,end,batch_size):
#         print('get_annotations_to_23',batch_array.shape)
        batch_array=np.squeeze(batch_array)
#         print(batch_array.shape)
        ys_new_train = np.zeros(batch_array.shape + (23,))
        for i in range(23):
            ys_new_train[batch_array == i, i] = 1
        ys_train = ys_new_train.astype('int32')
        return ys_train