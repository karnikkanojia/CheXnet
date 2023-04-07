from utils.dataloader import get_generator, categories
import numpy as np

BATCH_SIZE = 32


def compute_class_freqs(labels):
    N = len(labels)
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies
    return positive_frequencies, negative_frequencies

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += K.mean(-(pos_weights[i] *y_true[:,i] * K.log(y_pred[:,i] + epsilon)
                             + neg_weights[i]* (1 - y_true[:,i]) * K.log( 1 - y_pred[:,i] + epsilon)))
        return loss

    return weighted_loss


if __name__ == '__main__':
    train_generator = get_generator(
        dtype='train', batch_size=32, image_size=(1024, 1024))
    valid_generator = get_generator(
        dtype='val', batch_size=1, image_size=(1024, 1024))
    test_generator = get_generator(
        dtype='test', batch_size=1, image_size=(1024, 1024))
    print('Train generator length: ', len(train_generator))
    print('Valid generator length: ', len(valid_generator))
    print('Test generator length: ', len(test_generator))
    train_labels = np.zeros((32*len(train_generator), 14), dtype=np.int8)
    for i in range(len(train_generator)):
        if i+1 == len(train_generator): continue
        if i%100 == 0: print(i)
        train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE,
                     :14] = train_generator[i][1]
    train_pos_freq, train_neg_freq = compute_class_freqs(train_labels)
    print('Train positive frequencies: ', train_pos_freq)
    print('Train negative frequencies: ', train_neg_freq)
