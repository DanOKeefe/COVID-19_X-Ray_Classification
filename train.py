import argparse
import numpy as np
import os
import tensorflow as tf

from model_def import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    
    X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    print('X train', X_train.shape,'y train', y_train.shape)

    return X_train, y_train


def get_test_data(test_dir):
    
    X_test = np.load(os.path.join(test_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('X test', X_test.shape,'y test', y_test.shape)

    return X_test, y_test
   

if __name__ == "__main__":
        
    args, _ = parse_args()
    
    X_train, y_train = get_train_data(args.train)
    X_test, y_test = get_test_data(args.test)
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        model = get_model()
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.metrics.AUC)    
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(X_val, y_val))

        # evaluate on validation set
        scores = model.evaluate(X_val, y_val, batch_size, verbose=2)
        print("\nVal MSE :", scores)
        
        # save model
        model.save(args.model_dir + '/1')
