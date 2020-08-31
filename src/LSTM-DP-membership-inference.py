import numpy as np
import pandas as pd


path = ''

# depent on tensorflow 1.14
import tensorflow as tf
from mia.estimators import ShadowModelBundle, prepare_attack_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
# privacy package
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


# set random seed
import random
random.seed(19122)

# set random seed
np.random.seed(19122)
tf.set_random_seed(19122)

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

def split_to_be_divisible(X, y, shadow_perc, batch_size):
    """
    Split a dataframe into target dataset and shadow dataset, and make them divisible by batch size.

    :param X: genotype data
    :param y: phenotype data
    :param shadow_perc: specified percent for shadow dataset, target_perc = 1 - shadow_perc
    :param batch_size: batch_size for training process

    :return: target datasets, shadow datasets
    """

    # stop and output error, if X and y have different number of individuals.
    assert y.shape[0] == X.shape[0]

    # calculate sample size of target and shadow
    total_row = X.shape[0]
    num_shadow_row = int(total_row * shadow_perc) - int(total_row * shadow_perc) % batch_size
    num_target_row = (total_row - num_shadow_row) - (total_row - num_shadow_row) % batch_size

    # split train and valid
    random_row = np.random.permutation(total_row)
    shadow_row = random_row[:num_shadow_row]
    target_row = random_row[-num_target_row:]

    target_X = X.iloc[target_row]
    shadow_X = X.iloc[shadow_row]

    target_y = y.iloc[target_row]
    shadow_y = y.iloc[shadow_row]

    return target_X, target_y, shadow_X, shadow_y


def target_model():
    """The architecture of the target model.
    The attack is white-box, hence the attacker is assumed to know this architecture too.

    :return: target model
    """
    classifier = tf.keras.Sequential([
        tf.keras.Input((feature_size), name='feature'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax, kernel_regularizer=l1(kernel_regularization))
    ])

    if dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=int(microbatches_perc * batch_size),
            learning_rate=learning_rate)

        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)
    else:
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)

    # Compile model with Keras
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return classifier


def shadow_model():
    """The architecture of the shadow model is same as target model, because the attack is white-box,
    hence the attacker is assumed to know this architecture too.

    :return: shadow model
    """

    classifier = Sequential()
    classifier = tf.keras.Sequential([
        tf.keras.Input((feature_size), name='feature'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax,  kernel_regularizer=l1(kernel_regularization))
    ])

    if dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=int(microbatches_perc * batch_size),
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)

    else:
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)

    # Compile model with Keras
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return classifier


def  personal_income_classification():
    """
        Load a dataset and encode categorical variables.

    """


    df = pd.read_csv(path + "data/adult.all.txt", sep=", ")

    print(df.shape)
    print(df.columns)

    # Renaming the columns
    df.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation',
                  'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss',
                  'Hours-per-week', 'Native-country', 'Salary']

    df['Age'] = df['Age'].astype(np.float32)

    df['fnlwgt'] = df['fnlwgt'].astype(np.float32)
    df['Education-num'] = df['Education-num'].astype(np.float32)
    df['Capital-gain'] = df['Capital-gain'].astype(np.float32)
    df['Capital-loss'] = df['Capital-loss'].astype(np.float32)
    df['Hours-per-week'] = df['Hours-per-week'].astype(np.float32)

    df['Workclass'] = df['Workclass'].astype('category').cat.codes.astype(np.float32)
    df['Education'] = df['Education'].astype('category').cat.codes.astype(np.float32)
    df['Marital-status'] = df['Marital-status'].astype('category').cat.codes.astype(np.float32)
    df['Occupation'] = df['Occupation'].astype('category').cat.codes.astype(np.float32)
    df['Relationship'] = df['Relationship'].astype('category').cat.codes.astype(np.float32)
    df['Race'] = df['Race'].astype('category').cat.codes.astype(np.float32)
    df['Sex'] = df['Sex'].astype('category').cat.codes.astype(np.float32)
    df['Native-country'] = df['Native-country'].astype('category').cat.codes.astype(np.float32)
    df['Salary'] = df['Salary'].astype('category').cat.codes.astype(np.float32)

    df_train = df[
        ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship',
         'Race', 'Sex', 'Capital-gain', 'Capital-loss',
         'Hours-per-week', 'Native-country']]

    df_test = df[['Salary']]

    print(df.head())
    return df_train, df_test

def main():
    print("Training the target model...")
    # split target dataset to train and valid, and make them evenly divisible by batch size

    target_X_train, target_y_train, target_X_valid, target_y_valid = split_to_be_divisible(target_X,
                                                                                           target_y,
                                                                                           0.2,
                                                                                           batch_size)


    tm = target_model()
    tm.fit(target_X_train.values,
           target_y_train.values,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=[target_X_valid.values, target_y_valid.values],
           verbose=1)

    print("Training the shadow models.")
    # train only one shadow model
    SHADOW_DATASET_SIZE = int(shadow_X.shape[0] / 2)
    smb = ShadowModelBundle(
        shadow_model,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=1,
    )
    # Training the shadow models with same parameter of target model, and generate attack data...
    attacker_X, attacker_y = smb.fit_transform(shadow_X.values, shadow_y.values,
                                               fit_kwargs=dict(epochs=epochs,
                                                               batch_size=batch_size,
                                                               verbose=1),
                                               )

    print("Training attack model...")
    clf = RandomForestClassifier(max_depth=2)
    clf.fit(attacker_X, attacker_y)

    # Test the success of the attack.
    ATTACK_TEST_DATASET_SIZE = unused_X.shape[0]
    # Prepare examples that were in the training, and out of the training.
    data_in = target_X_train[:ATTACK_TEST_DATASET_SIZE], target_y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = unused_X[:ATTACK_TEST_DATASET_SIZE], unused_y[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(tm, data_in, data_out)

    # Compute the attack accuracy.
    attack_guesses = clf.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)
    print('attack accuracy: {}'.format(attack_accuracy))
    acc = accuracy_score(real_membership_labels, attack_guesses)
    print('attack acc: {}'.format(acc))

    prec =  precision_score(real_membership_labels, attack_guesses)
    print('Precision: {}'.format(prec))

    recall = recall_score(real_membership_labels, attack_guesses)
    print('Recall: {}'.format(recall))

    fscore = f1_score(real_membership_labels, attack_guesses)
    print('F1-Score: {}'.format(fscore))


if __name__ == '__main__':

    # parameters
    dpsgd = True

    # target model hyper-parameters same as Lasso-dp
    epochs = 10
    batch_size = 32
    microbatches_perc = .5
    learning_rate = 0.01
    kernel_regularization = 1.2 #0.001352
    noise_multiplier = 0.8
    l2_norm_clip = 0.8 #1.0

    print("Loading and splitting dataset")

    X, y = personal_income_classification()
    target_X, target_y, shadow_X, shadow_y = split_to_be_divisible(X, y, 0.5, batch_size=80)


    shadow_X, shadow_y, unused_X, unused_y = split_to_be_divisible(shadow_X,
                                                                   shadow_y,
                                                                   0.3,
                                                                   batch_size)
    feature_size = target_X.shape[1]
    main()
