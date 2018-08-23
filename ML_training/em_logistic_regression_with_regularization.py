from __future__ import print_function

import math

from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)

set_size = int(input("Please enter size of your training set (1-17000)"))
validation_number = 17000 - set_size

# gather desired features and include any synthetic features
def preprocess_features(california_housing_dataframe):
    """Processes input features from data setself.

    Args:
    california_housing_dataframe: Pandas dataFrame
    Returns:
    dataFrame that contains features to be used for model, including synthetic feature_columns
    """

    # Copy data to new object with only features we want to use
    selected_features = california_housing_dataframe[
    ["latitude", "longitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    ]
    processed_features = selected_features.copy()
    #create synthetic features
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]
    )
    return processed_features

def preprocess_targets(california_housing_dataframe):
        """Prepares target features (i.e., labels) from California housing data set.

        Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
        Returns:
        A dataframe that contains the target feature
        """
        output_targets = pd.DataFrame()
        #create boolean categorical feature representing whether the median_house_value is above a set threshold
        output_targets['median_house_value_is_high'] = (
            california_housing_dataframe["median_house_value"] > 265000).astype(float)
        return output_targets

#next four steps create sample/validation examples and targets
training_examples = preprocess_features(
    california_housing_dataframe.head(set_size)
)
training_targets = preprocess_targets(
    california_housing_dataframe.head(set_size)
)
validation_examples = preprocess_features(
    california_housing_dataframe.tail(validation_number)
)
validation_targets = preprocess_targets(
    california_housing_dataframe.tail(validation_number)
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple feature.

    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: T/F. Whether to shuffle data
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
        Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key,value in dict(features).items()}

    #Construct a dataset and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    #Shuffle data
    if shuffle:
        ds = ds.shuffle(10000)

    # Return next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)]
    )
    return [quantiles[q] for q in quantiles.keys()]

def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
        Set of feature columns
    """

    bucketized_households = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("households"), boundaries=get_quantile_based_buckets(training_examples["households"], 10)
    )
    bucketized_longitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("longitude"), boundaries=get_quantile_based_buckets(training_examples["longitude"], 50)
    )
    bucketized_latitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("latitude"), boundaries=get_quantile_based_buckets(training_examples["latitude"], 50)
    )
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("housing_median_age"), boundaries=get_quantile_based_buckets(training_examples["housing_median_age"], 10)
    )
    bucketized_total_rooms = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("total_rooms"), boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10)
    )
    bucketized_total_bedrooms = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("total_bedrooms"), boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10)
    )
    bucketized_population = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("population"), boundaries=get_quantile_based_buckets(training_examples["population"], 10)
    )
    bucketized_median_income = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("median_income"), boundaries=get_quantile_based_buckets(training_examples["median_income"], 10)
    )
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("rooms_per_person"), boundaries=get_quantile_based_buckets(training_examples["rooms_per_person"], 10)
    )

    long_x_lat = tf.feature_column.crossed_column(
        set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000
    )

    feature_columns = set([
        long_x_lat,
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_total_rooms,
        bucketized_total_bedrooms,
        bucketized_population,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person
    ])

    return feature_columns

# Model Size
def model_size(estimator):
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable for x in ['global_step', 'centered_bias_weight', 'bias_weight', 'Ftrl']):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size

def train_model(learning_rate, regularization_strength, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
  """Trains a linear classification model.

  In addition to training, this function also prints trainig progress information, as well as a plot of the training and validation loss over time.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

    Returns:
        A 'LinearClassifier' object trained on the training data
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object. USING FTRL Optimizer
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(),
      optimizer=my_optimizer
  )

  # Create input functions.
  training_input_fn = lambda:my_input_fn(
    training_examples, training_targets["median_house_value_is_high"], batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
    training_examples, training_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
    validation_examples, validation_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)

    #Print current log_loss
    print(" period %02d : %0.2f" % (period, training_log_loss))

    # Add the loss metrics from this period to our list
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel('LogLoss')
  plt.xlabel('Periods')
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.savefig('testlogisticregressionwithRegularization.png')
  plt.close()

  print("Final LogLoss (on training data): %0.2f" % training_log_losses[9])
  print("Final LogLoss on validation data: %0.2f" % validation_log_losses[9])
  print("Model size: ", model_size(linear_classifier))
  get_metrics(linear_classifier)
  return linear_classifier

# Function for printing and saving model metrics
def get_metrics(linear_classifier):
    predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)
    evaluation_metrics = linear_classifier.evaluate(input_fn = predict_validation_input_fn)
    print("AUC on validation set is %0.2f" % evaluation_metrics['auc'])
    print("Accuracy on validation set is %0.2f" % evaluation_metrics['accuracy'])
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    #get probabilities for positive class
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
    false_positive_rate, true_positive_rate, threshold = metrics.roc_curve(
        validation_targets, validation_probabilities
    )
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0,1], [0,1], label="random classifier")
    plt.legend(loc=2)
    plt.savefig('logisticregressionAUCwithRegularization.png')
