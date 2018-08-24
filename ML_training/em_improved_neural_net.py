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
input_nn_layers = int(input("Please enter number of layers for the neural network: "))
validation_number = 17000 - set_size

def get_hidden_units_input(layers):
    hidden = []
    for layer in range(0, layers):
        input_node_number = int(input("Please enter number of nodes for layer %2d:" % (layer+1)))
        hidden.append(input_node_number)
    return hidden

hidden_units = get_hidden_units_input(input_nn_layers)

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
        #Scale the target to be in units of thousands of dollars
        output_targets['median_house_value'] = (
        california_housing_dataframe["median_house_value"] / 1000.0
        )
        return output_targets

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

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Returns:
        Set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

#helper functions for normalizations
def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val / scale) - 1.0))

def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))
# End helper functions

#nomalize linear scales
def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input 'DataFrame' that has all its features normalized linearly."""
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
    processed_features["population"] = linear_scale(examples_dataframe["population"])
    processed_features["households"] = linear_scale(examples_dataframe["households"])
    processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
    processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
    return processed_features

def train_model(learning_rate, steps, batch_size, training_examples,
    training_targets, validation_examples, validation_targets, my_optimizer=None):
  """Trains a neural network regression model of multiple features.

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
        A 'DNNRegressor' object trained on the training data
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object. USING FTRL Optimizer
  if my_optimizer is None:
      my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )

  # Create input functions.
  training_input_fn = lambda:my_input_fn(training_examples, training_targets["median_house_value"], batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets)
    )
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets)
    )
    #Occasionally print current loss.
    print(" period %02d : %0.2f" % (period, training_root_mean_squared_error))
    #Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.savefig('testOptimizedNeuralNet.png')
  plt.close()

  print("Final RMSE (on training data): %0.2f" % training_root_mean_squared_error)
  print("Final RMSE on validation data: %0.2f" % validation_root_mean_squared_error)
  return dnn_regressor


#Option to train model using normalized dataset
def train_with_linear_normalization(input_rate, input_steps, input_batch_size):
    normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
    normalized_training_examples = normalized_dataframe.head(set_size)
    normalized_validation_examples = normalized_dataframe.tail(validation_number)
    dnn_regressor = train_model(learning_rate=input_rate, steps=input_steps, batch_size=input_batch_size, training_examples=normalized_training_examples, training_targets=training_targets, validation_examples=normalized_validation_examples, validation_targets=validation_targets)
    test(dnn_regressor)

#Option to train model with Adagrad Optimizer
def train_with_adagrad(input_rate, input_steps, input_batch_size):
    normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
    normalized_training_examples = normalized_dataframe.head(set_size)
    normalized_validation_examples = normalized_dataframe.tail(validation_number)
    dnn_regressor = train_model(my_optimizer = tf.train.AdagradOptimizer(learning_rate = input_rate), steps=input_steps, batch_size=input_batch_size, training_examples=normalized_training_examples, training_targets=training_targets, validation_examples=normalized_validation_examples, validation_targets=validation_targets)
    test(dnn_regressor)

#Option to train model with Adam Optimizer
def train_with_adagrad(input_rate, input_steps, input_batch_size):
    normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
    normalized_training_examples = normalized_dataframe.head(set_size)
    normalized_validation_examples = normalized_dataframe.tail(validation_number)
    dnn_regressor = train_model(my_optimizer = tf.train.AdamOptimizer(learning_rate = input_rate), steps=input_steps, batch_size=input_batch_size, training_examples=normalized_training_examples, training_targets=training_targets, validation_examples=normalized_validation_examples, validation_targets=validation_targets)
    test(dnn_regressor)

#Train with optimized feature normalizations
def train_optimal(input_rate, input_steps, input_batch_size):
    data = preprocess_features(california_housing_dataframe)

    normalized_dataframe = pd.DataFrame()

    normalized_dataframe["households"] = log_normalize(data["households"])
    normalized_dataframe["median_income"] = log_normalize(data["median_income"])
    normalized_dataframe["total_bedrooms"] = log_normalize(data["total_bedrooms"])

    normalized_dataframe["latitude"] = linear_scale(data["latitude"])
    normalized_dataframe["longitude"] = linear_scale(data["longitude"])
    normalized_dataframe["housing_median_age"] = linear_scale(data["housing_median_age"])

    normalized_dataframe["population"] = linear_scale(clip(data["population"], 0, 5000))
    normalized_dataframe["rooms_per_person"] = linear_scale(clip(data["rooms_per_person"], 0, 5))
    normalized_dataframe["total_rooms"] = linear_scale(clip(data["total_rooms"], 0, 10000))

    normalized_training_examples = normalized_dataframe.head(set_size)
    normalized_validation_examples = normalized_dataframe.tail(validation_number)
    dnn_regressor = train_model(learning_rate = input_rate, steps=input_steps, batch_size=input_batch_size, training_examples=normalized_training_examples, training_targets=training_targets, validation_examples=normalized_validation_examples, validation_targets=validation_targets)
    test(dnn_regressor)

# Test model with test dataset
def test(dnn_regressor):
    california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets["median_house_value"], num_epochs=1, shuffle=False)

    test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets)
    )

    print("Test RMSE is: %0.2f" % root_mean_squared_error)
