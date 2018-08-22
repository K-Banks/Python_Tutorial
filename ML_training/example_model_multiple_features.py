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

    Args:
        input_features: the names of numerical input features to use
    Returns:
        Set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
  """Trains a linear regression model of multiple features.

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
        A 'LinearRegressor' object trained on the training data
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
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
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
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
  plt.savefig('testmultiplefeatures.png')

  print("Final RMSE (on training data): %0.2f" % training_root_mean_squared_error)
  print("Final RMSE on validation data: %0.2f" % validation_root_mean_squared_error)
  return linear_regressor

# Test model with test dataset
def test(linear_regressor):
    california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets["median_house_value"], num_epochs=1, shuffle=False)

    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets)
    )

    print("Test RMSE is: %0.2f" % root_mean_squared_error)
