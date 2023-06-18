from KNN import KNN
from sklearn import metrics
import samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
import seaborn as sns
import pandas as pd


DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  width, height = datum.width, datum.height
  features = np.zeros((height, width+2), dtype=np.float64)
  ones_count_row = np.zeros(height, dtype=np.float64)
  zeros_count_row = np.zeros(height, dtype=np.float64)
  
  for y in range(height):
    for x in range(width):
      if datum.getPixel(x, y) > 0:
        features[y, x] = 1
        ones_count_row[y] += 1
      else:
        features[y, x] = 0
        zeros_count_row[y] += 1
  
  # Append counts as a new column to the feature array
  counts = np.column_stack((ones_count_row, zeros_count_row))
  features[:, -2:] = counts
    
  return features


def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  width, height = datum.width, datum.height
  features = np.zeros((height, width+2), dtype=np.float64)
  ones_count_row = np.zeros(height, dtype=np.float64)
  zeros_count_row = np.zeros(height, dtype=np.float64)
  
  for y in range(height):
    for x in range(width):
      if datum.getPixel(x, y) > 0:
        features[y, x] = 1
        ones_count_row[y] += 1
      else:
        features[y, x] = 0
        zeros_count_row[y] += 1
  
  # Append counts as a new column to the feature array
  counts = np.column_stack((ones_count_row, zeros_count_row))
  features[:, -2:] = counts
    
  return features

def perform_training(X_train, training_labels, X_test, test_labels, k_neighbors=3):
  final_result = []

  # perform training for 10%, 20%, ..., 100%
  # 10 iterations 
  for i in range(10):
    # for each iteration i, we will be randomly sampling i/10 of the total training data
    num_samples = X_train.shape[0]
    sample_size = int(((i + 1) / 10) * num_samples)
    random_indexes = np.random.choice(num_samples,size= sample_size, replace=False)
    sampled_X_train = X_train[random_indexes]
    sampled_training_labels = training_labels[random_indexes]
    training_elapsed_time = 0
    testing_elapsed_time = 0
    

    knn = KNN(k=k_neighbors)

    training_start_time = time.time()
    knn.train(sampled_X_train, sampled_training_labels)
    training_end_time = time.time()
    training_elapsed_time = training_end_time - training_start_time

    testing_start_time = time.time()
    knn_predictions = knn.predict(X_test)
    testing_end_time = time.time()
    testing_elapsed_time = testing_end_time - testing_start_time
    knn_accuracy = accuracy(test_labels, knn_predictions)
    
    
    # Generating Confusion Matrix
    knn_confusion_matrix = metrics.confusion_matrix(test_labels, knn_predictions)
    # Generate data to append to final_result
    sampled_res = {'title': str((i + 1) *10) + '% of data', 'percentage_of_data': int(i+1) / 10, 'training_elapsed_time': training_elapsed_time, 'testing_elapsed_time': testing_elapsed_time, 'best_prediction_list': knn_predictions, 'accuracy': knn_accuracy, 'confusion_matrix': knn_confusion_matrix, 'k': k_neighbors}
    # Append sampled data results to final_result
    final_result.append(sampled_res)
  return final_result

# Compute accuracy (ratio of the number of correct classifcation to the total number of samples)
def accuracy(y_true, y_pred):
    error = np.sum(y_true != y_pred) / len(y_pred)
    return 1 - error

def create_graph_for_training_results(plot_type, title, x_values, y_values, x_label, y_label, path, confusion_matrix=None) -> None:
    """
    Plotting results using matplotlib
    """
    fig, ax = plt.subplots()
    if plot_type == 'line':
        ax.plot(x_values, y_values, linewidth=2, color='red')
    elif plot_type == 'scatter':
        ax.scatter(x_values, y_values, color='blue')
    elif plot_type == 'confusion_matrix':
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    # Set plot title and axes labels
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    if plot_type != 'confusion_matrix':
      ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Add gridlines to the plot
    if plot_type != 'confusion_matrix':
      ax.grid()

    # Increase the font size of the title and axis labels
    ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)

    # Increase the size of the plot
    fig.set_size_inches(12, 10)
    # plt.show()
    plt.savefig(path)
    plt.close()


def run_training_on_digits():
# Get Training and test data
    numTest = 1000
    numTraining = 5000
    numValidation = 1000
    rawTrainingData = samples.loadDataFile("data/digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    training_labels = np.array(samples.loadLabelsFile("data/digitdata/traininglabels", numTraining))
    rawValidationData = samples.loadDataFile("data/digitdata/validationimages", numValidation,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validation_labels = samples.loadLabelsFile("data/digitdata/validationlabels", numValidation)
    rawTestData = samples.loadDataFile("data/digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    test_labels = np.array(samples.loadLabelsFile("data/digitdata/testlabels", numTest))


    # Extract features
    print("Starting Training and Testing for Digits Dataset")
    print("Starting Extracting features from digits...")
    training_data = list(map(basicFeatureExtractorDigit, rawTrainingData))
    validation_data = list(map(basicFeatureExtractorDigit, rawValidationData)) 

    test_data = list(map(basicFeatureExtractorDigit, rawTestData))

    X_train = np.array([img.flatten() for img in training_data])
    X_test = np.array([img.flatten() for img in test_data])
    
    print("Finished Extracting features from digits...")

    training_result_no_validation = perform_training(X_train=X_train, training_labels=training_labels, X_test=X_test, test_labels=test_labels, k_neighbors=10)

    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_no_validation]
    y_values = [res['accuracy'] for res in training_result_no_validation]
    create_graph_for_training_results(plot_type='line', title='Accuracy vs Percentage of Training Data (Digits) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Accuracy', path='./results/digits/accuracy_vs_percentage_of_Digits_Dataset_no_validation.png')

    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_no_validation]
    y_values = [res['training_elapsed_time'] for res in training_result_no_validation]
    create_graph_for_training_results(plot_type='scatter', title='Elapsed Training Time vs Percentage of Training Data (Digits) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Elapsed Time (seconds)', path='./results/digits/training_time_vs_percentage_of_Digits_Dataset_no_validation.png')

    # plot confusion_matrix
    for i in range(10):
      confusion_matrixx = training_result_no_validation[i]['confusion_matrix']
      create_graph_for_training_results(plot_type='confusion_matrix', title='Confusion Matrix for ' + str((i + 1) * 10) +'%' +' of training data (Digits) used for Training', x_values=None, y_values= None, x_label='Predicted Cases', y_label='True Cases', path='./results/digits/confusion_matrix_no_validation' + str((i + 1) * 10) +'percent.png', confusion_matrix=confusion_matrixx)
      
      # Define the labels in the order you want them to appear in the confusion matrix
      labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      # Compute the confusion matrix
      conf_matrix = metrics.confusion_matrix(test_labels, training_result_no_validation[i]['best_prediction_list'], labels=labels)
      # Unpack the values of the confusion matrix
      fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
      fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
      tp = np.diag(conf_matrix)
      tn = conf_matrix.sum() - (fp + fn + tp)
      accuracy = metrics.accuracy_score(test_labels, training_result_no_validation[i]['best_prediction_list'])
      precision = metrics.precision_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')
      recall = metrics.recall_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')
      f1 = metrics.f1_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')

      # Creating a pandas DataFrame to store the metrics
      data = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'], 'Value': [accuracy, precision, recall, f1, tp, tn, fp, fn]}
      metric_df = pd.DataFrame(data)

      # Exporting the DataFrame to a CSV file
      metric_df.to_csv('./results/digits/metrics_no_validation_digits_' + str((i + 1) * 10) + '_percent_of_data.csv', index=False)

    df = pd.DataFrame.from_dict(training_result_no_validation)
    df = df.drop('confusion_matrix', axis=1)
    df = df.drop('best_prediction_list', axis=1)
    df.to_csv('./results/digits/results_no_validation.csv', float_format='%g')

    accuracy_list_no_validation = [res['accuracy'] for res in training_result_no_validation]

    print("The highest accuracy for training on data reserved for training: ", max(accuracy_list_no_validation))
    print("The average accuracy for training on data reserved for training: ", np.average(np.array(accuracy_list_no_validation)))
    print("The standard deviation for training on data reserved for training: ", np.std(np.array(accuracy_list_no_validation)))



    # Training including validation data with training dataset
    print("Starting Training on Validation + Training data")

    # Concatenate training_data and validation_data if we want to train on both
    training_data = np.concatenate([training_data, validation_data], axis=0)
    # preprocess the concatenated training data
    X_train = np.array([img.flatten() for img in training_data])
    # concatentate traiing labels and validation labels if we want to train on both
    training_labels = np.concatenate([training_labels, validation_labels], axis=0)
    training_result_with_validation = perform_training(X_train=X_train, training_labels=training_labels, X_test=X_test, test_labels=test_labels, k_neighbors=9)

    print("Finished Training on Validation + Training data")

    # Plotting training results for Training + Validation data
    
    # plot accuracy vs percentage of data
    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_with_validation]
    y_values = [res['accuracy'] for res in training_result_with_validation]
    create_graph_for_training_results(plot_type='line', title='Accuracy vs Percentage of Training + Validation Data (Digits) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Accuracy', path='./results/digits/accuracy_vs_percentage_of_Digits_Dataset_with_validation.png')
    
    # plot elapsed training time vs percentage of data
    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_with_validation]
    y_values = [res['training_elapsed_time'] for res in training_result_with_validation]
    create_graph_for_training_results(plot_type='scatter', title='Elapsed Training Time vs Percentage of Training + Validation Data (Digits) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Elapsed Time (seconds)', path='./results/digits/training_time_vs_percentage_of_Digits_Dataset_with_validation.png')

    # plot confusion_matrix
    for i in range(10):
      confusion_matrixx = training_result_with_validation[i]['confusion_matrix']
      create_graph_for_training_results(plot_type='confusion_matrix', title='Confusion Matrix for ' + str((i + 1) * 10) +'%' +' of training + Validation data (Digits) used for Training', x_values=None, y_values= None, x_label='Predicted Cases', y_label='True Cases', path='./results/digits/confusion_matrix_with_validation' + str((i + 1) * 10) +'percent.png', confusion_matrix=confusion_matrixx)

      # Define the labels in the order you want them to appear in the confusion matrix
      labels = [0, 1, 2, 3, 4, 5 ,6 ,7 ,8, 9]
      # Compute the confusion matrix
      conf_matrix = metrics.confusion_matrix(test_labels, training_result_with_validation[i]['best_prediction_list'], labels=labels)
      # Unpack the values of the confusion matrix
      fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
      fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
      tp = np.diag(conf_matrix)
      tn = conf_matrix.sum() - (fp + fn + tp)
      accuracy = metrics.accuracy_score(test_labels, training_result_with_validation[i]['best_prediction_list'])
      precision = metrics.precision_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')
      recall = metrics.recall_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')
      f1 = metrics.f1_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')

      # Creating a pandas DataFrame to store the metrics
      data = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'], 'Value': [accuracy, precision, recall, f1, tp, tn, fp, fn]}
      metric_df = pd.DataFrame(data)

      # Exporting the DataFrame to a CSV file
      metric_df.to_csv('./results/digits/metrics_with_validation_digits_' + str((i + 1) * 10) + '_percent_of_data.csv', index=False)


    df = pd.DataFrame.from_dict(training_result_with_validation)
    df = df.drop('confusion_matrix', axis=1)
    df = df.drop('best_prediction_list', axis=1)
    df.to_csv('./results/digits/results_with_validation.csv', float_format='%g')

    accuracy_list_with_validation = [res['accuracy'] for res in training_result_with_validation]

    print("The highest accuracy for training on validation + training data: ", max(accuracy_list_with_validation))
    print("The average accuracy for training on validation + training data: ", np.average(np.array(accuracy_list_with_validation)))
    print("The standard deviation for training on validation + training data: ", np.std(np.array(accuracy_list_with_validation)))

    print("Finished Training and Testing for Digits Dataset")


def run_training_on_faces():

    # Training on Faces Dataset
    print("Starting Training and Testing for Faces Dataset")

    numTest = 150
    numTraining = 451
    numValidation = 301

    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    training_labels = np.array(samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining))
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numValidation,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validation_labels = samples.loadLabelsFile("facedata/facedatatrainlabels", numValidation)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    test_labels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)

    # Extract features
    print("Starting Extracting features from Faces...")
    training_data = list(map(basicFeatureExtractorFace, rawTrainingData))
    validation_data =list(map(basicFeatureExtractorFace, rawValidationData))

    test_data = list(map(basicFeatureExtractorFace, rawTestData))

    X_train = np.array([img.flatten() for img in training_data])
    X_test = np.array([img.flatten() for img in test_data])
    
    print("Finished Extracting features from Faces...")

    training_result_no_validation = perform_training(X_train=X_train, training_labels=training_labels, X_test=X_test, test_labels=test_labels)

    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_no_validation]
    y_values = [res['accuracy'] for res in training_result_no_validation]
    create_graph_for_training_results(plot_type='line', title='Accuracy vs Percentage of Training Data (Faces) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Accuracy', path='./results/faces/accuracy_vs_percentage_of_faces_dataset_no_validation.png')

    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_no_validation]
    y_values = [res['training_elapsed_time'] for res in training_result_no_validation]
    create_graph_for_training_results(plot_type='scatter', title='Elapsed Training Time vs Percentage of Training Data (Digits) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Elapsed Time (seconds)', path='./results/faces/training_time_vs_percentage_of_faces_dataset_no_validation.png')

    # plot confusion_matrix
    for i in range(10):
      confusion_matrixx = training_result_no_validation[i]['confusion_matrix']
      create_graph_for_training_results(plot_type='confusion_matrix', title='Confusion Matrix for ' + str((i + 1) * 10) +'%' +' of training data (Faces) used for Training', x_values=None, y_values= None, x_label='Predicted Cases', y_label='True Cases', path='./results/faces/confusion_matrix_no_validation' + str((i + 1) * 10) +'percent.png', confusion_matrix=confusion_matrixx)

      labels = [0,1]
      tn, fp, fn, tp = metrics.confusion_matrix(test_labels, training_result_no_validation[i]['best_prediction_list'], labels=labels).ravel()
      accuracy = metrics.accuracy_score(test_labels, training_result_no_validation[i]['best_prediction_list'])
      precision = metrics.precision_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')
      recall = metrics.recall_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')
      f1 = metrics.f1_score(test_labels, training_result_no_validation[i]['best_prediction_list'], average='macro')

      # Creating a pandas DataFrame to store the metrics
      data = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'], 'Value': [accuracy, precision, recall, f1, tp, tn, fp, fn]}
      metric_df = pd.DataFrame(data)

      # Exporting the DataFrame to a CSV file
      metric_df.to_csv('./results/faces/metrics_no_validation_faces_' + str((i + 1) * 10) + '_percent_of_data.csv', index=False)


    df = pd.DataFrame.from_dict(training_result_no_validation)
    df = df.drop('confusion_matrix', axis=1)
    df = df.drop('best_prediction_list', axis=1)
    df.to_csv('./results/faces/results_no_validation.csv', float_format='%g')

    accuracy_list_no_validation = [res['accuracy'] for res in training_result_no_validation]

    print("The highest accuracy for training on data reserved for training: ", max(accuracy_list_no_validation))
    print("The average accuracy for training on data reserved for training: ", np.average(np.array(accuracy_list_no_validation)))
    print("The standard deviation for training on data reserved for training: ", np.std(np.array(accuracy_list_no_validation)))


    # Training including validation data with training dataset
    print("Starting Training on Validation + Training data")
    # Concatenate training_data and validation_data if we want to train on both
    training_data = np.concatenate([training_data, validation_data], axis=0)
    # preprocess the concatenated training data
    X_train = np.array([img.flatten() for img in training_data])
    # concatentate traiing labels and validation labels if we want to train on both
    training_labels = np.concatenate([training_labels, validation_labels], axis=0)
    training_result_with_validation = perform_training(X_train=X_train, training_labels=training_labels, X_test=X_test, test_labels=test_labels)
    print("Finished Training on Validation + Training data")

    # Plotting training results for Training + Validation data

    # plot accuracy vs percentage of data
    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_with_validation]
    y_values = [res['accuracy'] for res in training_result_with_validation]
    create_graph_for_training_results(plot_type='line', title='Accuracy vs Percentage of Training + Validation Data (Faces) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Accuracy', path='./results/faces/accuracy_vs_percentage_of_faces_dataset_with_validation.png')
    
    # plot elapsed training time vs percentage of data
    # get x and y values
    x_values = [res['percentage_of_data'] for res in training_result_with_validation]
    y_values = [res['training_elapsed_time'] for res in training_result_with_validation]
    create_graph_for_training_results(plot_type='scatter', title='Elapsed Training Time vs Percentage of Training + Validation Data (Faces) used for Training', x_values=x_values, y_values = y_values, x_label='Percentage of Data', y_label='Elapsed Time (seconds)', path='./results/faces/training_time_vs_percentage_of_faces_dataset_with_validation.png')

    # plot confusion_matrix
    for i in range(10):
      confusion_matrix = training_result_with_validation[i]['confusion_matrix']
      create_graph_for_training_results(plot_type='confusion_matrix', title='Confusion Matrix for ' + str((i + 1) * 10) +'%' +' of training + Validation data (Faces) used for Training', x_values=None, y_values= None, x_label='Predicted Cases', y_label='True Cases', path='./results/faces/confusion_matrix_with_validation' + str((i + 1) * 10) +'percent.png', confusion_matrix=confusion_matrix)

      labels = [0,1]
      tn, fp, fn, tp = metrics.confusion_matrix(test_labels, training_result_with_validation[i]['best_prediction_list'], labels=labels).ravel()
      accuracy = metrics.accuracy_score(test_labels, training_result_with_validation[i]['best_prediction_list'])
      precision = metrics.precision_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')
      recall = metrics.recall_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')
      f1 = metrics.f1_score(test_labels, training_result_with_validation[i]['best_prediction_list'], average='macro')

      # Creating a pandas DataFrame to store the metrics
      data = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'], 'Value': [accuracy, precision, recall, f1, tp, tn, fp, fn]}
      metric_df = pd.DataFrame(data)

      # Exporting the DataFrame to a CSV file
      metric_df.to_csv('./results/faces/metrics_with_validation_faces_' + str((i + 1) * 10) + '_percent_of_data.csv', index=False)


    df = pd.DataFrame.from_dict(training_result_with_validation)
    df = df.drop('confusion_matrix', axis=1)
    df = df.drop('best_prediction_list', axis=1)
    df.to_csv('./results/faces/results_with_validation.csv', float_format='%g')

    accuracy_list_with_validation = [res['accuracy'] for res in training_result_with_validation]

    print("The highest accuracy for training on validation + training data: ", max(accuracy_list_with_validation))
    print("The average accuracy for training on validation + training data: ", np.average(np.array(accuracy_list_with_validation)))
    print("The standard deviation for training on validation + training data: ", np.std(np.array(accuracy_list_with_validation)))

    print("Finished Training and Testing for Faces Dataset")

def main():
    run_training_on_digits()
    run_training_on_faces()
    return 0

if __name__== "__main__":
    main()