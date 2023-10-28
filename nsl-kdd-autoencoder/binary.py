import os
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, recall_score, f1_score, precision_score

from enums import ModelMode
from constants import EPOCHS_AUTOENCODER, EPOCHS_CLASSIFIER, BATCH_SIZE, CLASSIFIER_SPLIT, MODEL_PATH, WEIGHTS_PATH, \
    ENCODING_DIM, PLOTS_PATH


def prepare_data(bin_data):
    # Split the dataset
    x_train, x_test = train_test_split(bin_data, test_size=0.25, random_state=42)

    # Drop unnecessary columns
    drop_cols = ['intrusion', 'abnormal', 'normal', 'label']
    x_train = x_train.drop(drop_cols, axis=1).values.astype('float32')
    y_test = x_test['intrusion'].values
    x_test = x_test.drop(drop_cols, axis=1).values.astype('float32')

    return x_train, x_test, y_test


def create_binary_autoencoder(input_dim):
    # Define the model architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(ENCODING_DIM, activation="relu")(input_layer)
    output_layer = Dense(input_dim, activation='softmax')(encoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return autoencoder


def train_model(model, x_train, x_test, model_type=ModelMode.AUTOENCODER):
    # Train the model
    if model_type == ModelMode.AUTOENCODER:
        history = model.fit(x_train, x_train, epochs=EPOCHS_AUTOENCODER, batch_size=BATCH_SIZE,
                            validation_data=(x_test, x_test)).history
    else:
        history = model.fit(x_train, x_test, epochs=EPOCHS_CLASSIFIER, batch_size=BATCH_SIZE,
                            validation_split=CLASSIFIER_SPLIT).history
    return history


def save_load_model(model, model_name, weights_name):
    # Check if the model file exists, if not, train and save it
    if not os.path.isfile(os.path.join(MODEL_PATH, model_name)):

        with open(os.path.join(MODEL_PATH, model_name), "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(os.path.join(WEIGHTS_PATH, weights_name))

    else:
        with open(os.path.join(MODEL_PATH, model_name), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(WEIGHTS_PATH, weights_name))

    return model


def plot_metrics(history, metric, filename):
    plt.plot(history[metric])
    plt.plot(history[f'val_{metric}'])
    plt.title(f"Plot of {metric} vs epoch for train and test dataset")
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(os.path.join(PLOTS_PATH, filename))
    plt.show()


def create_classifier(input_dim):
    # Define the classifier architecture
    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(50, activation="sigmoid")(input_layer)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer)
    classifier = Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def evaluate_classifier(classifier, x_test, y_test):
    # Evaluate the model
    test_results = classifier.evaluate(x_test, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1] * 100}%')
    return test_results


def plot_roc_curve(y_test, y_prediction):
    fpr, tpr, _ = roc_curve(y_test, y_prediction)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLOTS_PATH, 'ae_binary_roc.png'))
    plt.show()


def print_classification_metrics(y_test, y_classes):
    print("Recall Score - ", recall_score(y_test, y_classes))
    print("F1 Score - ", f1_score(y_test, y_classes))
    print("Precision Score - ", precision_score(y_test, y_classes))


def binary_auto_encoder(bin_data):
    x_train, x_test, y_test = prepare_data(bin_data)

    # Train or load autoencoder model
    autoencoder = create_binary_autoencoder(x_train.shape[1])
    history_ae = train_model(autoencoder, x_train, x_test, model_type=ModelMode.AUTOENCODER)
    # autoencoder = save_load_model(autoencoder, BINARY_AE_MODEL, BINARY_AE_WEIGHTS)

    # Plot metrics for autoencoder
    plot_metrics(history_ae, 'loss', 'ae_binary_loss.png')
    plot_metrics(history_ae, 'accuracy', 'ae_binary_accuracy.png')

    # Use the autoencoder to get predictions
    predictions = autoencoder.predict(x_test)

    # Train or load classifier model
    classifier = create_classifier(predictions.shape[1])
    history_classifier = train_model(classifier, predictions, y_test, model_type=ModelMode.CLASSIFIER)
    # classifier, history_classifier = save_load_model(classifier, CLASSIFIER_MODEL,
    #                                                  CLASSIFIER_WEIGHTS)

    # Plot metrics for classifier
    plot_metrics(history_classifier, 'loss', 'ae_classifier_binary_loss.png')
    plot_metrics(history_classifier, 'accuracy', 'ae_classifier_binary_accuracy.png')

    # Evaluate classifier
    evaluate_classifier(classifier, x_test, y_test)

    # Get classifier predictions
    y_pred = classifier.predict(x_test).ravel()
    y_classes = (classifier.predict(x_test) > 0.5).astype('int32')

    # ROC curve and other metrics
    plot_roc_curve(y_test, y_pred)
    print_classification_metrics(y_test, y_classes)
