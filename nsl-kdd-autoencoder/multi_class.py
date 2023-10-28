import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_score
from keras.layers import Dense, Input
from keras.models import Model, model_from_json

# Constants

ENCODING_DIM = 50
N_CLASSES = 5
TARGET_COLS = ['Dos', 'normal', 'Probe', 'R2L', 'U2R']
DROPPED_COLS = ['intrusion', 'Dos', 'normal', 'Probe', 'R2L', 'U2R', 'label']


def preprocess_data(multi_data):
    x_train, x_test = train_test_split(multi_data, test_size=0.25, random_state=42)
    y_train = x_train[TARGET_COLS]
    y_test = x_test[TARGET_COLS]

    x_train = x_train.drop(DROPPED_COLS, axis=1).values.astype('float32')
    x_test = x_test.drop(DROPPED_COLS, axis=1).values.astype('float32')
    y_test = y_test.values

    return x_train, x_test, y_train, y_test


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(ENCODING_DIM, activation="relu")(input_layer)
    output_layer = Dense(input_dim, activation='softmax')(encoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return autoencoder


def save_load_model(model, filepath, weights_path, mode='save'):
    if mode == 'save':
        # Serialize model to JSON and save weights
        with open(filepath, "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(weights_path)

    elif mode == 'load':
        # Load model from JSON and load weights
        with open(filepath, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)

    return model


def plot_metrics(history, metric, filename):
    plt.plot(history[metric])
    plt.plot(history['val_' + metric])
    plt.title(f"Plot of {metric} vs epoch for train and test dataset")
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(filename)
    plt.show()


def build_classifier(input_dim):
    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(48, activation="sigmoid")(input_layer)
    output_layer = Dense(N_CLASSES, activation='sigmoid')(hidden_layer)

    classifier = Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return classifier


def plot_roc_curve(y_test, y_prediction, n_classes):
    fpr_ae = dict()
    tpr_ae = dict()
    roc_auc_ae = dict()
    for i in range(n_classes):
        fpr_ae[i], tpr_ae[i], _ = roc_curve(y_test[:, i], y_prediction[:, i])
        roc_auc_ae[i] = auc(fpr_ae[i], tpr_ae[i])

    for i in range(n_classes):
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_ae[i], tpr_ae[i], label='Keras (area = {:.3f})'.format(roc_auc_ae[i]))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig('plots/ae_classifier_multi_roc' + str(i) + '.png')
        plt.show()

    for j in range(0, y_prediction.shape[1]):
        for i in range(0, y_prediction.shape[0]):
            y_prediction[i][j] = int(round(y_prediction[i][j]))


def print_classification_metrics(y_test, y_prediction):
    print("Recall Score - ", recall_score(y_test, y_prediction.astype('uint8'), average='micro'))
    print("F1 Score - ", f1_score(y_test, y_prediction.astype('uint8'), average='micro'))
    print("Precision Score - ",
          precision_score(y_test, y_prediction.astype('uint8'), average='micro'))


def multi_class_auto_encoder(multi_data, label_encoding):
    n_classes = len(label_encoding.classes_)
    x_train, x_test, y_train, y_test = preprocess_data(multi_data)

    autoencoder = build_autoencoder(x_train.shape[1])
    history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=500, validation_data=(x_test, x_test)).history

    autoencoder = save_load_model(autoencoder, './models/ae_multi.json', './weights/ae_multi.h5')
    test_results = autoencoder.evaluate(x_test, x_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1] * 100}%')

    plot_metrics(history, 'loss', 'plots/ae_multi_loss.png')
    plot_metrics(history, 'accuracy', 'plots/ae_multi_accuracy.png')

    # Classifier
    predictions = autoencoder.predict(x_test)
    ae_classifier = build_classifier(predictions.shape[1])
    his = ae_classifier.fit(predictions, y_test, epochs=200, batch_size=700, validation_split=0.2).history

    ae_classifier = save_load_model(ae_classifier, './models/ae_classifier_multi.json',
                                    './weights/ae_classifier_multi.h5')
    test_results = ae_classifier.evaluate(x_test, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1] * 100}%')

    plot_metrics(his, 'loss', 'plots/ae_classifier_multi_loss.png')
    plot_metrics(his, 'accuracy', 'plots/ae_classifier_multi_accuracy.png')

    y_prediction = ae_classifier.predict(x_test)
    plot_roc_curve(y_test, y_prediction, n_classes)

    print_classification_metrics(y_test, y_prediction)
