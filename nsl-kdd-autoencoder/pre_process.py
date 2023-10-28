import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

from enums import ClassifierMode


def normalize_columns(df, columns):
    """Normalizes specified columns of a given dataframe using StandardScaler."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def change_labels(df):
    """Replace specific attack labels with broader categories."""
    mappings = {
        'Dos': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                'udpstorm', 'worm'],
        'R2L': ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'],
        'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
        'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    }
    for category, labels in mappings.items():
        df['label'].replace(labels, category, inplace=True)


def plot_distribution(df, column, title, save_path):
    """Plot distribution of specified column."""
    plt.figure(figsize=(8, 8))
    plt.pie(df[column].value_counts(), labels=df[column].unique(), autopct='%0.2f%%')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def process_data(data, numeric_cols, categorical_cols):
    # Normalize numeric columns
    data = normalize_columns(data, numeric_cols)

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols)

    return data


def classification_processing(data, numeric_cols, categorical_cols, show_plot, mode):
    data = process_data(data, numeric_cols, categorical_cols)

    # Binary classification
    def binary_classification():
        bin_data = data.copy()
        bin_data['label'] = bin_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'abnormal')
        le1 = LabelEncoder()
        bin_data['intrusion'] = le1.fit_transform(bin_data['label'])
        bin_data = pd.get_dummies(bin_data, columns=['label'], prefix="", prefix_sep="")
        bin_data['label'] = bin_data['normal'].apply(lambda x: 'normal' if x == 1 else 'abnormal')
        np.save("./labels/le1_classes.npy", le1.classes_, allow_pickle=True)
        if show_plot:
            plot_distribution(bin_data, 'label', 'Pie chart distribution of normal and abnormal labels',
                              './plots/Pie_chart_binary.png')
        bin_data.to_csv('./datasets/bin_data.csv')
        return bin_data

    def multi_class_classification():
        # Multi-class classification
        multi_data = data.copy()
        le2 = LabelEncoder()
        multi_data['intrusion'] = le2.fit_transform(multi_data['label'])
        multi_data = pd.get_dummies(multi_data, columns=['label'], prefix="", prefix_sep="")
        multi_data['label'] = le2.inverse_transform(multi_data['intrusion'])
        np.save("./labels/le2_classes.npy", le2.classes_, allow_pickle=True)
        if show_plot:
            plot_distribution(multi_data, 'label', 'Pie chart distribution of multi-class labels',
                              './plots/Pie_chart_multi.png')
        multi_data.to_csv('./datasets/multi_data.csv')
        return multi_data, le2

    if mode == ClassifierMode.BINARY:
        return binary_classification()
    elif mode == ClassifierMode.MULTI_CLASS:
        return multi_class_classification()
    else:
        raise ValueError("Mode should be either 'binary' or 'multi_class'.")


def pre_process(col_names, training_data_filepath, show_plot='false', mode='binary'):
    data = pd.read_csv(training_data_filepath, header=None, names=col_names)
    data.drop(['difficulty_level'], axis=1, inplace=True)

    change_labels(data)

    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = ['protocol_type', 'service', 'flag']

    return data, numeric_cols, categorical_cols

