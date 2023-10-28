import argparse
import binary
import multi_class
import pre_process
from enums import ClassifierMode
from constants import COL_NAMES, TRAIN_FILEPATH

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="Train autoencoders on NSL KDD dataset")
    parser.add_argument("--mode", type=str, choices=['binary', 'multi'], required=True,
                        help="Choose the mode: 'binary' or 'multi' for binary or multi-class autoencoder respectively")
    args = parser.parse_args()

    show_plot = True

    data, numeric_cols, categorical_cols = pre_process.pre_process(COL_NAMES, TRAIN_FILEPATH)

    if args.mode == 'binary':
        bin_data = pre_process.classification_processing(
            data, numeric_cols, categorical_cols, show_plot, ClassifierMode.BINARY)
        binary.binary_auto_encoder(bin_data)
    elif args.mode == 'multi':
        multi_data, label_encoding = pre_process.classification_processing(
            data, numeric_cols, categorical_cols, show_plot, ClassifierMode.MULTI_CLASS)
        multi_class.multi_class_auto_encoder(multi_data, label_encoding)
