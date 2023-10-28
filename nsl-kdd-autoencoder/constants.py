# Constants

COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

TRAIN_FILEPATH = 'NSL-KDD/KDDTrain+.txt'

ENCODING_DIM = 50
EPOCHS_AUTOENCODER = 100
EPOCHS_CLASSIFIER = 300
BATCH_SIZE = 500
CLASSIFIER_SPLIT = 0.2
MODEL_PATH = './models/'
WEIGHTS_PATH = './weights/'
PLOTS_PATH = './plots/'
BINARY_AE_MODEL = 'ae_binary.json'
BINARY_AE_WEIGHTS = 'ae_binary.h5'
CLASSIFIER_MODEL = 'ae_classifier_binary.json'
CLASSIFIER_WEIGHTS = 'ae_classifier_binary.h5'

AUTOENCODER_EPOCHS = 100
AUTOENCODER_BATCH_SIZE = 500
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 700
N_CLASSES = 5
