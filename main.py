"""Main script for ADDA."""

import os
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for dataset and data loader
    home = os.path.expanduser("~/")
    data_root = home + "Dataset"

    dataset_rootm = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN'))

    dataset_mean_value = 0.5
    dataset_std_value = 0.5
    dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
    dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
    batch_size = 128
    # image_size = 64

    # params for source dataset
    src_dataset = "MNIST"
    src_encoder_restore = home + "snapshots/ADDA-source-encoder-final.pt"
    src_classifier_restore = home + "snapshots/ADDA-source-classifier-final.pt"
    src_model_trained = True

    # params for target dataset
    tgt_dataset = "USPS"
    tgt_encoder_restore = home + "snapshots/ADDA-target-encoder-final.pt"
    tgt_model_trained = True

    # params for setting up models
    model_root = home + "snapshots"
    d_model_restore = home + "snapshots/ADDA-critic-final.pt"

    # params for training network
    num_gpu = 2

    num_epochs_pre = 100
    log_step_pre = 20
    eval_step_pre = 20
    save_step_pre = 100

    num_epochs = 100
    log_step = 10
    save_step = 100
    manual_seed = None

    # params for optimizing models
    d_learning_rate = 2e-4
    c_learning_rate = 2e-4
    beta1 = 0.5
    beta2 = 0.999

params = Config()

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
