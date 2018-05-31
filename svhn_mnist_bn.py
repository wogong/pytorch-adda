"""Main script for ADDA."""

import os

from core import eval_src, eval_tgt, train_src, train_tgt
from models.svhn import Discriminator, Encoder, Classifier
from utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for dataset and data loader
    dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-adda'))

    batch_size = 128
    # image_size = 64

    # params for source dataset
    src_dataset = "svhn"
    src_encoder_restore = os.path.join(model_root, src_dataset + "-source-encoder-final.pt")
    src_classifier_restore = os.path.join(model_root, src_dataset + "-source-classifier-final.pt")
    src_model_trained = True

    # params for target dataset
    tgt_dataset = "mnist"
    tgt_encoder_restore = os.path.join(model_root, tgt_dataset + "-target-encoder-final.pt")
    tgt_model_trained = True

    # params for setting up models
    d_model_restore = os.path.join(model_root , src_dataset + '-' + tgt_dataset + "-critic-final.pt")

    # params for training network
    num_gpu = 1

    num_epochs_pre = 50
    log_step_pre = 20
    eval_step_pre = 10
    save_step_pre = 100

    num_epochs = 500
    log_step = 50 # iter
    eval_step = 10 # epoch
    save_step = 1000
    manual_seed = None

    # params for optimizing models
    pre_learning_rate = 0.001
    tgt_learning_rate = 2e-4
    critic_learning_rate = 2e-4
    beta1 = 0.5
    beta2 = 0.999

params = Config()

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=True)
    src_data_loader_eval = get_data_loader(params.src_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=True)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=False)

    # load models
    src_encoder = init_model(net=Encoder(), restore=params.src_encoder_restore)
    src_classifier = init_model(net=Classifier(), restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=Encoder(), restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(), restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")

    if not (src_encoder.restored and src_classifier.restored and params.src_model_trained):
        src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader, params)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader)
    print("=== Evaluating classifier for target domain ===")
    eval_src(src_encoder, src_classifier, tgt_data_loader)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, src_classifier, tgt_encoder, critic, src_data_loader, tgt_data_loader, params)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader)