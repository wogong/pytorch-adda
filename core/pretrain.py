"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

from utils import make_variable, save_model
from .test import eval

def train_src(encoder, classifier, src_data_loader, tgt_data_loader, params):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.pre_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        # set train state for Dropout and BN layers
        encoder.train()
        classifier.train()

        for step, (images, labels) in enumerate(src_data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(src_data_loader),
                              loss.data[0]))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            print ("eval model on source dataset")
            eval(encoder, classifier, src_data_loader)
            print ("eval model on target dataset")
            eval(encoder, classifier, tgt_data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, params.model_root, "{}-source-encoder-{}.pt".format(params.src_dataset, epoch + 1))
            save_model(classifier, params.model_root, "{}-source-classifier-{}.pt".format(params.src_dataset, epoch + 1))

    # # save final model
    save_model(encoder, params.model_root, params.src_dataset+"-source-encoder-final.pt")
    save_model(classifier, params.model_root, params.src_dataset+"-source-classifier-final.pt")

    return encoder, classifier
