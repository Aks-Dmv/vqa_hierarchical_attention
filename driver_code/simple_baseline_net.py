import torch.nn as nn
import torch

from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()
        ############ 2.2 TODO
        self.googlenet = googlenet(pretrained=True)
        # Word embedding size is 1024 and img embedding is 1024
        # refer https://github.com/zhoubolei/VQAbaseline/blob/master/opensource_utils.lua
        self.word_feature = nn.Linear(5746 + 1, 1024)
        self.output_layer = nn.Linear(1024 + 1024, 5216 + 1)
        ############

    def forward(self, image, question_encoding):
        ############ 2.2 TODO
        # print(image.shape)
        img_feature = self.googlenet(image)
        
        bow_feature = torch.sum(question_encoding, dim=1)
        word_feature = self.word_feature(bow_feature)
        # print(img_feature.shape, word_feature.shape)
        concat_feature = torch.cat([img_feature, word_feature], dim=-1)
        pred = self.output_layer(concat_feature)
        ############
        return pred

