from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

from torchvision.models import resnet152, resnet18
from torchvision import transforms
import os
import torch

import torch.nn as nn
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 3.1 TODO: set up transform and image encoder
        transform = transforms.Compose([transforms.Resize((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                       ])
        use_resnet18 = True
        if use_resnet18:
            image_encoder = resnet18(pretrained=True)#.cuda()
            # image_encoder.avgpool = Identity()
            image_encoder.fc = Identity()
        else:
            image_encoder = resnet152(pretrained=True)#.cuda()
            image_encoder.fc = Identity()
        
        self.resnet_model = image_encoder
        ############ 

        question_word_list_length = 5746
        answer_list_length = 1000

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   question_word_list_length=question_word_list_length,
                                   answer_list_length=answer_list_length,
                                   cache_location=os.path.join(cache_location, "tmp_train"),
                                   ############ 3.1 TODO: fill in the arguments
                                   question_word_to_id_map= None,
                                   answer_to_id_map= None,
                                   ############
                                   pre_encoder=image_encoder)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 question_word_list_length=question_word_list_length,
                                 answer_list_length=answer_list_length,
                                 cache_location=os.path.join(cache_location, "tmp_val"),
                                 ############ 3.1 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 pre_encoder=image_encoder)

        self._model = CoattentionNet(ans_list_len=len(list(train_dataset.answer_to_id_map.keys())))

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers, log_validation=log_validation)

        ############ 3.4 TODO: set up optimizer
        self.optimizer = torch.optim.Adam(params=self._model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        ############ 

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 3.4 TODO: implement the optimization step
        self.optimizer.zero_grad()
        # print("optimizing preds target ",predicted_answers.shape,true_answer_ids.shape)
        loss = self.loss_fn(predicted_answers, true_answer_ids)
        loss.backward()
        self.optimizer.step()
        return loss
        ############ 
        # raise NotImplementedError()
