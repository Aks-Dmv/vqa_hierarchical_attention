from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch
from torchvision import transforms


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 2.3 TODO: set up transform

        transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                       ])

        ############

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   question_word_to_id_map= None,
                                   answer_to_id_map= None,
                                   ############
                                   )
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 )

        model = SimpleBaselineNet()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        ############ 2.5 TODO: set up optimizer
        self.optimizer = torch.optim.SGD([{'params':model.word_feature.parameters(),'lr':0.8}, 
                                          {'params':model.output_layer.parameters(),'lr':0.01} ])

        ############


    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        self.optimizer.zero_grad()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        loss = self.loss_fn(predicted_answers, true_answer_ids)
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self._model.parameters(), 20)
        with torch.no_grad():
            self._model.output_layer.weight.clamp_(min= -20, max=20)
            self._model.word_feature.weight.clamp_(min= -1500, max=1500)
        
        self.optimizer.step()
        return loss
        ############

