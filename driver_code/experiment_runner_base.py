from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=4, log_validation=True):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 200  # Steps
        self._test_freq = 200  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        self.writer = SummaryWriter()
        self.curr_epoch = 0
        self._batch_sz = batch_size
        self._val_batches = int(1000/batch_size)
        self._train_dataset = train_dataset
        
        self.invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                             std = [ 1., 1., 1. ]),
                                       ])
        self.i_transform = transforms.Compose([transforms.ToTensor()])
        

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        ############ 2.8 TODO
        # Should return your validation accuracy
        loss_fn = torch.nn.CrossEntropyLoss()
        self._model.eval()
        total_acc = 0
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if batch_id == self._val_batches:
                break # only taking the first 1000 datapoints
                
            img = batch_data['image'].cuda()
            ques = batch_data['question'].cuda()
            gt_ans = batch_data['answer'].cuda()

            gt_ensemble_ans = torch.sum(gt_ans, dim=1)
            ground_truth_answer = torch.argmax(gt_ensemble_ans, dim=1) # TODO
            with torch.no_grad():
                predicted_answer = self._model(img, ques).cuda() # TODO
                predicted_answer = torch.argmax(predicted_answer, dim=1)
                # print(torch.cat((predicted_answer.reshape(-1,1), ground_truth_answer.reshape(-1,1)), 1).tolist())
                correct_ans = torch.eq(predicted_answer, ground_truth_answer)
                correct_ans = correct_ans * (ground_truth_answer != 5216)
                # print(correct_ans.tolist())
                total_acc += correct_ans.sum()
                # print(total_acc)
            
            

            ############
            # print(self._log_validation)
            # print(list(self._train_dataset.question_word_to_id_map.keys())[0])
            if batch_id==0 and self._log_validation:
                ############ 2.9 TODO
                # you probably want to plot something here
                rand_idx = np.random.choice(batch_data['question'].shape[0]-1,1)
                test_question = batch_data['question'][rand_idx]
                q_sentence = ''
                not_seen_padding = True
                for word_vec in test_question.squeeze():
                    wrd_idx = torch.argmax(word_vec).detach().cpu().numpy().item()
                    if not_seen_padding:
                        if wrd_idx==0:
                            not_seen_padding = False
                    else:
                        if wrd_idx==0:
                            q_sentence = q_sentence[:-4]
                            break
                        else:
                            not_seen_padding = True
                    if wrd_idx==5746:
                        continue
                    else:
                        q_sentence = q_sentence + " " + list(self._train_dataset.question_word_to_id_map.keys())[wrd_idx]
                
                # print(self.curr_epoch, q_sentence)
                # print(batch_data['question'].shape, batch_data['image'].shape)
                self.writer.add_text('val{}/question: '.format(self.curr_epoch), q_sentence, self.curr_epoch)
                
                img = batch_data['image'][rand_idx].squeeze()
                if len(img.shape)<3:
                    # print(rand_idx, len(batch_data['img_file_path']), batch_data['img_file_path'][0])
                    img_path = batch_data['img_file_path'][rand_idx[0]]
                    image = Image.open(img_path)
                    image = image.convert("RGB")
                    img = self.i_transform(image).cpu().numpy()
                else:
                    img = self.invTrans(img).detach().cpu().numpy()
                self.writer.add_image('images/image{}: '.format(self.curr_epoch), img, self.curr_epoch)

                gt_answer = torch.argmax(gt_ans[rand_idx].squeeze()).detach().cpu().numpy().item()
                # print("gt ans ",gt_answer)
                # print("len of answer keys ",len(list(self._train_dataset.answer_to_id_map.keys())))
                if gt_answer==len(list(self._train_dataset.answer_to_id_map.keys())):
                    self.writer.add_text('val{}/gt_answer: '.format(self.curr_epoch), 'UNKNOWN', self.curr_epoch)
                else:
                    final_answer = list(self._train_dataset.answer_to_id_map.keys())[gt_answer]
                    self.writer.add_text('val{}/gt_answer: '.format(self.curr_epoch), final_answer, self.curr_epoch)
                    
                
                pred_answer = torch.argmax(predicted_answer[rand_idx].squeeze()).detach().cpu().numpy().item()
                # print("pred ",list(self._train_dataset.answer_to_id_map.keys())[pred_answer])
                if pred_answer==len(list(self._train_dataset.answer_to_id_map.keys())):
                    self.writer.add_text('val{}/pred_answer: '.format(self.curr_epoch), 'UNKNOWN', self.curr_epoch)
                else:
                    final_answer = list(self._train_dataset.answer_to_id_map.keys())[pred_answer]
                    self.writer.add_text('val{}/pred_answer: '.format(self.curr_epoch), final_answer, self.curr_epoch)
                
                
                ############
        self._model.train()
        return total_acc/(self._val_batches * self._batch_sz)#(len(self._val_dataset_loader) * self._batch_sz)
        

    def train(self):

        for epoch in range(self._num_epochs):
            self.curr_epoch = epoch

            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                img = batch_data['image'].cuda()
                ques = batch_data['question'].cuda()
                gt_ans = batch_data['answer'].cuda()
                
                gt_ensemble_ans = torch.sum(gt_ans, dim=1)
                # print(self._batch_sz, "batch size", img.shape)
                predicted_answer = self._model(img, ques).cuda() # TODO
                ground_truth_answer = torch.argmax(gt_ensemble_ans, dim=1) # TODO

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('train/loss', loss.item(), current_step)
                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('val/accuracy', val_accuracy, current_step)
                    ############

