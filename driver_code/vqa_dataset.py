from torch.utils.data import Dataset
from external.vqa.vqa import VQA

import string
from collections import Counter
from PIL import Image
from torchvision import transforms
import torch
import os


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            print(len(self._vqa.getQuesIds()))
            all_sentences = [self._vqa.qqa[q_id]['question'] for q_id in self._vqa.getQuesIds()]
            word_list = self._create_word_list(all_sentences)
            self.question_word_to_id_map = self._create_id_map(word_list, question_word_list_length)
            ############
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            all_answers = []
            for q_id in self._vqa.getQuesIds():
                # proxy_sentence = [indiv_ans['answer'] for indiv_ans in self._vqa.qa[q_id]['answers']]
                proxy_sentence = ["".join(indiv_ans['answer'].split()) for indiv_ans in self._vqa.qa[q_id]['answers']]
                all_answers.append(" ".join(proxy_sentence))
            all_answers = self._create_word_list(all_answers)
            # print(all_answers[:10])
            self.answer_to_id_map = self._create_id_map(all_answers, answer_list_length)
            ############
        else:
            self.answer_to_id_map = answer_to_id_map
    
    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        if type(sentences) == list:
            sentences = " ".join(sentences)
        
        sentences = sentences.lower()
        sentences = sentences.translate(str.maketrans('', '', string.punctuation))
        return sentences.split(" ")


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO
        cntr_obj = Counter(word_list)
        most_cmn_wrds =  cntr_obj.most_common(max_list_length)
        
        dict_to_return = {}
        
        rank_of_wrd = 0
        for i in most_cmn_wrds:
            # print("i is ",i[0])
            dict_to_return[i[0]] = rank_of_wrd
            rank_of_wrd += 1
        
        ############
        return dict_to_return


    def __len__(self):
        ############ 1.8 TODO
        return len(self._vqa.getQuesIds())
        ############

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        img_file_path = str(self._vqa.questions['questions'][idx]['image_id'])
        zeros_to_append = ['0']*(12 - len(img_file_path))
        zeros_to_append = "".join(zeros_to_append)
        img_file_code = zeros_to_append + img_file_path
        img_file_path = self._image_dir + '/' + self._image_filename_pattern.format(img_file_code)
        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here
            
            cached_filename = self._cache_location  + '/' + (self._image_filename_pattern[:-3]).format(img_file_code) + 'pt'
            # print("outside ", cached_filename)
            if os.path.exists(cached_filename):
                image = torch.load(cached_filename)
                # print(image.shape)
            else:
                image = Image.open(img_file_path)
                image = image.convert("RGB")
                if self._transform is None:
                    self._transform = transforms.Compose([transforms.ToTensor()])
                image = self._transform(image)
                image = torch.unsqueeze(image, 0)
                img_feature_to_cache = self._pre_encoder(image).detach().cpu()
                torch.save(img_feature_to_cache, cached_filename)
                image = img_feature_to_cache
                # print(image.shape)
            ############
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            image = Image.open(img_file_path)
            image = image.convert("RGB")
            if self._transform is None:
                self._transform = transforms.Compose([transforms.ToTensor()])
            image = self._transform(image)
            ############

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        q_enc = torch.zeros(self._max_question_length, self.question_word_list_length)
        a_enc = torch.zeros(10, self.answer_list_length)
        
        
        question_words = self._create_word_list(self._vqa.questions['questions'][idx]['question'])
        for i, words in enumerate(question_words):
            if i == self._max_question_length:
                break
            if words not in self.question_word_to_id_map.keys():
                embed_idx = self.unknown_question_word_index
            else:
                embed_idx = self.question_word_to_id_map[words]
            q_enc[i][embed_idx] = 1
        
        i=0
        q_id_for_ans = self._vqa.questions['questions'][idx]['question_id']
        answer_list_to_return = ["".join(indiv_ans['answer'].split()) for indiv_ans in self._vqa.qa[q_id_for_ans]['answers']]
        answer_words = self._create_word_list(answer_list_to_return)
        # print("answer words", answer_words[:10])
        # print("list keys ",list(self.answer_to_id_map.keys())[:10])
        for indiv_ans in answer_words:
            # print(indiv_ans," individual answer")
            if indiv_ans not in list(self.answer_to_id_map.keys()):
                embed_idx = self.unknown_answer_index
            else:
                embed_idx = self.answer_to_id_map[indiv_ans]
            a_enc[i][embed_idx] = 1
            i+=1
        ############
        return {'image':image, 'question':q_enc, 'answer':a_enc, 'img_file_path': img_file_path}

