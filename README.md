# Hierarchical VQA using Alternating Co-Attention

This repository was created in partial fulfillment for the course [Visual Learning and Recognition (16-824) Fall 2021](https://visual-learning.cs.cmu.edu/), which I took at CMU. 



We use [PyTorch](pytorch.org) to create our models and [TensorBoard](https://www.tensorflow.org/tensorboard/) for visualizations and logging. This repository contains a PyTorch implementation of the following papers:

1. Simple Baseline for Visual Question Answering (Zhou et al, 2015): https://arxiv.org/pdf/1512.02167.pdf

2. Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017):  https://arxiv.org/pdf/1606.00061.pdf

## Results
We should note that we exclude the unknown ground truth answers. If the unknown answers are included, then we observe a 10%.

## Getting Started

### Software setup

We will use the following python libraries for the homework:

1. Python 3.6
2. PyTorch 1.6.0
3. VQA Python API (https://github.com/GT-Vision-Lab/VQA)
4. tensorboardX

A python 3.6 compatible VQA API has been provided in the `external` folder of this repository, courtesy of [16-824: VLR](https://visual-learning.cs.cmu.edu/).

### Downloading dataset

The full dataset itself is quite large (the COCO training images are ~13 GB, for instance), and we need to download (only 'Real Images', not 'Abstract Scenes') the train and validation data from [here](https://visualqa.org/vqa_v1_download.html) .

#### Option 1: using our script
You can run the `get-dataset.sh` script to get all the necessary data.

```
bash get-dataset.sh
```

#### Option 2: Manual Download

1. You'll need to get all three things: the annotations, the questions, and the images for both the training and validation sets.
    1. We're just using the validation set for testing, for simplicity. (In other words, we're not creating a separate set for parameter tuning.)
1. If you're using AWS Volumes we suggest getting a volume with at least 50 GB for caching.
1. We're using VQA v1.0 Open-Ended for easy comparison to the baseline papers.

Download the files to a single folder. After that, `unzip` the files. Now you should have a directory `$DATA` containing the following items.

    mscoco_train2014_annotations.json
    mscoco_val2014_annotations.json
    OpenEnded_mscoco_train2014_questions.json
    OpenEnded_mscoco_val2014_questions.json
    train2014/
    val2014/

### Design decisions for data loading with PyTorch

We implemented customized dataloader, a subclass of `torch.utils.data.Dataset` (https://pytorch.org/docs/stable/data.html), 
to provide easy access to the VQA data, and includes multi-threaded data loading. You will find the details in the file `student_code/vqa_dataset.py`. We use a closed vocabulary for the question embedding. In other words, we choose a set of words that has the **highest frequency** in the training set. All the remaining words will be considered as an 'unknown' class.


Some answers could have multiple words, such as semi circle. Despite the fact that such answers comprise of multiple words, they must be treated as an atomic answer, where the entire phrase is treated as the answer and the corresponding token/ID is generated for the phrase/sentence. Thus we have one ID per sentence.

If the question length is larger than 26, we trim the question and ignore the words after the 26th word. We handle questions of variable length less than 26, by padding our sequence. The output shape of the question tensor is (26 x 5747). We create sentence-level **one-hot encodings** for the answers. 10 answers are provided for each question. We encode each of them and stack them together, giving us an output shape of the answer tensor is (10 x 5217). Again, we make sure to handle the answers not in the answer list by mapping them to an 'unknown' class.

## Pytorch Implementation of Simple Baseline
We describe the implementation of the simple method described in [2]. This serves to validate our Co-Attention model and provides a strong baseline for comparison.

To run the code, you can do:

```python
python -m student_code.main --log_validation --num_epochs 4
```

We implement a frquency based adaptation of the bag of word representation, which is a vector that encodes a sentence based on the frequency of the words that appear in it. We can create a bag of words representation by summing the tensor along the sentence dimension, and converting it into the 0-1 representation by just assigning 1 to values greater than 0. For the bag of words vector, the paper mentioned that the output should be converted to the range of [0, 1]. However, in my implementation I used the frequency vectors and observed competitive performance.

The advantages of a bag of words representation is that it is simple and elegant to use. It is also computationally efficient. While, the disadvantage is that you lose spatial information as your are summing up along the sentence.

We use the output of pretrained GoogLeNet as visual features. An implementation of GoogLeNet is provided in `external/googlenet/googlenet.py`.
    
We provide a brief overview shapes of the internal represenations. This could provide a better intuition of the computation graph. 

1. input shape of pre-trained feature extractor is: `torch.Size([100, 3, 224, 224]) `

1. output shape of pre-trained feature extractor is: `torch.Size([100, 1024])`

1. input shape of word features encoder is: `torch.Size([100, 5747])`

1. output shape of word features encoder is: `torch.Size([100, 1024])`

1. input shape of output or softmax layer is: `torch.Size([100, 2048])` 

1. output shape of output or softmax layer is: `torch.Size([100, 5217])`


## Pytorch Implementation of Alternating Co-Attention

In this task you'll implement [3]. This paper introduces three things not used in the Simple Baseline paper: hierarchical question processing, attention, and 
the use of recurrent layers.

```python
python -m student_code.main --log_validation --num_epochs 4 --model coattention
```

The paper explains attention fairly thoroughly, so we encourage you to, in particular, closely read through section 3.3 of the paper.

To implement the Co-Attention Network you'll need to:

1. Implement the image caching method to allow large batch size.
1. Implement CoattentionExperimentRunner's optimize method.
1. Implement CoattentionNet
    1. Encode the image in a way that maintains some spatial awareness; you may want to skim through [5] to get a sense for why they upscale the images.
    1. Implement the hierarchical language embedding (words, phrases, question)
        1. Hint: All three layers of the hierarchy will still have a sequence length identical to the original sequence length. 
        This is necessary for attention, though it may be unintuitive for the question encoding.
    1. Implement your selected co-attention method
        1. Consider the attention mechanism separately from the hierarchical question embedding. In particular, you may consider writing a separate nn.Module that handles only attention (e.g some "AttentionNet"), that the CoattentionNet can then use.
    1. Attend to each layer of the hierarchy, creating an attended image and question feature for each
    1. Combine these features to predict the final answer

Once again feel free to refer to the [official Torch implementation](https://github.com/jiasenlu/HieCoAttenVQA).

***

The paper uses a batch_size of 300. One way you can make this work is to pre-compute the pretrained network's (e.g ResNet) encodings of your images and cache them, and then load those instead of the full images. This reduces the amount of data you need to pull into memory, and greatly increases the size of batches you can run. This is why we recommended you create a larger AWS Volume, so you have a convenient place to store this cache.

**3.1 Set up transform and image encoder used in the Co-attention paper. Here, we use ResNet18 as the image feature extractor. The transform should be similar to question 2.3, except a different input size. What is the input size used in the Co-Attention paper [3]?** Similar to 2.4, specify the arguments `question_word_to_id_map` and `answer_to_id_map` passed into `VqaDataset`.

**3.2 In `student_code/vqa_dataset.py`, implement your caching and loading logic.** The basic idea is to check whether a cached file for an image exists. If not, load original image from the disk, **apply certain transform if necessary**, extract feature using the encoder, and cache it to the disk; if the cached file exists, directly load the cached feature.

Once you finish this part, immediately run `python -m student_code.run_resnet_encoder` plus any arguments (preferably with batch size 1).

1. It will call the data loader for both training and validation set, and start the caching process.
1. This process will take some time. You can check the progress by counting the number of files in the cache location.
1. Once all the images are pre-cached, the training process will run very fast despite the large batch size we use.
1. In the meanwhile, you can start working on the later questions.

**3.3 Implement Co-attention network in `student_code/coattention_net.py`. The paper proposes two types of co-attention: parallel co-attention and alternating co-attention. In this assignment, please implement alternating co-attention. Use you own words to answer the following questions.**

1. What are the three levels in the hierarchy of question representation? How do you obtain each level of representation?
1. What is attention? How does the co-attention mechanism work? Why do you think it can help with the VQA task?
1. Compared to networks we use in previous assignments, the co-attention network is quite complicated. How do you modularize your code so that it is easy to manage and reuse?

**3.4 In `student_code/coattention_experiment_runner.py`, set up the optimizer and implement the optimization step. The original paper uses RMSProp, but feel free to experiment with other optimizers.**

At this point, you should be able to train you network. You implementation in `student_code/experiment_runner_base.py` for Task 2 should be directly reusable for Task 3.

**3.5 Similar to question 2.10, describe anything special about your implementation in the report. Include your figures of training loss and validation accuracy. Compare the performance of co-attention network to the simple baseline.**


