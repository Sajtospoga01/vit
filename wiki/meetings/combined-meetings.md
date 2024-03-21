# Meeting

Subject: individual project
Lecture number: 1

Meeting time: 60 mins

Present: 

- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 

- Project goals
    - Come up with possible goals - choose top 3
- Recommended research to look into
    - BeIT-3
    - Vision Transformers
    - SA-Net

# Meeting

Subject: individual project
Lecture number: <number>

Meeting time: <time> mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Progress:
Project progress:

- Vision transformers:
    - papers:
        - [SwiftTF encoder](https://www.mdpi.com/2072-4292/13/24/5100) - encoder used for remote sensing image data encoding using transformer models
        - [Hyperspectral segmentation with masked transformers](https://github.com/HSG-AIML/MaskedSST)
        - [Hyperspectral segmentation with transformers](https://github.com/zilongzhong/SSTN)
        
    - findings:
        - Many available on hugging face, for finetuning, so starting would be quite easy.
        - However, it requires for a huge migration to torch as the primary underlying framework
- Project Goals:
    - Comparing vision transformer performance, to conv network performance ← I really like this one and should be executable in the time.
        - paper
            - [Evaluation metrics on segmentation model performance](https://arxiv.org/pdf/1908.05005.pdf)
        - findings
            - As long as the models are ready benchmarking should be very straight forward
            - It requires the model setup though, and an understanding in torch
    - Migrated pretrained model's vs Fully trained models
        - paper
            - [Exploring the change of training domains](https://arxiv.org/pdf/2204.03144v1.pdf)
        - findings
            - Domain change can offer significant performance uplift for training on hyperspectral data
    - Classification performance analysis of 2D conv vs 3D conv in segmenting hyperspectral image data
        - paper
            - [Evaluation metrics on segmentation model performance](https://arxiv.org/pdf/1908.05005.pdf)
            - 
        - findings
            - some findings
    - Hybrid architecture from combining ViT and CNN
        - paper
            - [Hybrid Conv-ViT Network for Hyperspectral Image Classification | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10155242)
        - findings
            - combining vision transformer global classification and CNN local feature extraction could provide increased classification performance
    - Using Masked Vision Transformers for training on unspecified spectral bands
        - paper
            - Did not find paper on it
        - finding
            - seems very straight forward using the capabilities of masked vision transformers.
            - Could offer looking into multiple ways of tokenizing image data (like per band)
            - Could be a facilitator to the project below
    - (Similar to the one above) Using vision transformers for training on multiple datasets
        - paper
            - found no paper
        - finding
            - likely not a common thing to do (understandably). → very hard project would love to do it but not this year possibly.
- gpt collected time estimate, ideas
    
    With a 150-hour time constraint, it's crucial to pick projects that are not only feasible but also can be executed efficiently. Here's a breakdown:
    
    ### Comparing Vision Transformer to Conv Network Performance
    
    - **Estimated Time**: 40-50 hours
    - **Reason**: Benchmarking is straightforward, but you'll need to spend time migrating to PyTorch and setting up the models.
    
    ### Using Masked Vision Transformers
    
    - **Estimated Time**: 30-40 hours
    - **Reason**: This seems straightforward and could be a good starting point for other projects. You can explore tokenization methods and their impact on performance.
    
    ### Hybrid Architecture from Combining ViT and CNN
    
    - **Estimated Time**: 50-60 hours
    - **Reason**: The architecture design and integration could take a significant amount of time, but the project is interesting and could yield valuable insights.
    
    ### Migrated Pretrained Model's vs Fully Trained Models
    
    - **Estimated Time**: 60-70 hours
    - **Reason**: Preparing both migrated and fully trained models for comparison could be time-consuming.
    
    ### Classification Performance Analysis of 2D conv vs 3D conv
    
    - **Estimated Time**: 60-70 hours
    - **Reason**: Requires a deep understanding of both 2D and 3D convolutional networks, as well as the metrics for evaluating segmentation models.
    
    ### Using Vision Transformers for Training on Multiple Datasets
    
    - **Estimated Time**: 80+ hours
    - **Reason**: This is a complex project that would likely exceed your time limit.
    
    ### Additional Idea: Fine-tuning Pretrained Models on Custom Dataset
    
    - **Estimated Time**: 20-30 hours
    - **Reason**: You can take a pretrained model and fine-tune it on a custom dataset relevant to your field. This could be a quick way to get meaningful results.
    
    Given your 150-hour limit, I'd recommend the following combinations:
    
    1. **Comparing Vision Transformer to Conv Network Performance (40-50 hours)** + **Using Masked Vision Transformers (30-40 hours)**: These two projects are related and could even be integrated. Total estimated time: 70-90 hours, leaving you with time for unexpected issues or further analysis.
    2. **Comparing Vision Transformer to Conv Network Performance (40-50 hours)** + **Fine-tuning Pretrained Models on Custom Dataset (20-30 hours)**: This combination allows you to explore both benchmarking and fine-tuning, giving you a well-rounded experience. Total estimated time: 60-80 hours. 

Questions:

- Problem if diss is public?
- Which project seems decent above?
- are CI/CD pipelines required: such as latex builder etc

Plans:
- TBD

# Meeting

Subject: individual project
Lecture number: 3

Meeting time: 12:15 - 12:45

Present:

- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:

- Model architecture for pretraining
    - papers:
        - Based on the ViT architecture
- Implemented an autoencoder based on ViT and Unet Decoder
- Added visualisation for attention and output

Project blockers:

- S3 Storage read encountered io bottleneck

Planned steps:

- Implementing DINO trainer
- Implementing tensorboard/weights and biases
- Masked encoder implementation

# Meeting

Subject: individual project
Lecture number: 4

Meeting time: 15:15 - 16:00

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:

- Implemented validation loop
- Added prediction artifact upload to WandB
- Fixed error with dataset loader
- Added masking to MAE

Project blockers:

- None

Planned steps:

- Problem in dataset
    - Analysing data what's wrong
    - Analysing relation of bands relative to each other
    - Analysing feature reflectance of data
- Problem in model - investigate model problems

# Meeting

Subject: individual project
Lecture number: 5

Meeting time: 12:15 - 13:00

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:

- Data analytics
    - papers:
        - [https://www.sciencedirect.com/science/article/pii/S1569843222002102](https://www.sciencedirect.com/science/article/pii/S1569843222002102)
        - [Spectral Reflectance - an overview | ScienceDirect Topics](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/spectral-reflectance)
    - findings:
        - Feature maps built on reflectance value
        - Channel correlation of dataset
        - Brightness distribution of dataset

Project blockers:

- Wrong datatype was used in the dataset optimization algorithm

Planned steps:

- Train on fixed data
- Upscale model

# Meeting

Subject: individual project
Lecture number: 6

Meeting time: 0 mins

Meeting got cancelled due to Gerardo being sick


# Meeting

Subject: individual project
Lecture number: 7

Meeting time: 12:15 - 13:00

Present:

- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:

- Implemented full masked autoencoder based on original paper
    - papers:
        - 
- Implemented preprocessing pipepline and added preprocessing function used in the original MAE paper
- Trained on HSI and RGB data with results

Project blockers:

- Due to the data hungriness of the MAE design there is not enough data to train such a model on hyperspectral image data (would require 14 million images and there is clearly not enough data for that)

Planned steps:

- Preloading biases in the model
- Adding adapter into the model for converting HSI image data to RGB
- Maybe mocking HSI image data if possible

# Meeting

Subject: individual project
Lecture number: 8

Got cancelled due to insufficient progress being made that week

# Meeting

Subject: individual project
Lecture number: 9

Meeting time: 12:15 - 13:00

Present:

- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:

- Preloaded biases into autoencoder encoder for much fairer evaluation
    - papers:
        - [2111.06377.pdf (arxiv.org)](https://arxiv.org/pdf/2111.06377.pdf)
- Added HSI adapter based on PCA

Project blockers:

- No time

Planned steps:

- Changing decoder for finetuning on segmentation task and for providing a baseline measurement
- Preparation for self-distillation testing

# Meeting

Subject: individual project
Lecture number: 10

Meeting time: 0 mins 

Gerardo was out of office



# Meeting

Subject: individual project
Lecture number: 11

Meeting time: 12:30-13:00 mins

Present:

- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:
-Refactored files and packages
- Tested previously stated warming up approach
    - Did not work 😢
- Split previous dataset – into small image patches – 300k
    - Sort of promising
- Run training job on cluster – also sort of promising 


Project blockers:

- None

Planned steps:
- Possibility of HSI specific model
- Possibility of scalable encoder
- New settings – training 
- New tokenizer
- Model evaluation

# Meeting

Subject: individual project
Lecture number: 12

Meeting time: 12:30-13:00 mins

Present:

- Gerardo Aragon Camarasa
- Andras Bodrogai

Project progress:
- Performed pretraining on MAE – 100 epochs
- Added checkpoint callbacks, and MSSSIM accuracy for more evaluation



Project blockers:

- None

Planned steps:
- New Architecture
- New Pretraining, Trainer
- New tokenizer

# Meeting

Subject: individual project
Lecture number: 13

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Fixed MMCV pipeline
- Built GPU compatible docker image

Blockers:
- Having a hard time building a GPU enabled image
- Or running the code on the cluster… (no results yet)


Plans:
- Come up with results, then evaluate data on metrics: IoU

# Meeting

Subject: individual project
Lecture number: 14

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Built image
- Evaluated linear classifier

Blocker:
- M2F – something is misconfigured such that the segmentor doesn’t learn anything, likely no time to fix it 🥲


Plans:
- Implementation of HSI specialised encoder
- Evaluation (over same data and same metrics and same decoder)
- New decoder – needs reference/ implementation


# Meeting

Subject: individual project
Lecture number: 15

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Tried decoder networks:
    - Basic CNN decoder - 😢- Maybe was miss configured
    - Transformer decoder – drastically reduced embedding dim 2 blocks 12 heads(still big though)
        - Rebuilds structures very well, Tends to be biased towards majority classes – requires a loss such as focal
        - Large (as of now) – comp: transformer decoder 60 MB to 200 MB linear classifier 1 MB  (considering the decoder is 380 MB it is pretty large)
        
Blockers:
- Some problems with mmcv, but getting there

Plans:
- Adding secondary loss based on patch level loss for taking advantage of the transformer - architecture
- Testing the decoder on the MAE trained network
- HSI specialised Encoder – after next week needs to be done

# Meeting

Subject: individual project
Lecture number: 16

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Updated transformer decoder with evaluation function to check how the training is progressing
- Added patch level loss for segmentor tasks (might will be a hybrid loss when I can run it)
- Started the implementation of Hyper-Spectral network
    - Based on the network above
        - Uses two encoders (spatial - spectral)
        - Uses feature fusion in between encoder block

Blockers:
- There are problems currently with the clusters container store which makes training impossible for the downstream



Plans:
- Testing the decoder on the DINOv2/MAE trained network
- Finishing and training the new spectral network

# Meeting

Subject: individual project
Lecture number: 17

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Adapted MetaArcSSL(DINOv2) trainer to work with the new HSI model
    - Random masking on both the spatial spectral data
    - Loss calculation is performed on both outputs
- Running training on the model (close to finishing – takes a lot of time)

Blocker:
- Model was training for 4 days in straight blocking other development progress


Plans:
- Testing the decoder on the DINOv2/MAE/DINOv2HSI trained network

# Meeting

Subject: individual project
Lecture number: 18

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Experimenting with decoder configurations: Did not result in anything better than before: - possible problems with encoders – very strong class imbalance in the data.
- Run linear classifier again with different parameters


Plans:
- Running linear classifier first on all models for comparison
- Can try other decoders after

# Meeting

Subject: individual project
Lecture number: 19

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Tried multiple classifier configs
- Wrote dissertation

Plans:
- Write dissertation
- Run baseline experiment

# Meeting

Subject: individual project
Lecture number: 20

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Model is queueing at this time
- Was writing more dissertation

Plans:
- Write more dissertation
- Hopefully run baseline experiment and report results

# Meeting

Subject: individual project
Lecture number: 21

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Dissertation writing
- Baseline performance was underwhelming, need to investigate

Plans:
- Investigate higher resolution training

# Meeting

Subject: individual project
Lecture number: 22

Meeting time: 30 mins

Present: 
- Gerardo Aragon Camarasa
- Andras Bodrogai

Discussion: 
- Investigated higher resolution training, without any success
- Writing more disseration

Plans:
- Investigate attention maps
- if time allows it, investigate original codebase