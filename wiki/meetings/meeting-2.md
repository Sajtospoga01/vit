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