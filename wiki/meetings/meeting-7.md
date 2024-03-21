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