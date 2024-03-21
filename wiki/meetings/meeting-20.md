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
