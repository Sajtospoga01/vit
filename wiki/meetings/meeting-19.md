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