## 1. Input embedding:
1. Original input sentence split into many tokenss (in this case we can see it as words)
2. Map every word into a number (the input id) which represent a position in our vocabulary 
    * Image we have a vocabulary for all the words appears in the training set. 
    * If there are two same word in a sentence, they will have the same input id
3. Map the input ids into a learnable (its parameters change while we train) tensor with size of d_model (in the paper d_model=512)

4. positional encoding: a fix tensor (element wise addition) that is added to the embedding 
    * It add a positional information to the embedding
    * The output is a tensor with the size of d_model
    * we calculate once and use it everytime

## Single-head Self-Attention
a sequence with 6 tokens, which means we can embed it into a tensor size (6,512) 
    --> Q:(6,512), K:(6,512),V:(6,512)
    --> softmax( Q @ K^T / sqrt(d_model) ) @ V 
    -->                              (6,6) @ (6,512)


- Permutation invariant
- it does not require extra parameter other than the input embedding
- If we dont want certain word to interact with each other we can replace it with -inf before softmax. Which we use it in the decoder as a mask


## Multi-head
a sequence with 6 tokens, which means we can embed it into a tensor size (6,512). We set heads=4
    --> Q:(6,512), K:(6,512),V:(6,512)
    --> Q @ W_q, K @ W_k, V @ W_v  (each of the W are with the same size, but with learnable parameters)
    --> split each of them into 4 smaller tensor on the d_model dim
            (6,512) --> (6,128),(6,128),(6,128),(6,128)
    --> head_i = attention(Q @ W_qi, K @ W_ki, V @ W_vi)  size:(6,128)
    --> Head = concate(head_1,head_2,head_3,head4) @ W_o
                                           (6,512) @ (512,512)


## Layer Normalization



## Masked Multi-head Attention
The output at a certain position can only depend on the words on the previous position. They must not see the future words
- We replace all the value of we don't want to see in the Q_i@K_i with -inf, so it would turn 0 after softmax.


## Training
- Two special tokens, start and end








