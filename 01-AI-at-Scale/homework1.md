Transfomer inputs src and tgt:
1. Changing the sequence length of source from 1 to 16 did not make much of a difference, but when I tried 128, the training was much slower. It took 47 seconds with 4 gpus.
Also when target sequence length was changed to 128, the training time was 55 seconds. My guess here is, with a larger sequence length, each iteration of the training loop takes longer due to larger matrix size per element in the batch and larger matrices to work with.
2. Reducing the batch size from 2048 to 1024 reduced the training time by half, as the number of iterations remained the same and number of elements to process per iteration was halved.
3. Changing the embedding size of both src and tgt to 2048 resulted in this error - RuntimeError: the feature number of src and tgt must be equal to d_model. I could not figure out how to fix this issue to use a different embedding size

Profiling different variants on 2 nodes, with 1 process per node:
1. enumerate(DataLoader) took 32ms as compared to in-memory data. This definitely seems very inefficient and highlights the I/O bottleneck.
