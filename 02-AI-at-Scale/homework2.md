### Trying out different combinations of model sizes

1. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=4 --n-layers=2
Execution Time: 65 seconds, Lowest Training loss: 11.44, started increasing after iteration 233.
Loss is high possibly because the network is not deep enough.

2. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=4 --n-layers=4
Execution Time: 80.5 seconds, Lowest Training loss: 11.07
This is better than before as we have a slightly deeper network and for the last epoch, the training loss increased.
The training took longer as we have more layers in the network.

3. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=4 --n-layers=8
Execution Time: 110.62 seconds , Lowest Training Loss: 11.21, achieved at around 250 iterations and then started increasing.
Interestingly, more layers caused the network to perform worse. The training loss graph is all over the place. My hunch is we need to play with other hyper parameters such as learning rate to get the training to become smoother and be able to find the global minima.

### How the performance changes with 8-layer model and TP of 1,2,4?

1. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=1 --n-layers=8
Execution Time: 120.77 seconds, Lowest Training Loss: 12.22

2. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=2 --n-layers=8
Execution Time: 120.52 seconds, Lowest Training Loss: 12.04 

3. 1. ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=4 --n-layers=8
Execution Time: 109.69 seconds, Lowest Training Loss: 6.09, achieved very early on.

These experiments confused me quite a bit, as tensor parallelism is supposed to provide performance speedups, but not change the performance. So I am not sure what was going on where, as we achieve different loss each time. It probably has to do with data initialization as well. But this left me confused on why the time did not go down much as we increased tensor parallelism, and also why the performance changed. 

