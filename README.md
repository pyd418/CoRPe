# Reproduction process of CoRPe
Please see the paper on https://ieeexplore.ieee.org/document/10433728.

**Here is a tip**: This is the code for CTE process. For the batch_size in Section V, it means the batch size of the decoder. In the training process, it depends on your device. Maybe 8 or 16 is better.
Moreover, in the Equation (12) and (13), the distance of negative sample and positive sample should be inverted: 
$$\mathcal{L}_ o  = \text{max}(0, d(\bf{h}_ \texttt{[MASK]}, \bf{h}_ {obj}) - d(\bf{h}_ \texttt{[MASK]}, \bf{h}_ {obj}^-) + \eta)$$
