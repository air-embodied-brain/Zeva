# Third-party notices

Parts of the three-stream temporal encoder and action-prior implementation were
adapted from `iLearn-Lab/ICML26-BehaviorVLA`, commit
`0dbabc7e79791a325c4e76acde0ddfd7a18e8326`, licensed under Apache-2.0.

This attribution does not change ZeVA's public model terminology or its own
RoboTwin training, BIT effect prediction, task-language retrieval, recurrent
H15 execution, or cross-attempt PIM contracts.

The Zeva-Ego action encoder includes dependency-free PyTorch adaptations of
the encoder, decoder, and vector-quantization blocks from
`OpenDriveLab/UniVLA`, licensed under Apache-2.0. The adapted implementation
adds variable-length action masks and uses PyTorch scaled-dot-product
attention without copying UniVLA's dataset or training infrastructure.
