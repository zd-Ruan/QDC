# Third-party notices

QDC is distributed under the Apache License 2.0 in `LICENSE`. This release adapts the supplied author research implementation and retains its upstream lineage:

| Project | Source | License | Retained contribution |
| --- | --- | --- | --- |
| QT | https://github.com/charleshsc/QT | Apache-2.0 | Offline trajectory pipeline, twin critics, Q regularization, target updates, candidate inference |
| Decision ConvFormer | https://github.com/beanie00/Decision-ConvFormer | MIT | Modality-specific causal local filters and convolutional policy structure |
| Decision Transformer | https://github.com/kzl/decision-transformer | MIT | Trajectory tokenization and sampling lineage inherited through the research implementation |
| Diffusion-QL | https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL | Apache-2.0 | Actor-critic training lineage acknowledged by QT |

Full MIT copyright and permission notices are included in `third_party/`. The Apache-2.0 license text is included in the root `LICENSE`.

The supplied legacy Transformer / GPT-2 files carried OpenAI, Hugging Face, and NVIDIA notices. Those vendored files are excluded from this source release; the released convolutional backbone does not vendor GPT-2 or require Transformers. The local backup of the historical sources retains their original headers.

PyTorch, NumPy, and optional D4RL / Gym / MuJoCo dependencies remain separately licensed projects. This release does not redistribute benchmark datasets, simulator binaries, or historical checkpoint files.
