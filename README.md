# Pytorch Implementation of TIM

This code is a Pytorch implementation of TIM. For implementation, I refer to the other [matlab code](https://github.com/zsdust/TIM-temporal-interpolation-model). Original paper is [Towards a practical lipreading system](https://scholar.google.co.kr/scholar?q=Towards+a+practical+lipreading+system&hl=ko&as_sdt=0&as_vis=1&oi=scholart).

For use of TIM, please follow below steps.
1. Install required packages.
    ```
    pip install einops torch
    ```

2. Copy `tim.py` or clone this repository to your local machine.

## Example

```python
import torch
from tim import FixSequenceTIM

sample_video = torch.randn(18, 3, 224, 224)
tim_transform = FixSequenceTIM(sequence_length=16)
sample_video_transformed = tim_transform(sample_video) # shape: (16, 3, 224, 224)
```