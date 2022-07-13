<img src="vit.gif" width="500px"></img>

## ViT-flax

Implementation of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Google's JAX and Flax.

## Acknowledgement:
This repository has been created in collaboration with [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

### Usage:
```python
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

key = PRNGSequence(42)

img = random.normal(next(key), (1, 256, 256, 3))

params = v.init(next(key), img)
logits = v.apply(params, next(key), img)
print(logits.shape)
```