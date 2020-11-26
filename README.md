```python
transform_batch = transforms.Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

for images in data_iterator:
    images = transform_batch(images)
    output = model(images)
```

## Normalize
Applies the equivalent of [`torchvision.transforms.Normalize`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize) to a batch of images.
> Note: This transform acts out of place by default, i.e., it does not mutate the input tensor.
#### `__init__(mean, std, inplace=False, dtype=torch.float, device='cpu')`
* __mean__ _(sequence)_ – Sequence of means for each channel.
* __std__ _(sequence)_ – Sequence of standard deviations for each channel.
* __inplace__ _(bool,optional)_ – Bool to make this operation in-place.
* __dtype__ _(torch.dtype,optional)_ – The data type of tensors to which the transform will be applied.
* __device__ _(torch.device,optional)_ – The device of tensors to which the transform will be applied.
#### `__call__(tensor)`
* __tensor__ _(Tensor)_ – Tensor of size (N, C, H, W) to be normalized.


&nbsp;


## RandomCrop
Applies the equivalent of [`torchvision.transforms.RandomCrop`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomCrop) to a batch of images. Images are independently transformed.
#### `__init__(size, padding=None, device='cpu')`
* __size__ _(int)_ – Desired output size of the crop.
* __padding__ _(int, optional)_ – Optional padding on each border of the image. Default is None, i.e no padding.
* __device__ _(torch.device,optional)_ – The device of tensors to which the transform will be applied.
#### `__call__(tensor)`
* __tensor__ _(Tensor)_ – Tensor of size (N, C, H, W) to be randomly cropped.


&nbsp;


## RandomHorizontalFlip
Applies the equivalent of [`torchvision.transforms.RandomHorizontalFlip`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomHorizontalFlip) to a batch of images. Images are independently transformed.
> Note: This transform acts out of place by default, i.e., it does not mutate the input tensor.
#### `__init__(p=0.5, inplace=False)`
* __p__ _(float)_ – probability of an image being flipped.
* __inplace__ _(bool,optional)_ – Bool to make this operation in-place.
#### `__call__(tensor)`
* __tensor__ _(Tensor)_ – Tensor of size (N, C, H, W) to be randomly flipped.


&nbsp;


## ToTensor
Applies the equivalent of [`torchvision.transforms.ToTensor`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor) to a batch of images.
#### `__init__()`
#### `__call__(tensor)`
* __tensor__ _(Tensor)_ – Tensor of size (N, C, H, W) to be tensorized.
