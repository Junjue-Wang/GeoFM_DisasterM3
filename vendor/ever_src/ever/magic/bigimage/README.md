### Generalized AutoPatch
```python
import tifffile
import torch
import ever as er
from ever.magic.transform import segm

model, _ = er.infer_tool.build_and_load_from_file('xxx','xxx')
model.cuda()
model= er.autopatch(model,
                   config=er.AutoPatchConfig((512, 512), 256, distributed=False,
                                          batch_size=2, progress_bar=True),
                   preprocess_fn=lambda x: x.permute(0, 3, 1, 2).float(),
                   ensemble_transforms=[segm.HorizontalFlip(), segm.VerticalFlip(), segm.Identity()],
                   ensemble_fn=lambda out_list: torch.stack(out_list, dim=0).mean(dim=0),
                   merge_fn=lambda out_list: out_list)
bigimage = tifffile.imread('xxx')
out = model(bigimage)
```
#### AutoPatch for Segmentation
```python
import tifffile
import ever as er
from ever.magic.bigimage.autopatch import preprocess_fn

model, _ = er.infer_tool.build_and_load_from_file('xxx','xxx')
model.cuda()

bigimage = tifffile.imread('xxx',out='memmap')
model = er.AutoPatchSegm(model,
                      # configurable
                      er.AutoPatchConfig(kernel_size=(928, 928),
                                         stride=464,
                                         distributed=False,
                                         batch_size=4,
                                         progress_bar=True),
                      image_size=(7200, 7200),
                      preprocess_fn=preprocess_fn.mean_std_normalization_totensor(mean=(123.675, 116.28, 103.53),
                                                                                  std=(58.395, 57.12, 57.375)))
y_pred = model(bigimage)
```
#### AutoPatchMapping for super big image (>4GB)
```python
import tifffile
import ever as er
from ever.magic.geo import write_georeference
model, _ = er.infer_tool.build_and_load_from_file('xxx','xxx')
model.cuda()

input_path = 'xxxxxx.tif'
bigimage = tifffile.imread(input_path,out='memmap')
output_path = 'xxxx.tif'
model = er.AutoPatchMapping(model, er.AutoPatchConfig(kernel_size=(928, 928),
                                         stride=464,
                                         batch_size=4),output_path)
# result is saved in output_path
model(bigimage)

write_georeference(input_path, output_path)
```


#### AutoPatch for Detection

```python
# todo
```