## Installation

## nightly version (master)
```bash
pip install --upgrade git+https://gitee.com/zhuozheng/ever.git
```

## stable version (1.9.1)
```bash
pip install git+https://gitee.com/zhuozheng/ever.git@1.9.1
```

## reinstall ever
```bash
pip install --upgrade --no-deps --force-reinstall git+https://gitee.com/zhuozheng/ever.git
```

## Getting Started
[basic usage](https://gitee.com/zhuozheng/ever/tree/master/docs/USAGE.md)



## Highlights
#### AutoPatch Module
Make your deep learning model process big images on the fly.
Please see [ever.magic.bigimage.autopatch](https://gitee.com/zhuozheng/ever/tree/master/ever/magic/bigimage) for more details.

#### infer tool
```python
import ever as er
er.registry.register_modules()

# only build model
model = er.infer_tool.build_from_file('xxx')

# build model and load its parameters
model = er.infer_tool.build_and_load_from_file('xxxx', 'xxx.pth')
```

#### search learning rate

```python
import ever as er
# configure learning rate as follows
learning_rate=dict(
        type='search',
        params=dict(
            init_lr=1e-6, final_lr=1e-2, max_iters=1000
        )
)
# add a callback
def register_hook(launcher):
    hook = er.util.lr_search.PlotLearningRateAndLoss(1,'./find_lr.png')
    launcher.logger.register_train_log_hook(hook)

trainer = ...
trainer.run(after_construct_launcher_callbacks=[register_hook])
```


## License
Copyright 2020 - 2023 Zhuo Zheng. All Rights Reserved