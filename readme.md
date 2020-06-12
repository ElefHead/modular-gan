## Modular GAN 

A DCGAN implemented with keras model subclassing.   
Implemented because I wanted to replicate [François Chollet's tweet](https://twitter.com/fchollet/status/1250622989541838848). 
Uses MNIST dataset.

### Usage 
To run training : 

```bash
python -m gan.main --mode=train --save '{"batch_size": 32, "epochs": 50, "dataset": "mnist", "latent_dim": 100, "buffer_size": 60000}'
```

Once model is trained and saved, modify the following for running evaluation :

```bash
python -m gan.main --mode=evaluate --save '{"generator_path": "<model_save_path>", "latent_dim": 100, "num_images": 20}'
```
