# Unsupervised deeplearning method for IR and RGB videos registration
The goal of this project is to align RGB and Infrared videos. For this, we use an unsupervised deep learning method. We use two different networks but there are very similar. The first network do only linear transformation on the frames and the second one can do not linear transformation (it can deform shapes). If you are intersted about them, check the [report](final_report.pdf) or directly their source code [model.py](model.py).

## Dataset
The dataset is formed of a 4-tuple (RGB, IR, mask RGB and mask IR) of video frames. Here you can see an example how load the dataset.
```python
dataset = np.load(path+'dataset.npz')

imgs_rgb = dataset['rgb']
imgs_ir = dataset['ir']
imgs_mask_rgb = dataset['mask_rgb']
imgs_mask_ir = dataset['mask_ir']
```
[dataset \(2.4 GB\)](https://drive.google.com/file/d/1dRi3L7eXd7uTt6tPTKGhrIMDgOc9WTHu/view?usp=sharing)
