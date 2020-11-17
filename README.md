# HCFS3D
HCFS3D: Hierarchical Coupled Feature Selection Network for 3D Semantic-Instance Segmentation
## Framework
### illustration
The illustration of CFS layer seen as follow

<img src="https://github.com/Biotan/HCFS3D/blob/master/misc/f1.png" width="600"/>

### HCFS3D architecture
The details of our proposed HCFS3D architecture seen as follow

<img src="https://github.com/Biotan/HCFS3D/blob/master/misc/f2.png"/>

## Visulization
### Qualitative results on the S3DIS

<img src="https://github.com/Biotan/HCFS3D/blob/master/misc/f3.png"/>

### Qualitative results on the ShapeNet

<img src="https://github.com/Biotan/HCFS3D/blob/master/misc/f4.png"/>

## Evaluation
### Quantitative results on the S3DIS
The results of our method on S3DIS Area5 and 6-Fold CV respectively.

<img src="https://github.com/Biotan/HCFS3D/blob/master/misc/f6.png" width="400"/>

## Usage
1.clone the HCFS3D repository:
```
cd ~
git clone https://github.com/Biotan/HCFS3D
```
2.download the dataset such as S3DIS and modify the path in train and test file.

3.prepare the environment including tensorflow1.2.0 and python2.7.

4. make the code of pointnet according to [[Pointnet]](https://github.com/charlesq34/pointnet).

5. trainning
```
cd HCFS3D/models
python train.py
```

6. testing and evaluation
```
python test.py && python eval_iou_accuracy.py
```
7. As for Joint CRF, please compile excutable file as [[JSIS3D]](https://github.com/pqhieu/jsis3d)