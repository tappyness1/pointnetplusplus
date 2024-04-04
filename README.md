# PointNet++

## What this is about
A follow up to the PointNet implementation, this implementation is for PointNet++. This considered a ++ because firstly they want to consider multiple scales using sampling and grouping method. 

What this really means is at multiple magnifications, they will group points together by different radii using sampled points and then some kind of nearest neighbour algorithm before doing feature extraction through the MLP. The idea is that the different radii can act as ways to magnify up and down the points to find common features among the closely grouped ones. 

After various feature extractions from these magnifications, they then concatenate these features to the original points and then make a final MLP to deteremine the points/scene class. 

Also the interpolation was quite an interesting idea even though implementing it was a bit of a headache. 

Not doing training because not intending to. 

## What has been done 

1. Set up the Architecture

## What else needs to be done

## How to run (won't work)

Make sure you change the directory of your data. I used PSNet (not yet implemented yet)

```
python -m src.main
```

## Useful Sources

1. [Paper itself](https://arxiv.org/abs/1706.02413) - read very carefully because the network explanation is strewn all over the paper, similar to their PointNet paper. 
1. [Tensorflow Implementation](https://github.com/charlesq34/pointnet2) - but since this is TF, not followed much.
1. [Sample PyTorch implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master) - a good reference to make sure to have not missed anything. The interpolating function was hard to follow. The best way is to do a double for loop then refactor to remove the loop using some kind of broadcasting in torch based on the number of batches you have. 
1. [Some explanation on PointNet](https://youtu.be/FAqN0KK_2kg?si=ufYgjvy4FSlD3Yq-) - a good place to start to know what is going on in there. 