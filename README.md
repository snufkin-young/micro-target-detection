## An Improved Detection Method for Tiny Target in Remote Sensing Images Based on Yolov7-tiny

### 1. description

​      With the exponential increase in the volume of remote sensing images, the demand for target detection in this field has significantly risen. However, challenges such as long shooting distances, minuscule target sizes, and indistinct features contribute to lower detection accuracy in real-time scenarios. In response, this paper proposes an improved detection method based on the modification of the YOLOv7-tiny network.The proposed enhancements include the substitution of a combined loss function involving Normalized Weighted Intersection over Union (NWD) and Complete Intersection over Union (CIOU) in place of a singular CIOU loss function. Additionally, a lightweight and universal upsampling operator Content-Aware Reassembly of Features (CARAFE), replaces the original bilinear interpolation upsampling algorithm. Furthermore, a spatial pyramid structure is added into the small target layer.The algorithm is trained and validated on the remote sensing image dataset AI-TOD. Comparative analysis reveals that the modified algorithm, compared to YOLOv7-tiny, achieves a 6.7% improvement in mAP0.5 while maintaining detection speed. Moreover, the mAP0.5:0.95 demonstrates a 2.1% increase. The effectiveness of the proposed modifications is further validated on the SIMD dataset. In summary, the network structure model presented in this paper maintains a speed advantage while enhancing detection metrics for minuscule targets, showcasing strong generalization capabilities.

### 2. code

## Training

dataset:

https://github.com/jwwangchn/AI-TOD

Single GPU training

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco128_A_tod.yaml --img 640 640 --cfg cfg/training/modify/yolov7-tiny-carafe.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco128_A_tod.yaml --img 1280 1280 --cfg cfg/training/modify/yolov7-tiny-carafe.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

Multiple GPU training

``` shell
# train p5 models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

## Testing

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

## Inference

## Inference

On video:

``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:

``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```
