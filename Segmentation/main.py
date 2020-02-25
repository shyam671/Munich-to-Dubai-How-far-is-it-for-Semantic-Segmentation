import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union
import gc 
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=400,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=535,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
##################################################change---num-worker################################
parser.add_argument('--workers', type=int, default=0,
                    help='number of data loading workers')
args = parser.parse_args()

def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    f_of_X = f_of_X.float()
    f_of_Y = f_of_Y.float()
    
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    f_of_X = f_of_X.float()
    f_of_Y = f_of_Y.float()
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss/535

def DeepCoral(source, target):
    source = source.float()
    target = target.float()

    #source = source/source.sum(0).expand_as(source) 
    #source[torch.isnan(source)]=0

    #target = target/target.sum(0).expand_as(target) 
    #target[torch.isnan(target)]=0

    source = torch.unsqueeze(source, 1)
    target = torch.unsqueeze(target, 1)

    d = source.data.shape[1]
    xm = torch.mean(source, 1, keepdim=True)
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)  # source covariance
    
    xmt = torch.mean(target, 1, keepdim=True)
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)   # target covariance
    
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))   # frobenius norm between source and target
    return (loss/(4*1024*2048))


def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
      args.backbone, args.dataset, args.exp)
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
####################################change###################################################datasplitin cityscapes
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',train=args.train, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  if args.train:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers)
    max_iter = args.epochs * len(dataset_loader)
    losses = AverageMeter()
    start_epoch = 131

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
      for i, (inputs, target, train_target) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult
        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        outputs = model(inputs)
        loss = criterion(outputs, target)
	#mmd_loss = (linear_mmd2(torch.max(outputs, 1)[1], target))*0.001   
	coral_loss = torch.log(DeepCoral(torch.max(outputs, 1)[1], target))*(0.002)
        print(loss, coral_loss)
        #loss = coral_loss + 0*loss
        loss = coral_loss  + 0*loss
        if np.isnan(loss.item()) or np.isinf(loss.item()):
           loss.data = torch.tensor([1e-8]).cuda()
          #pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
	
        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))
      #if 
      #torch.save({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, model_fname % (epoch + 1))  
      if epoch % 5 == 0:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))

  else:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname % args.epochs)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    model = model.cpu()
    #cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    #cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
    cmap = [128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32]
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in tqdm(range(len(dataset))):
      if i == 40:
         break;
      inputs, target, _ = dataset[i]		
      outputs = model(Variable(inputs).unsqueeze(0))
      _, outputs = torch.max(outputs, 1)
      pred = outputs.data.cpu().numpy().squeeze().astype(np.uint8)	
      mask = target.numpy().astype(np.uint8)
      imname = dataset.masks[i].split('/')[-1]
      mask_pred = Image.fromarray(pred)
      mask_pred.putpalette(cmap)
      #mask_pred.save(os.path.join('data/val', imname))
      print('eval: {0}/{1}'.format(i + 1, len(dataset)))
      
      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)
      del inputs, target, outputs, pred, mask, imname, mask_pred, inter, union
      gc.collect()

    print(inter_meter.sum, union_meter.sum)  
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
  main()
      ###
#      inputs = inputs[:,:-1,:-1]
#      target = target[:-1,:-1]
#      print(inputs.dtype, target.dtype)
#      inputs = torch.reshape(inputs, (1, inputs.size()[0], inputs.size()[1], inputs.size()[2]))
#      inputs = F.upsample_nearest(inputs, scale_factor=(0.5,0.25))
#      target = torch.reshape(target, (1, 1, target.size()[0], target.size()[1]))
#      target = F.upsample_nearest(target, scale_factor=(0.5,0.25))

#      target = target.squeeze()
#      inputs = inputs.squeeze()
      ###
