import os 
import cv2 
import numpy as np 

from training_data import get_training_data
from util import get_image_paths, load_images, stack_images
from models import Autoencoder, var_to_np

use_cuda = False
if use_cuda is True:
    paddle.device.set_device('gpu:0')
    print('=======> using GPU to train')
else:
    print("=======> using CPU to train")


print("=======> loading data")
images_A = get_image_paths('../data/trump')
images_B = get_image_paths('../data/cage')
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0 
images_A += images_B.mean(exis=(0,1,2)) - images_A.mean(axis=(0,1,2))

model = Autoencoder()
model.train()

print('=======> Try resume from checkpoint')

if os.path.isdir('checkpoint'):
    try:
        checkpoint = paddle.load('../checkpoint/autoencoder.pdmodel')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('=======> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.pdmodel')
else:
    start_epoch = 0
    print('=======> Start from scratch')

optim1 = paddle.optimizer.Adam(parameters=[model.encoder.parameters(),model.decoder_A.parameters()])
optim2 = paddle.optimizer.Adam(parameters=[model.encoder.parameters(),model.decoder_B.parameters()])

criterion = paddle.nn.L1Loss()

epochs = 200
batch_size = 32



for epoch in range(epochs):

    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    warped_A = model(warped_A, 'A')
    warped_B = model(warped_B, 'B')
    

    loss1 = criterion(warped_A,target_A)
    loss2 = criterion(warped_B,target_B)

    loss1.backward()
    loss2.backward()

    if (batch_id+1) % 900 == 0:
        print("epoch: {}, batch_id: {}, loss1 is: {}, loss2 is: {}".format(epoch, batch_id+1, loss1.numpy(), loss2.numpy()))

    optim1.step()
    optim2.step()

    optim1.clear_grad()
    optim2.clear_grad()

    if epoch % args.log_interval == 0:

        test_A_ = target_A[0:14]
        test_B_ = target_B[0:14]
        test_A = var_to_np(target_A[0:14])
        test_B = var_to_np(target_B[0:14])
        print('===> Saving models...')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        paddle.save(state, './checkpoint/autoencoder.t7')

    figure_A = np.stack([
        test_A,
        var_to_np(model(test_A_, 'A')),
        var_to_np(model(test_A_, 'B')),
    ], axis=1)
    figure_B = np.stack([
        test_B,
        var_to_np(model(test_B_, 'B')),
        var_to_np(model(test_B_, 'A')),
    ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.transpose((0, 1, 3, 4, 2))
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = np.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()



