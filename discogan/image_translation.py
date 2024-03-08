import os
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataset import *
from model import *
import scipy
import imageio
from progressbar import ETA, Bar, Percentage, ProgressBar

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default=os.path.abspath('./results/'), help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default=os.path.abspath('./models/'), help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=10000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--constraint', type=str, default=None, help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
parser.add_argument('--constraint_type', type=str, default=None, help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

parser.add_argument("--checkpoint_dir", type=str, default=None, help="If specified, continue training using the specified checkpoint directory.  an exception will be raised if this is used without also specifying --checkpoint_iters")
parser.add_argument("--checkpoint_iters", type=int, default=None, help="The number of iterations of training done when the checkpoint model was saved.  this must be specified when --checkpoint_dir is used.")

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    # celebA / edges2shoes / edges2handbags / ...
    if args.task_name == 'summer2winter_yosemite':
        data_A, data_B = get_summer2winter_yosemite_files( test=False )
        test_A, test_B = get_summer2winter_yosemite_files( test=True )
    else:
        print("bad taskname:", args.task_name)
        exit()

    return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion, cuda):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        
        labels = torch.ones(l2.size())
        
        if cuda:
            labels = labels.cuda()
        
        loss = criterion( l2, labels )
        losses += loss

    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = torch.ones(dis_real.size()[0], 1, 1, 1)
    labels_dis_fake = torch.zeros(dis_fake.size()[0], 1, 1, 1)
    labels_gen = torch.ones(dis_fake.size()[0], 1, 1, 1)

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss


def main():

    global args
    args = parser.parse_args()
    
    checkpoint_save_ext = ".pt"
    checkpoint_save_files = ["model_gen_A" + checkpoint_save_ext, "model_gen_B" + checkpoint_save_ext, "model_dis_A" + checkpoint_save_ext, "model_dis_B" + checkpoint_save_ext]

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    result_path = os.path.join( args.result_path, args.task_name )
    result_path = os.path.join( result_path, args.model_arch )
    result_path = os.path.abspath(result_path)

    model_path = os.path.join( args.model_path, args.task_name )
    model_path = os.path.join( model_path, args.model_arch )
    model_path = os.path.abspath(model_path)

    data_style_A, data_style_B, test_style_A, test_style_B = get_data()
    
    test_A = read_images( test_style_A, args.image_size )
    test_B = read_images( test_style_B, args.image_size )

    test_A = torch.FloatTensor( test_A )
    test_B = torch.FloatTensor( test_B )

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator_A = Generator()
    generator_B = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    models = [generator_A, generator_B, discriminator_A, discriminator_B]
    
    iters = 0
    
    if args.checkpoint_dir != None:
        # check that checkpoint_iters is also defined
        if args.checkpoint_iters == None:
            print("--checkpoint_dir was specified without --checkpoint_iters")
            exit()
        else:
            iters = args.checkpoint_iters
            
            checkpoint_dir = os.path.join(model_path, os.path.abspath(args.checkpoint_dir))
            
            for i in range(len(checkpoint_save_files)):
                p = checkpoint_save_files[i]
                model = models[i]
                
                file = os.path.join(checkpoint_dir, p)
                
                model_state_dict = torch.load(file, map_location='cpu')
                
                model.load_state_dict(model_state_dict)
    
    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size )

    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    gen_loss_total = []
    dis_loss_total = []

    for epoch in range(epoch_size):
        data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B )

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        for i in range(n_batches):

            pbar.update(i)

            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
            B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]

            A = read_images( A_path, args.image_size )
            B = read_images( B_path, args.image_size )

            A = torch.FloatTensor( A )
            B = torch.FloatTensor( B )

            if cuda:
                A = A.cuda()
                B = B.cuda()

            AB = generator_B(A)
            BA = generator_A(B)

            ABA = generator_A(AB)
            BAB = generator_B(BA)

            # Reconstruction Loss
            recon_loss_A = recon_criterion( ABA, A )
            recon_loss_B = recon_criterion( BAB, B )

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A( A )
            A_dis_fake, A_feats_fake = discriminator_A( BA )

            dis_loss_A, gen_loss_A = get_gan_loss( A_dis_real, A_dis_fake, gan_criterion, cuda )
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion, cuda)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B( B )
            B_dis_fake, B_feats_fake = discriminator_B( AB )

            dis_loss_B, gen_loss_B = get_gan_loss( B_dis_real, B_dis_fake, gan_criterion, cuda )
            fm_loss_B = get_fm_loss( B_feats_real, B_feats_fake, feat_criterion, cuda)

            # Total Loss

            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate

            gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
            gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate

            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print("---------------------")
                print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
                print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
                print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
                print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))

            if iters % args.image_save_interval == 0:
                AB = generator_B( test_A )
                BA = generator_A( test_B )
                ABA = generator_A( AB )
                BAB = generator_B( BA )

                n_testset = min( test_A.size()[0], test_B.size()[0] )

                save_dir = "iter_%d" % iters
                subdir_path = os.path.join( result_path, save_dir )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range( n_testset ):
                    A_val = test_A[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
                    B_val = test_B[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
                    BA_val = BA[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    AB_val = AB[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.

                    filename_prefix = os.path.join (subdir_path, str(im_idx))
                    imageio.imwrite( filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:,:,::-1])
                    imageio.imwrite( filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:,:,::-1])
                    imageio.imwrite( filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:,:,::-1])
                    imageio.imwrite( filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:,:,::-1])
                    imageio.imwrite( filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:,:,::-1])
                    imageio.imwrite( filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:,:,::-1])

            if iters % args.model_save_interval == 0:
                save_dir = "iter_%d" % iters
                subdir_path = os.path.join(model_path, save_dir)
                pt_ext = ".pt"
                
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
        
                torch.save( generator_A.state_dict(), os.path.join(subdir_path, 'model_gen_A' + pt_ext))
                torch.save( generator_B.state_dict(), os.path.join(subdir_path, 'model_gen_B' + pt_ext))
                torch.save( discriminator_A.state_dict(), os.path.join(subdir_path, 'model_dis_A' + pt_ext))
                torch.save( discriminator_B.state_dict(), os.path.join(subdir_path, 'model_dis_B' + pt_ext))

            iters += 1

if __name__=="__main__":
    main()