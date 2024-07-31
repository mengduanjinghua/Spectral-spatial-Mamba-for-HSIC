import argparse

def load_args():
    parser = argparse.ArgumentParser()

    # Pre training
    parser.add_argument('--dataset', type=str, default='Pavia')
    parser.add_argument('--train_num', type=int, default=20)
    parser.add_argument('-windowsize', type=int, default=27)
    parser.add_argument('--type', type=str, default='none')

    # training parameter
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=190)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--lr_decay', type=float, default=0.5)

    # model parameter
    parser.add_argument('--model_id', type=int, default=2,
                        help='0: 1D, 1: 2D, 2: SS')
    
    parser.add_argument('--spe_windowsize', type=int, default=3)
    parser.add_argument('--spa_patch_size', type=int, default=3)
    parser.add_argument('--spe_patch_size', type=int, default=2)
    parser.add_argument('--hid_chans', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)

    parser.add_argument('--use_bi', default=True, type=lambda x: (str(x).lower() == 'true'), help='use bidirection or not' )  
    parser.add_argument('--use_global', default=True, type=bool,
                        help='use token meaning or not') 
    parser.add_argument('--use_cls', default=True, type=bool,
                        help='use class tken or not') 
    parser.add_argument('--use_fu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use center augmentation fusion or not') 
      
    args = parser.parse_args()
    return args

