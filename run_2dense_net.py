import argparse
import tensorflow as tf

"""
you can change the following codes to decide which model to use.
"""

from models.AGDN_model import DenseNet
# from models.AGDN_nonlocal import DenseNet
# from models.AGDN_SONA import DenseNet
# from models.AGDN_DML import DenseNet
# from models.AGDN_cam_att import DenseNet
# from models.AGDN_gradcam import DenseNet
# from models.AGDN_crop_input2 import DenseNet

train_params = {
    'batch_size': 8,
    'n_epochs': 80,
    'initial_learning_rate': 0.01,
    'reduce_lr_epoch_1': 60,  # epochs * 0.5
    'reduce_lr_epoch_2': 120,  # epochs * 0.75
    'reduce_lr_epoch_3': 180,
    'reduce_lr_epoch_4': 200,
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'divide_255',  # None, divide_256, divide_255, divide_127.5, by_chanels
}


def get_train_params():
    return train_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet-BC',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 16, 24, 32, 40],
        default= 12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[41, 69, 57, 85, 121],
        default=69,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=4, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=0.00001, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)

    parser.add_argument(
        '--which_split', type=str, default='split1',
        help='Which split for cross validation')

    parser.add_argument(
        '--lamda', type=float, default=0.5,
        help='The weight of CGG in the AGD module')

    args = parser.parse_args()

    if not args.keep_prob:
        args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params()
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Initialize the model..")
    model = DenseNet(**model_params)
    if args.train:
        model.load_model()
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Testing...")

        _, _, _, acc1, acc2, acc, nr1, br1, ir1, kappa1, nr2, br2, ir2, kappa2, \
            nr, br, ir, kappa = model.test(batch_size=8)

        print ("Net1_accuracy:", acc1)
        print ("Net2_accuracy:", acc2)
        print ("Total_accuracy:", acc)

        print ("Net1_normal_recall:", nr1)
        print ("Net2_normal_recall:", nr2)
        print ("Total_normal_recall:", nr) 

        print ("Net1_bleed_recall:", br1)
        print ("Net2_bleed_recall:", br2)               
        print ("Total_bleed_recall:", br)   

        print ("Net1_inflam_recall:", ir1)  
        print ("Net2_inflam_recall:", ir2)                                           
        print ("Total_inflam_recall:", ir) 

        print ("Net1_kappa:", kappa1)  
        print ("Net2_kappa:", kappa2)
        print ("Total_kappa:", kappa)                                                                                                 