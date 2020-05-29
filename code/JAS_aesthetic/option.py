import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--path_to_train_csv', dest='path_to_train_csv',
                        help='directory to train dataset',
                        default="/data/mayme/dataset/JenAesthetics/train.csv",
                        type=str)
    parser.add_argument('--path_to_val_csv', dest='path_to_val_csv',
                        help='directory to val dataset',
                        default='/data/mayme/dataset/JenAesthetics/val.csv',
                        type=str)
    parser.add_argument('--path_to_test_csv', dest='path_to_test_csv',
                        help='directory to test dataset',
                        default='/data/mayme/dataset/JenAesthetics/test.csv',
                        type=str)
    parser.add_argument('--path_to_imgs', dest='path_to_imgs',
                        help='directory to images',
                        default="/data/mayme/dataset/JenAesthetics/dataset/", type=str)

    parser.add_argument('--path_to_model_weight', type=str, default='./pretrain_model/u_model.pth',
                        help='directory to pretrain model')

    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--init_lr', type=int, default=0.0003, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int,default=4,help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help ='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='2', help='which gpu to use')


    args = parser.parse_args()
    return args