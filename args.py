import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Referred Part Segmentation. Model = Image+Text')

    
    parser.add_argument('-b', '--batch-size', default=10, type=int)

    
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--starting-epoch', default=1, type=int, help='epoch number to start with')


    parser.add_argument('--base-lr', default=3e-4, type=float, help='the initial learning rate')
  

    parser.add_argument('--clip-model', default='RN50', help='which clip model to use')

    parser.add_argument('--result-dir', default='./Results/', help='path to save resultant images')
    parser.add_argument('--model-dir', default='../ContextPromptPart_model', help='path to save models')

    parser.add_argument('--dataset-dir', default= '/home/harsh21122/tmp/cat_dataset', help='path where dataset is uploaded')


    parser.add_argument('--resume', default=False, type = bool, help='resume from checkpoint')
    parser.add_argument('--model-name', default='../ContextPromptPart_model/last_model',  help='if resume from checkpoint is true, name and path of model to load')


    parser.add_argument('--calc_accuracy_training', default=False, type = bool, help='if need to calculate accuracy while training each epoch')
    parser.add_argument('--multi-step-scheduler', default=False, type = bool, help='if need to add MultiStepLR')
    parser.add_argument('--lr-decay', default=0.1, type=float, help='lr decay rate for MultiStepLR')
    parser.add_argument('--milestones', default=[30, 50, 80], type=list, help='milestones for MultiStepLR')
    parser.add_argument('--wandb', default = True, type = bool, help='if need to log to wandb')



    parser.add_argument('--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')



    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
