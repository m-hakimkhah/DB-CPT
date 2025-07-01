import numpy as np
import torch
from data import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from solver_merge import Solver
from Backup_pesq import numParams

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

def main(args, model):
    tr_dataset = TrainDataset(json_dir=args.json_dir,
                              batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_dir,
                           batch_size=args.cv_batch_size)
    tr_loader = TrainDataLoader(data_set=tr_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                #num_workers=0,
                                pin_memory=True)
    cv_loader = CvDataLoader(data_set=cv_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             #num_workers=0,
                             pin_memory=True)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    #print(model)
    model.cuda()
    #model.to(device)
    print('The number of trainable parameters of the net is:%d' % (numParams(model)))

    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 weight_decay=args.l2)
    
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    solver = Solver(data, model, optimizer, args)
    solver.train()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     model = train_model
#     print(args)
#     main(args, model)