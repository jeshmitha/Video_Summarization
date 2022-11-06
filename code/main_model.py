import argparse
import torch
from configs import get_config
from solver import Solver
from final_evaluation import evaluator

#training and testing with tvsum on 5 splits; one model for each split
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting')
    parser.add_argument('--split_index')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model_config = get_config(setting=args.setting, split_index=int(args.split_index))
    model = Solver(model_config)
    model = model.to(device)
    model.build()
    model.train()
    evaluator(model_config)