import torch, argparse, yaml, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
device = torch.device("cuda:0")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--operation', type=str, choices=['train', 'test'], default='test')
    parser.add_argument("--input_feed", type=str, default='direct_label_input')
    
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--data_path', type=str, default='data/yahoo/')
    parser.add_argument('--log_path', type=str, default='log/')
    parser.add_argument('--model_save_path', type=str, default='model/model.pt')
    parser.add_argument('--output_path', type=str, default='output/output.csv')
    parser.add_argument('--test_while_train', action='store_true')
    
    parser.add_argument("-sd", "--seed", type=int, default=0)
    parser.add_argument("-k", "--k", type=int, default=10)
    parser.add_argument("-s", "--step", type=int, default=10000)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("--valid_step", type=int, default=50)
    parser.add_argument("-w", "--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    args = parser.parse_args()
    
    if args.operation == 'train':
        os.makedirs("/".join(args.model_save_path.split('/')[:-1]), exist_ok=True)
    else:
        os.makedirs("/".join(args.output_path.split('/')[:-1]), exist_ok=True)
    
    return args


def train(config, args):
    model = model_dict[config['model']['name']](config['model'], config['data']['feature_size']).to(device)
    
    train_data = input_feed_dict[args.input_feed](args.data_path + '/train.txt', config, args)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_data = input_feed_dict['eval'](args.data_path + '/valid.txt', config, args)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    train_writer = SummaryWriter(log_dir=args.log_path+"/train.log")
    valid_writer = SummaryWriter(log_dir=args.log_path+"/valid.log")
    
    if args.test_while_train:
        test_data = input_feed_dict['eval'](args.data_path + '/test.txt', config, args)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_writer = SummaryWriter(log_dir=args.log_path+"/test.log")
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio * args.step, num_training_steps=args.step)
    
    algorithm = algorithm_dict[config['algorithm']['name']](optimizer, scheduler, config, device, args)
    
    best_object = 0.
    
    valid_result, _ = eval(model, algorithm, config['data'], config['eval'], valid_dataloader)
    print("Step 0, valid:", valid_result)
    valid_writer.add_scalars("Valid", valid_result, 0)
    
    train_dataloader = itertools.cycle(train_dataloader)
    
    for step in tqdm(range(1, args.step + 1)):
        input_data = next(train_dataloader)
        input_data = train_data.process_input(input_data, model, algorithm, device)
        
        loss, reward = algorithm.fit(model, input_data, device)
        
        train_writer.add_scalars("Train", {'loss' : loss}, step)
        if reward:
            train_writer.add_scalars("Train", {'reward' : reward}, step)
        for param_group in optimizer.param_groups:
            args.lr = param_group['lr']
        
        # summary / log
        if step % args.valid_step == 0:
            if reward:
                print(f"Step {step}: lr={args.lr}, loss={loss}, reward={reward}")
            else:
                print(f"Step {step}: lr={args.lr}, loss={loss}")
            valid_result, _ = eval(model, algorithm, config['data'], config['eval'], valid_dataloader)
            print(f"Step {step}, valid:", valid_result)
            valid_writer.add_scalars("Valid", valid_result, step)
            
            if 'objective' in config['eval'] and config['eval']['objective'] in valid_result:
                cur_object = valid_result[config['eval']['objective']]
                if best_object <=  cur_object:
                    best_object = cur_object
                    print(f"Save model, best object: {best_object}")
                    torch.save(model.state_dict(), args.model_save_path)
                else:
                    print(f"Current object:{cur_object}, best objectbest_object:{best_object}")
                
            if args.test_while_train:
                test_result, _ = eval(model, algorithm, config['data'], config['eval'], test_dataloader)
                print(f"Step {step}, test:", test_result)
                test_writer.add_scalars("Test", test_result, step)
            

        
def test(config, args):
    model = model_dict[config['model']['name']](config['model'], config['data']['feature_size']).to(device)
    load_model(model, args.model_save_path)
    
    test_data = input_feed_dict['eval'](args.data_path + '/test.txt', config, args)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    algorithm = algorithm_dict[config['algorithm']['name']](None, None, config, device, args)
    
    test_result, result_df = eval(model, algorithm, config['data'], config['eval'], test_dataloader, return_df=True)
    result_df.to_csv(args.output_path, index=False)
    print(test_result)

    
def eval(model, algorithm, data_setting, eval_setting, eval_dataloader, return_df=False):
    qids, preds, labels, eval_results, df_result = [], [], [], {}, {}
    for input_data in eval_dataloader:
        qids.extend(input_data['qid'])
        mask = input_data['mask'].to(device)
        output = algorithm.eval(model, input_data, device)
        output = torch.where(mask, output, torch.tensor(-1e8, dtype=torch.float32, device=device))
        preds.append(output)
        labels.append(input_data['label'].to(device))
        
    preds, labels = torch.cat(preds, dim = 0), torch.cat(labels, dim = 0)
    sorted_preds, ranking = torch.sort(preds, dim = 1, descending = True)
    
    if return_df:
        df_result['qid'] = qids
    
    for metric in eval_setting['metrics']:
        if metric == 'fair':
            for topn in eval_setting['metric_topn']:
                chunked_preds = sorted_preds[:, :topn]
                chunked_labels = torch.gather(labels, dim=1, index=ranking[:, :topn])
                all_result, avg_result = fair.fairness_metric(chunked_preds, chunked_labels, data_setting['max_label'], device)
                eval_results[f'{metric}@{topn}'] = avg_result.item()
        else:
            for topn in eval_setting['metric_topn']:
                all_result, avg_result = metric_dict[metric](ranking, labels, data_setting['max_label'], device, topn)
                eval_results[f'{metric}@{topn}'] = avg_result.item()
                if return_df:
                    df_result[f'{metric}@{topn}'] = all_result.tolist()
    
    return eval_results, pd.DataFrame(df_result)
    

if __name__ == "__main__":
    args = parse()
    set_seed(args.seed)
    with open(args.config_path, 'r', encoding='utf-8') as fr:
        config = yaml.load(fr.read(), Loader=yaml.FullLoader)
    with open(args.data_path + "/settings.yaml", 'r', encoding='utf-8') as fr:
        config['data'] = yaml.load(fr.read(), Loader=yaml.FullLoader)
    print(config)
    if args.operation == 'train':
        train(config, args)
    else:
        test(config, args)