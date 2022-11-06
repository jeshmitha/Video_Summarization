import json
from re import I
from turtle import color
import matplotlib.pyplot as plt
import os
from calculate_fscores import tvsum_fscore_gts
from calculate_fscores import tvsum_fscore_us
from calculate_fscores import summe_fscore_gts
from calculate_fscores import summe_fscore_us
from calculate_fscores import tvsum_kendall_gts
from calculate_fscores import tvsum_kendall_us
from calculate_fscores import tvsum_spearman_gts
from calculate_fscores import tvsum_spearman_us
from calculate_fscores import tvsum_mse
from calculate_fscores import summe_mse

losses = ['reconstruction_loss', 'sparsity_loss', 'g_loss', 'd_o_loss','d_f_loss', 'critic_real', 'critic_fake']
nl=len(losses)

def evaluator(config):

    splits_filename = './data/splits/' + config.setting + '_splits.json'
    evaluation_result_dir = "./results/"+config.setting+"/split"+str(config.split_index)
    if not os.path.exists(evaluation_result_dir):
            os.makedirs(evaluation_result_dir) 
    with open(splits_filename) as f:
        data = json.loads(f.read())
        train_videos=data[config.split_index]['train_keys']
        test_videos=data[config.split_index]['test_keys']
    n_train=len(train_videos)
    n_test=len(test_videos)

    l_train = [list() for j in range(nl)]
    l_test = [list() for j in range(nl)]

    for epoch_i in range(0, 100):
        train_score_save_path = config.train_score_dir.joinpath(f'epoch_{epoch_i}.json')
        test_score_save_path = config.test_score_dir.joinpath(f'epoch_{epoch_i}.json')

        f1=open(train_score_save_path)
        f2=open(test_score_save_path)

        train_data = json.load(f1)
        test_data = json.load(f2)

      
        le_train=[0]*nl
        for v in train_videos:
            for li in range(nl):
                le_train[li] = le_train[li]+train_data[v][losses[li]]

        for i in range(nl):
            l_train[i].append(le_train[i]/n_train)
        
        le_test=[0]*nl
        for v in test_videos:
            for li in range(nl):
                le_test[li] = le_test[li]+test_data[v][losses[li]]

        for i in range(nl):
            l_test[i].append(le_test[i]/n_test)
    
    for i in range(nl):
        plt.plot(l_train[i], color='green')
        plt.plot(l_test[i], color='red')
        plt.title(losses[i])
        plt.savefig(evaluation_result_dir+"/"+losses[i]+".png")
        plt.clf()
     
    if config.setting[0:5]=='tvsum':

        f_us=tvsum_fscore_us(config.test_score_dir)
        f_us_final=f_us[-1]
        f_us_max = max(f_us)
        f_us_me=f_us.index(f_us_max)
        
        f_gt=tvsum_fscore_gts(config.test_score_dir)
        f_gt_final=f_gt[-1]
        f_gt_max = max(f_gt)
        f_gt_me=f_gt.index(f_gt_max)

        k_gt=tvsum_kendall_gts(config.test_score_dir)
        k_gt_final=k_gt[-1]
        k_gt_max = max(k_gt)
        k_gt_me=k_gt.index(k_gt_max)

        k_us=tvsum_kendall_us( config.test_score_dir)
        k_us_final=k_us[-1]
        k_us_max = max(k_us)
        k_us_me=k_us.index(k_us_max)

        s_gt=tvsum_spearman_gts(config.test_score_dir)
        s_gt_final=s_gt[-1]
        s_gt_max = max(s_gt)
        s_gt_me=s_gt.index(s_gt_max)

        s_us=tvsum_spearman_us( config.test_score_dir)
        s_us_final=s_us[-1]
        s_us_max = max(s_us)
        s_us_me=s_us.index(s_us_max)

        msee=tvsum_mse(config.test_score_dir)
        msee_final=msee[-1]
        msee_min=min(msee)
        msee_me=msee.index(msee_min)

        plt.plot(k_gt, color='red')
        plt.title("Kendall with GT")
        plt.savefig(evaluation_result_dir+"/k_gt.png")
        plt.clf() 

        plt.plot(s_gt, color='red')
        plt.title("Spearman with GT")
        plt.savefig(evaluation_result_dir+"/s_gt.png")
        plt.clf() 

        plt.plot(k_us, color='red')
        plt.title("Spearman with GT")
        plt.savefig(evaluation_result_dir+"/s_gt.png")
        plt.clf() 

        plt.plot(s_us, color='blue')
        plt.title("Spearman with GT")
        plt.savefig(evaluation_result_dir+"/s_gt.png")
        plt.clf() 

        plt.plot(msee, color='green')
        plt.title("Spearman with GT")
        plt.savefig(evaluation_result_dir+"/s_gt.png")
        plt.clf() 

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(k_us, color='red')
        ax1.plot(s_us, color='blue')
        ax1.plot(msee, color='green')
        ax2.plot(f_us, color='purple')
        ax1.set_ylabel('K,S,MSE')
        ax2.set_ylabel('F-score')
        plt.title("Kendall/Spearman/MSE/f-score with User Annotations")
        plt.savefig(evaluation_result_dir+"/ksmf_us.png")
        plt.clf() 

        out_dict={}
        out_dict['f_gt']=f_gt 
        out_dict['f_gt_final']=f_gt_final
        out_dict['f_gt_max']=f_gt_max
        out_dict['f_gt_me']=f_gt_me
        out_dict['k_gt']=k_gt 
        out_dict['k_gt_final']=k_gt_final
        out_dict['k_gt_max']=k_gt_max
        out_dict['k_gt_me']=k_gt_me
        out_dict['s_gt']=s_gt 
        out_dict['s_gt_final']=s_gt_final
        out_dict['s_gt_max']=s_gt_max
        out_dict['s_gt_me']=s_gt_me
        out_dict['f_us']=f_us 
        out_dict['f_us_final']=f_us_final
        out_dict['f_us_max']=f_us_max
        out_dict['f_us_me']=f_us_me
        out_dict['k_us']=k_us 
        out_dict['k_us_final']=k_us_final
        out_dict['k_us_max']=k_us_max
        out_dict['k_us_me']=k_us_me
        out_dict['s_us']=s_us 
        out_dict['s_us_final']=s_us_final
        out_dict['s_us_max']=s_us_max
        out_dict['s_us_me']=s_us_me
        out_dict['mse']=msee
        out_dict['mse_final']=msee_final
        out_dict['mse_min']=msee_min
        out_dict['mse_me']=msee_me

        with open(evaluation_result_dir+"/results.json", "w+") as f:
            json.dump(out_dict, f)
        
    elif config.setting[0:5]=='summe':

        f_us=summe_fscore_us( config.test_score_dir)
        f_us_final=f_us[-1]
        f_us_max = max(f_us)
        f_us_me=f_us.index(f_us_max)
                
        f_gt=summe_fscore_gts(config.test_score_dir)
        f_gt_final=f_gt[-1]
        f_gt_max = max(f_gt)
        f_gt_me=f_gt.index(f_gt_max)

        msee=summe_mse(config.test_score_dir)
        msee_final=msee[-1]
        msee_min=min(msee)
        msee_me=msee.index(msee_min)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(msee, color='green')
        ax2.plot(f_us, color='purple')
        ax1.set_ylabel('MSE')
        ax2.set_ylabel('F-score')
        plt.title("MSE/f-score with User Annotations")
        plt.savefig(evaluation_result_dir+"/mf_us.png")
        plt.clf() 

        out_dict={}
        out_dict['f_gt']=f_gt 
        out_dict['f_gt_final']=f_gt_final
        out_dict['f_gt_max']=f_gt_max
        out_dict['f_gt_me']=f_gt_me
        out_dict['f_us']=f_us 
        out_dict['f_us_final']=f_us_final
        out_dict['f_us_max']=f_us_max
        out_dict['f_us_me']=f_us_me
        out_dict['mse']=msee
        out_dict['mse_final']=msee_final
        out_dict['mse_min']=msee_min
        out_dict['mse_me']=msee_me

        with open(evaluation_result_dir+"/results.json", "w+") as f:
            json.dump(out_dict, f)

    plt.plot(f_gt, color='red')
    plt.title("F-score with GT")
    plt.savefig(evaluation_result_dir+"/f_gt.png")
    plt.clf() 
    
    plt.plot(f_us, color='red')
    plt.title("F-score with US")
    plt.savefig(evaluation_result_dir+"/f_us.png")
    plt.clf() 

    plt.plot(msee, color='red')
    plt.title("MSE loss")
    plt.savefig(evaluation_result_dir+"/mse.png")
    plt.clf()