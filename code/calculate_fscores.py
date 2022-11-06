from os import listdir
import json
import numpy as np
import h5py
import hdf5storage
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
import math as m
from sklearn.metrics import mean_squared_error

def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        
        item_dict = {
        'video': video[0, 0],
        'category': category[0, 0],
        'title': title[0, 0],
        'length': length[0, 0],
        'nframes': nframes[0, 0],
        'user_anno': user_anno,
        'gt_score': gt_score
        }
        
        data_list.append((item_dict))
    return data_list
tvsum_data = load_tvsum_mat('ydata-tvsum50.mat')

def get_rc_func(metric):
    if metric == 'kendalltau':
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == 'spearmanr':
        f = lambda x, y: spearmanr(x, y)
    else:
        raise RuntimeError
    return f
def upsample(scores, v):
    PATH_TvSum = h5py.File('./data/TVSum/eccv16_dataset_tvsum_google_pool5.h5','r')

    n_frames=PATH_TvSum[v]['n_frames'][()]
    positions=PATH_TvSum[v]['picks'][:]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]
    return frame_scores

def tvsum_kendall_gts(path):
    rc_func=get_rc_func('kendalltau')
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_TvSum = h5py.File('./data/TVSum/eccv16_dataset_tvsum_google_pool5.h5','r')

    k_gt_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vk=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = upsample(np.asarray(data[video_name]['frame_scores']).flatten(), "video_"+str(video_index))
                gts= upsample(PATH_TvSum["video_"+str(video_index)]['gtscore'][:], "video_"+str(video_index))
                ta=rc_func(gts, scores)[0]
                if m.isnan(ta):
                        print("k_gt", epoch, video_name)
                        vk.append(0)
                else:
                        vk.append(ta)
            k_gt_epochs.append(np.mean(vk))
    return k_gt_epochs
    
def tvsum_spearman_gts(path):
    rc_func=get_rc_func('spearmanr')
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_TvSum = h5py.File('./data/TVSum/eccv16_dataset_tvsum_google_pool5.h5','r')

    s_gt_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vs=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = upsample(np.asarray(data[video_name]['frame_scores']).flatten(), "video_"+str(video_index))
                gts= upsample(PATH_TvSum["video_"+str(video_index)]['gtscore'][:], "video_"+str(video_index))
                sp=rc_func(gts, scores)[0]
                if m.isnan(sp):
                    print("s_gt", epoch, video_name)
                    vs.append(0)
                else:
                    vs.append(sp)
            s_gt_epochs.append(np.mean(vs))
    return s_gt_epochs

def tvsum_mse(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_TvSum = h5py.File('./data/TVSum/eccv16_dataset_tvsum_google_pool5.h5','r')

    mse_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vmse=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                gts= PATH_TvSum["video_"+str(video_index)]['gtscore'][:]
                mse=mean_squared_error(gts, scores)
                vmse.append(mse)
            mse_epochs.append(np.mean(vmse))
    return mse_epochs

def tvsum_kendall_us(path):
    rc_func=get_rc_func('kendalltau')
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    
    k_us_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vk=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = upsample(np.asarray(data[video_name]['frame_scores']).flatten(), "video_"+str(video_index))
                user_anno=tvsum_data[int(video_index)-1]['user_anno'].T
                D = [rc_func(x, scores)[0] for x in user_anno]
                ta=np.mean(D)
                if m.isnan(ta):
                    print("k_us", epoch, video_name)
                    vk.append(0)
                else:
                    vk.append(ta)
            k_us_epochs.append(np.mean(vk))
    return k_us_epochs

def tvsum_spearman_us(path):
    rc_func=get_rc_func('spearmanr')
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    
    s_us_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vs=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = upsample(np.asarray(data[video_name]['frame_scores']).flatten(), "video_"+str(video_index))
                user_anno=tvsum_data[int(video_index)-1]['user_anno'].T
                D = [rc_func(x, scores)[0] for x in user_anno]
                sp=np.mean(D)
                if m.isnan(sp):
                    print("s_us", epoch, video_name)
                    vs.append(0)
                else:
                    vs.append(sp)
            s_us_epochs.append(np.mean(vs))
    return s_us_epochs

def tvsum_fscore_gts(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_TvSum = './data/TVSum/eccv16_dataset_tvsum_google_pool5.h5'
    eval_method = 'avg' 
    
    f_score_epochs = []
    for epoch in results:
        all_scores = []
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())

            for video_name in keys:
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                all_scores.append(scores)

        all_gt_scores, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(PATH_TvSum, 'r') as hdf:
            for video_name in keys:
                video_index = video_name[8:] 

                gt_score = np.array( hdf.get('video_'+video_index+'/gtscore') )
                sb = np.array( hdf.get('video_'+video_index+'/change_points') )
                n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
                positions = np.array( hdf.get('video_'+video_index+'/picks') )
                
                all_gt_scores.append(gt_score)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)
        
        all_gt_summary = generate_summary(all_shot_bound, all_gt_scores, all_nframes, all_positions)

        all_f_scores = []

        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            gt_summary = all_gt_summary[video_index]
            f_score = evaluate_summary(summary, np.expand_dims(gt_summary, axis=0), eval_method)	
            all_f_scores.append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))
 
    return f_score_epochs
    
def tvsum_fscore_us(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_TvSum = './data/TVSum/eccv16_dataset_tvsum_google_pool5.h5'
    eval_method = 'avg' 

    f_score_epochs = []
    for epoch in results:
        all_scores = []
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())

            for video_name in keys:
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                all_scores.append(scores)

        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(PATH_TvSum, 'r') as hdf:        
            for video_name in keys:
                video_index = video_name[8:]
                
                user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
                sb = np.array( hdf.get('video_'+video_index+'/change_points') )
                n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
                positions = np.array( hdf.get('video_'+video_index+'/picks') )

                all_user_summary.append(user_summary)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

        all_f_scores = []

        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            user_summary = all_user_summary[video_index]
            f_score = evaluate_summary(summary, user_summary, eval_method)	
            all_f_scores.append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))
    return f_score_epochs

def summe_mse(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_SumMe = h5py.File('./data/SumMe/eccv16_dataset_summe_google_pool5.h5')

    mse_epochs = []
    for epoch in results:
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            vmse=[]
            for video_name in keys:
                video_index = video_name[8:] 
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                gts= PATH_SumMe["video_"+str(video_index)]['gtscore'][:]
                mse=mean_squared_error(gts, scores)
                vmse.append(mse)
            mse_epochs.append(np.mean(vmse))
    return mse_epochs

def summe_fscore_gts(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_SumMe = './data/SumMe/eccv16_dataset_summe_google_pool5.h5'
    eval_method = 'max'  
    
    f_score_epochs = []
    for epoch in results:
        all_scores = []
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())

            for video_name in keys:
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                all_scores.append(scores)

        all_gt_scores, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(PATH_SumMe, 'r') as hdf:
            for video_name in keys:
                video_index = video_name[8:] 

                gt_score = np.array( hdf.get('video_'+video_index+'/gtscore') )
                sb = np.array( hdf.get('video_'+video_index+'/change_points') )
                n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
                positions = np.array( hdf.get('video_'+video_index+'/picks') )
                
                all_gt_scores.append(gt_score)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)
        
        all_gt_summary = generate_summary(all_shot_bound, all_gt_scores, all_nframes, all_positions)

        all_f_scores = []

        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            gt_summary = all_gt_summary[video_index]
            f_score = evaluate_summary(summary, np.expand_dims(gt_summary, axis=0), eval_method)	
            all_f_scores.append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))
 
    return f_score_epochs

def summe_fscore_us(path):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    PATH_SumMe = './data/SumMe/eccv16_dataset_summe_google_pool5.h5'
    eval_method = 'max' 

    f_score_epochs = []
    for epoch in results:
        all_scores = []
        with open(str(path)+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())

            for video_name in keys:
                scores = np.asarray(data[video_name]['frame_scores']).flatten()
                all_scores.append(scores)

        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(PATH_SumMe, 'r') as hdf:        
            for video_name in keys:
                video_index = video_name[8:]    
                user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
                sb = np.array( hdf.get('video_'+video_index+'/change_points') )
                n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
                positions = np.array( hdf.get('video_'+video_index+'/picks') )

                all_user_summary.append(user_summary)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

        all_f_scores = []

        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            user_summary = all_user_summary[video_index]
            f_score = evaluate_summary(summary, user_summary, eval_method)	
            all_f_scores.append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))
    return f_score_epochs