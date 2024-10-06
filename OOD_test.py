from ood_utils import get_device, get_ood_score, plot_distribution
import os
import torch
from data_utils import DatasetLoader
from torch.utils.data import DataLoader
import metrics
import matplotlib
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    dict_results_sample = {}
    dict_results_SAR_ACD = {}
    dict_results_FUSAR_ship = {}
    dict_results_MSTAR_OOD = {}

    test_model = False # Whether to batch test models or not

    parser = argparse.ArgumentParser("OOD_test")
    parser.add_argument('--model_root', type=str,
                        default=r'./filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/model')
    parser.add_argument('--npy_root', type=str,
                        default=r'./filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/npy')
    parser.add_argument('--save_excel', type=str,
                        default=r'./results/OOD_results.xlsx')
    parser.add_argument('--fig_results', type=str,
                        default=r'./results/')
    
    args = parser.parse_args()

    DMPC_model_path = r'./init_model_npy/resnet18CE.pt' # class number: 10
    DMPC_centroids_path = r'./init_model_npy/resnet18_CE_10_K70_HierarchicalClustering_centroids.npy'  # obtained by clustering

    DMPL_centroids_path = r'./filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/npy/DMPL_HierarchicalClustering_K40_0.5_0.9_last.npy'# obtained by DMPL
    DMPL_model_path = r'./filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/model/DMPL_HierarchicalClustering_K40_0.5_0.9_last.pt' #trained model supervised by JPEL
    
    # method_list = ['MaxLogit', 'MaxNorm', 'MSP', 'DML', 'Energy', 'ODIN', 'GradNorm', 'KNN', 'DMPC', 'DMPL']
    method_list = ['DMPL']

    for method in method_list:

        name = method
        if method == 'DMPL':
            try:
                model = torch.load(DMPL_model_path)
                print('model path', DMPL_model_path)
            except:
                print('can not load model')
        else:
            model = torch.load(DMPC_model_path)

        # The path for the trained model
        # The path for the centroids file
        centroids = None
        train_loader = None
        normalize_value = None

        device = get_device('0')

        model_f = torch.nn.Sequential(*list(model.children())[:-1])
        model_f = model_f.to(device)
        model = model.to(device)
        model_f = model_f.to(device)

        data_test_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/MSTAR/SOC'
        dataset = DatasetLoader('test', data_test_path)
        id_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        #ood test SAR SAMPLE
        data_test_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SAMPLE'
        dataset_sample = DatasetLoader('ood', data_test_path)
        ood_sample_loader = DataLoader(dataset=dataset_sample, batch_size=1, shuffle=False)

        # ood test FUSAR-ship
        data_test_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SHIP/FUSAR-ship'
        dataset_ship = DatasetLoader('ood', data_test_path)
        ood_ship_loader = DataLoader(dataset=dataset_ship, batch_size=1, shuffle=False)

        # ood test SAR-ACD
        data_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/AIRPLANE/SAR-ACD-main'
        ood_testset_airplane = DatasetLoader('ood', data_path)
        ood_airplane_loader = DataLoader(dataset=ood_testset_airplane, batch_size=1,
                                         shuffle=False)  # set batch_size=1 when test uncertainty

        # ###################### multiprototypes based ood detection ##########################
        #load centroids of ID data
        if method == 'DMPL':
            print('##############################')
            print('DMPL_centroids_path', DMPL_centroids_path)
            centroids = np.load(DMPL_centroids_path)
            print('centroids.shape', centroids.shape)
            if len(centroids.shape) == 3:
                centroids = centroids.reshape(-1, centroids.shape[-1])
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / norms
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader)
        elif method == 'DMPC':
            print('##############################')
            print('DMPC_centroids_path', DMPC_centroids_path)
            centroids = np.load(DMPC_centroids_path)
            print('centroids.shape', centroids.shape)
            if len(centroids.shape) == 3:
                centroids = centroids.reshape(-1, centroids.shape[-1])
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / norms
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader)
        elif method in ['KNN', 'KNN+']:
            #################### KNN score OOD detection #################
            data_train_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/MSTAR/SOC'
            trainset = DatasetLoader('train', data_train_path)
            train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)
        elif method in ['MaxLogit', 'DML', 'MaxNorm', 'MSP', 'Energy']:
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)
            normalize_value = np.sum(id_score)
            id_score = id_score / normalize_value
        else:
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)

        #calculate score with method
        sample_score = get_ood_score(ood_sample_loader, model, model_f, device, method, centroids, train_loader, normalize_value)
        FUSAR_ship_score = get_ood_score(ood_ship_loader, model, model_f, device, method, centroids, train_loader, normalize_value)
        SAR_ACD_score = get_ood_score(ood_airplane_loader, model, model_f, device, method, centroids, train_loader, normalize_value)

        results_sample = metrics.cal_metric(id_score, sample_score)
        print(name+': sample', results_sample)
        dict_results_sample[name] = results_sample

        results_airplane = metrics.cal_metric(id_score, SAR_ACD_score)
        print(name+': airplane', results_airplane)
        dict_results_SAR_ACD[name] = results_airplane

        results_ship = metrics.cal_metric(id_score, FUSAR_ship_score)
        print(name+': ship', results_ship)
        dict_results_FUSAR_ship[name] = results_ship

        plot_distribution([id_score, sample_score, SAR_ACD_score, FUSAR_ship_score],
                         ['MSTAR (ID)', 'SAMPLE (OOD)', 'SAR-ACD (OOD)', 'FUSAR-ship (OOD)'],
                         savepath=os.path.join(args.fig_results, name + '.png')) #dawn

    # Convert the dictionary to a DataFrame and then transpose it
    df_transposed_sample = pd.DataFrame(dict_results_sample).T
    df_transposed_SAR_ACD = pd.DataFrame(dict_results_SAR_ACD).T
    df_transposed_FUSAR_ship = pd.DataFrame(dict_results_FUSAR_ship).T

    with pd.ExcelWriter(args.save_excel) as writer:
        df_transposed_sample.to_excel(writer, header=False, sheet_name='sample')
        df_transposed_SAR_ACD.to_excel(writer, header=False, sheet_name='SAR_ACD')
        df_transposed_FUSAR_ship.to_excel(writer, header=False, sheet_name='FUSAR_ship')

