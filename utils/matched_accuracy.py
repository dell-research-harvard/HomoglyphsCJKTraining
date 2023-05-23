# %%
# Import dependencies 
import json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
from pathlib import Path

# %%
def same_matched(a,b):
    '''Whether the matched is in ground truth'''
    for ele in a:
        ele_name = Path(ele).name
        for truth in b:
            if ele_name == Path(truth).name: # If any of the ele in a equals b, return 1, after the iter, if nothing returns, just return 0, remove the path
                return 1
    return 0


## In order to calculate the accuracy, you need to change everything to matched_tk_path format...
## So maybe for simstring you want to implement it separately and evaluate separately...
def calculate_matched_accuracy(matched_results = "DATAFRAME/FOR/CALCULATE/ACCURACY", match_ground_truth = "/mnt/data01/yxm/record_linkage_clean_dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0314.json", \
    match_test_subset = "/mnt/data01/yxm/record_linkage_clean_dataset/test_paths.json", source_image_path = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_partner_crop_36673/"):

    with open(match_ground_truth) as f:
        truth_TK_partnerpath_2_titlepath = json.load(f)

    with open(match_test_subset) as f:
        matched_test_subset_load = json.load(f)
    # Should only use file names
    matched_test_subset = []
    for image in matched_test_subset_load["images"]:
        image_name = source_image_path + image["file_name"].split('/')[-1]+'.png'
        matched_test_subset.append(image_name)

    matched_results["TK_truth_image"] = matched_results.apply(lambda x:truth_TK_partnerpath_2_titlepath[x["source"]] if x["source"] in truth_TK_partnerpath_2_titlepath else None,axis=1)
    matched_results.dropna(subset=["TK_truth_image"],inplace = True,axis = 0)
    matched_results_in_test = matched_results[matched_results['source'].isin(matched_test_subset)]           

    matched_results_in_test[f"accuracy"] = matched_results_in_test.apply(lambda x:same_matched([x["matched_tk_path"]],x["TK_truth_image"]), axis = 1)
    accuracy = matched_results_in_test[f"accuracy"].mean()
    return accuracy

def calculate_pr_matched_accuracy(matched_results = "DATAFRAME/FOR/CALCULATE/ACCURACY", match_ground_truth = "/mnt/data01/yxm/record_linkage_clean_dataset/ground_truth/matched/PR_matched_1092.csv"):
    truth_PR = pd.read_csv(match_ground_truth)

    source_list = truth_PR.source.values.tolist()
    target_list = truth_PR.target.values.tolist()
    
    pr_source_2_target = {}
    for source, target in zip(source_list,target_list):
        pr_source_2_target[source] = [target]
    # Should only use file names

    matched_results["PR_truth_image"] = matched_results.apply(lambda x:pr_source_2_target[x["source"]] if x["source"] in pr_source_2_target else None,axis=1)
    matched_results.dropna(subset=["PR_truth_image"],inplace = True,axis = 0)    
    
    print(len(matched_results))
    
    matched_results.dropna(subset=["matched_tk_path"],inplace = True,axis = 0)    
    
    print('after drop NA', len(matched_results))

    matched_results[f"accuracy"] = matched_results.apply(lambda x:same_matched([x["matched_tk_path"]],x["PR_truth_image"]), axis = 1)
    accuracy = matched_results[f"accuracy"].mean()
    return accuracy
