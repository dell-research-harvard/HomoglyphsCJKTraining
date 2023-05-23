# %%
# Import dependencies 
from multiprocessing.spawn import import_main_path
import time
import json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import pickle
from hyperopt import hp
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
from itertools import repeat
from fuzzychinese import FuzzyChineseMatch
import matplotlib.pyplot as plt
import textdistance as td
import sys
import copy
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure import CosineMeasure
from simstring.measure import JaccardMeasure
from simstring.measure import OverlapMeasure
from simstring.measure import DiceMeasure
from simstring.searcher import Searcher
from simstring.database import DictDatabase

from utils.nomatch_accuracy import calculate_nomatch_accuracy
from utils.matched_accuracy import calculate_matched_accuracy
from utils.matched_accuracy import calculate_pr_matched_accuracy

sys.path.append("..")
# %%
def custom_edit_distance(str1,str2):
    m = len(str1)
    n = len(str2)
    #dp = np.zeros([m+1,n+1]) # it is m rows, n columns
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] # This list is quicker than the above numpy array.
    for i in range(m+1):
        for j in range(n+1):
            if i==0 and j==0:
                dp[i][j]=0
            elif i==0:
                dp[i][j]=dp[i][j-1]+1
            elif j==0:
                dp[i][j]=dp[i-1][j]+1         
            elif str1[i-1]==str2[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                if str1[i-1] in cluster_dict_Japanese:
                    if str2[j-1] in cluster_dict_Japanese[str1[i-1]]:
                        dist=1*(1-cluster_dict_Japanese[str1[i-1]][str2[j-1]]) # This is gamma actually, the substitution cost is the homoglyphic distance
                    else:
                        dist=1 
                else:
                    dist=1
                str2_mean_10=1
                str1_mean_10=1
                #return 999
                insertion=str2_mean_10
                deletion=str1_mean_10
                dp[i][j] =  min(dp[i][j-1]+insertion,	 # Insert
                                dp[i-1][j]+deletion,	 # Remove
                                dp[i-1][j-1]+dist) 

    return dp[m][n]

def list_fd(word,list2):
    dist_list = []
    smallest_dist = 1000
    for word2 in tqdm(list2):
        if abs(len(str(word2[0]))-len(str(word))) > smallest_dist:
            dist = 1000
        dist = custom_edit_distance(str(word2[0]),str(word))
        dist_list.append(dist)
        if dist<smallest_dist:
            smallest_dist = dist# update the smallest distance
    min_dist = float(np.min(dist_list))
    min_dist_word_path = list2[np.argmin(dist_list)] # Which word in the ground truth dict get matched to
    return [min_dist, min_dist_word_path]

def list_lev(word,list2):
    dist_list = []
    smallest_dist = 1000
    for word2 in tqdm(list2):
        if abs(len(str(word2[0]))-len(str(word))) > smallest_dist:
            dist = 1000
        dist = td.levenshtein(str(word2[0]),str(word))
        dist_list.append(dist)
        if dist<smallest_dist:
            smallest_dist = dist# update the smallest distance
    min_dist = float(np.min(dist_list))
    min_dist_word_path = list2[np.argmin(dist_list)] # Which word in the ground truth dict get matched to
    return [min_dist, min_dist_word_path]

# You just need to change the analyzer or searcher
def list_simstring(list2, partners):
    db = DictDatabase(CharacterNgramFeatureExtractor(2)) # 2 grams graph
    title_dict = {}
    for title in list2:
        print(title)
        db.add(str(title[0]))
        title_dict[title[0]] = title[1]
    res_dict = {}
    thresh=0.01# 0.01 is too slow
    # Simstring set a thresh = 0.1 will be much quicker
    searcher_cos = Searcher(db, CosineMeasure())
    searcher_over = Searcher(db, OverlapMeasure())
    searcher_dice = Searcher(db, DiceMeasure())
    searcher_jac = Searcher(db, JaccardMeasure())

    mapper={"cos":searcher_cos,"over":searcher_over,"dice":searcher_dice,"jac":searcher_jac}

    for aux in tqdm(["cos","over","dice","jac"]):
        searcher=mapper[aux]
        nearest = []
        dist = []
        path = []
        for partner in tqdm(partners):#Can we directly?
            results=searcher.ranked_search(str(partner),thresh)
            try:
                nearest.append(list(results.items())[:1][0][0])
                path.append(title_dict[list(results.items())[:1][0][0]])
            except:
                nearest.append('')
                path.append('')
            try:
                dist.append(list(results.items())[:1][0][1])
            except:
                dist.append(1)
        # Not change here
        res_dict[f'sim_{aux}_nearest'] = nearest
        res_dict[f'sim_{aux}_nearest_dist'] = dist
        res_dict[f'sim_{aux}_nearest_img_path'] = path
    return res_dict


def list_fuzzyChinese(raw_word,test_dict, title_dict, task_name):# raw_word is the partner list/list 1, test_dict is the  title list/list 2
    #return all the results, just pass in two lists, don't pass in too many other things.
    '''
    Input: test_dict, raw_word
    Output: nearest neighbor word list, nearest neighbor dist list  
    '''
    fcm = FuzzyChineseMatch(ngram_range=(3,3),analyzer="stroke")# 3-gram
    fcm.fit(test_dict)
    top1_similar_stroke = fcm.transform(raw_word,n=1) # This is the nearest neighbor list - return top 10 similar

    fcm_char = FuzzyChineseMatch(ngram_range=(3,3),analyzer="char")
    fcm_char.fit(test_dict)
    top1_similar_char = fcm_char.transform(raw_word,n=1)

    res = pd.concat([
        pd.DataFrame(top1_similar_stroke,columns=[f'fuzzychinese_stroke_{task_name}_matched_word_1']),
        pd.DataFrame(fcm.get_similarity_score(),columns=[f'fuzzychinese_stroke_{task_name}_word_dist_1']),
        pd.DataFrame(top1_similar_char,columns=[f'fuzzychinese_char_{task_name}_matched_word_1']),
        pd.DataFrame(fcm_char.get_similarity_score(),columns=[f'fuzzychinese_char_{task_name}_word_dist_1'])
    ],axis = 1)

    res[f"fuzzychinese_stroke_{task_name}_matched_path_1"]=res.apply(lambda x:title_dict[x[f'fuzzychinese_stroke_{task_name}_matched_word_1']],axis=1)
    res[f"fuzzychinese_char_{task_name}_matched_path_1"]=res.apply(lambda x:title_dict[x[f'fuzzychinese_char_{task_name}_matched_word_1']],axis=1)

    return res
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partner_csv", type=str, #/mnt/data01/yxm/homo/homo_match_dataset/japan
        default="/mnt/data01/yxm/record_linkage_clean_dataset/ocr_json/partner_list_clean_final_eff_gcv_paddle_easy.csv",# let's use csv 
        help="Path to Partners list")
    parser.add_argument("--json_path", type=str,
        default='/mnt/data01/yxm/record_linkage_clean_dataset/ocr_json')# This is the path to things on Guppy
    parser.add_argument("--match_task", type=list, # You can change the task as you wantf the gcv is already gcv
        default=[# TK task# Please update the GCV titles list - also make to no dup...
#             ['eff_2_efftk','effocr_partner','effocr_tk_title_dup_68352_clean_path.json','eff']
        #['easy_2_paddlepr','easy_ocr_partner','paddleocr_pr_title.json','cjk'], ['easy_2_easypr','easy_ocr_partner','easyocr_pr_title.json','cjk'], \
        #['paddle_2_paddlepr','paddle_ocr_partner','paddleocr_pr_title.json','cjk'],
        ['paddle_2_easypr','paddle_ocr_partner','easyocr_pr_title.json','cjk']
        ],#We only want this for now...
        help="Matching tasks to perform")
    parser.add_argument("--homo", action="store_true", default=False, 
        help="Levenstein Distance Matching")
    parser.add_argument("--simstring", action="store_true", default=False, 
        help="Levenstein Distance Matching")
    parser.add_argument("--lev", action="store_true", default=False, 
        help="Levenstein Distance Matching")
    parser.add_argument("--fuzzychinese_stroke", action="store_true", default=False, 
        help="Fuzzychinese Stroke Matching")
    # Save output!
    parser.add_argument("--save_output", type=str, required=True, 
        help="Save output!")
    args = parser.parse_args()

    # Load the homo dict change to a new expanded dict
    with open("/mnt/data01/homo/cjk_homoglyphs/char_char_dist_dict_800_ja_expanded_easy.pickle",'rb') as f:
        cluster_dict_Japanese = pickle.load(f)

    partner_csv = pd.read_csv(args.partner_csv)
    match_task = args.match_task
    # Save output
    os.makedirs(args.save_output, exist_ok=True)
    partner_dict_list = []
    # Iterate over dfferent tasks
    '''
    list 1 is the source
    list 2 is what we parallize on since it is vert long
    '''
    accuracy_dict = {}
    # accuracy dict is initialized here
    store_time = {}

    # df_matched = pd.DataFrame(list(zip(list2,result_list)), columns=['truth','result'])

    # Run the matching if do_match is True
    for task_name, partner_ocr_choice, title, homo_type in tqdm(match_task):
        with open(os.path.join(args.json_path,f'{title}')) as f:
            title_list = json.load(f) # This is list2: title list
        partner_list_for_match = partner_csv[partner_ocr_choice].values.tolist()
        # Maybe you can change the format of this dataset...
        if args.homo:
            match_method = 'homo'
            start_time = time.time()
            with Pool(32) as p:
                mindist_WordPath_list = p.map(partial(list_fd,list2=title_list),partner_list_for_match) #result_list is the picture
            time_span = time.time()-start_time
            store_time[f"{task_name}_{match_method}"] = time_span
            with open(os.path.join(args.save_output,'time_speed.json'),'w') as f:
                json.dump(store_time, f, ensure_ascii=False)

            matched_list, distance_list, path_list = map(lambda x: list(x),repeat([],3))
            for id, word_dist_min in enumerate(mindist_WordPath_list):
                distance_list.append(round(word_dist_min[0],2))
                matched_list.append(word_dist_min[1][0])
                path_list.append(word_dist_min[1][1])

            df_match_result = pd.DataFrame({f'{match_method}_{task_name}_matched_word_1':matched_list, \
                f'{match_method}_{task_name}_matched_word_dist_1':distance_list, \
                f'{match_method}_{task_name}_matched_path_1': path_list})

            df_matched = pd.concat([df_match_result,partner_csv], axis=1)

            df_matched["matched_tk_path"] = df_matched[f"{match_method}_{task_name}_matched_path_1"]
            df_matched["source"] = df_matched["partner_path"]
            df_matched["distance"] = df_matched[f'{match_method}_{task_name}_matched_word_dist_1']
            df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_{match_method}.csv'))


            if 'pr' not in task_name:
                accuracy_dict[f"{task_name}_{match_method}_match"] = calculate_matched_accuracy(matched_results = df_matched)
                print('matched test accuracy:', calculate_matched_accuracy(matched_results = df_matched))
            
            else:
                accuracy_dict[f"{task_name}_{match_method}_match"] = calculate_pr_matched_accuracy(matched_results = df_matched)
                print('matched test accuracy:', calculate_pr_matched_accuracy(matched_results = df_matched))
                   
            '''
            From a file storing the best threshold
            '''
            # if "gcv" in task_name:
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched, file_name = f"df_full_matched_gcv_2_gcvtk_{match_method}.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_{match_method}_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched, file_name = f"df_full_matched_gcv_2_gcvtk_{match_method}.csv", levenshtein_match = False)
     
            # else:        
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched, file_name = f"df_full_matched_eff_2_efftk_{match_method}.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_{match_method}_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched, file_name = f"df_full_matched_eff_2_efftk_{match_method}.csv", levenshtein_match = False)

            with open(os.path.join(args.save_output,'japan_task_accuracy.json'),'w') as f:
                json.dump(accuracy_dict,f,ensure_ascii=False)

        if args.lev:
            match_method = 'lev'
            start_time = time.time()
            with Pool(32) as p:
                mindist_WordPath_list = p.map(partial(list_lev,list2=title_list),partner_list_for_match) #result_list is the picture
            time_span = time.time()-start_time
            store_time[f"{task_name}_lev"] = time_span
            with open(os.path.join(args.save_output,'time_speed.json'),'w') as f:
                json.dump(store_time, f, ensure_ascii=False)

            matched_list, distance_list, path_list = map(lambda x: list(x),repeat([],3))
            for id, word_dist_min in enumerate(mindist_WordPath_list):
                distance_list.append(round(word_dist_min[0],2))
                matched_list.append(word_dist_min[1][0])
                path_list.append(word_dist_min[1][1])

            df_match_result = pd.DataFrame({f'lev_{task_name}_matched_word_1':matched_list, \
                f'lev_{task_name}_matched_word_dist_1':distance_list, \
                f'lev_{task_name}_matched_path_1': path_list})

            df_matched = pd.concat([df_match_result,partner_csv], axis=1)

            df_matched["matched_tk_path"] = df_matched[f"lev_{task_name}_matched_path_1"]
            df_matched["source"] = df_matched["partner_path"]
            df_matched["distance"] = df_matched[f'lev_{task_name}_matched_word_dist_1']
            df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_{match_method}.csv'))

            if 'pr' not in task_name:
                accuracy_dict[f"{task_name}_{match_method}_match"] = calculate_matched_accuracy(matched_results = df_matched)
                print('matched test accuracy:', calculate_matched_accuracy(matched_results = df_matched))
            
            else:
                accuracy_dict[f"{task_name}_{match_method}_match"] = calculate_pr_matched_accuracy(matched_results = df_matched)
                print('matched test accuracy:', calculate_pr_matched_accuracy(matched_results = df_matched))

            '''
            From a file storing the best threshold
            '''
            # if "gcv" in task_name:
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched, file_name = "df_full_matched_gcv_2_gcvtk_lev.csv", levenshtein_match = True))
            #     accuracy_dict[f"{task_name}_lev_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched, file_name = "df_full_matched_gcv_2_gcvtk_lev.csv", levenshtein_match = True)
     
            # else:        
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched, file_name = "df_full_matched_eff_2_efftk_lev.csv", levenshtein_match = True))
            #     accuracy_dict[f"{task_name}_lev_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched, file_name = "df_full_matched_eff_2_efftk_lev.csv", levenshtein_match = True)

            with open(os.path.join(args.save_output,'japan_task_accuracy.json'),'w') as f:
                json.dump(accuracy_dict,f,ensure_ascii=False)

        if args.simstring:
            '''
            Let's not change the list_simstring functions, just pass different methods in the matched_tk_path each time...
            You have title_dict and partner_list_for_match
            '''
            match_method = 'simstring'
            # Just follow the fuzzychinese methods, no need to set threshold, just use 0.01
            all_dict_list = []
            # title_dict = {x[0]:x[1] for x in title_list}
            # for_test = list(title_dict.keys())
            for i in range(0,len(partner_list_for_match),500):
                res_dict = list_simstring(title_list, partner_list_for_match[i:min(i+500,len(partner_list_for_match))])
                all_dict_list.append(res_dict)
            # Initialize another return_dict
            res_dict = {}
            for aux in ["cos","over","dice","jac"]:
                res_dict[f"sim_{aux}_{task_name}_matched_word_1"] = []
                res_dict[f"sim_{aux}_{task_name}_matched_word_dist_1"] = []
                res_dict[f"sim_{aux}_{task_name}_matched_path_1"] = []

            for sim_list in all_dict_list:
                for aux in ["cos","over","dice","jac"]:
                    # The path should be retreived here based on the title dict, and add here using the aux
                    res_dict[f"sim_{aux}_{task_name}_matched_path_1"] = res_dict[f"sim_{aux}_{task_name}_matched_path_1"] + sim_list[f"sim_{aux}_nearest_img_path"]
                    res_dict[f"sim_{aux}_{task_name}_matched_word_1"] = res_dict[f"sim_{aux}_{task_name}_matched_word_1"] + sim_list[f"sim_{aux}_nearest"]
                    res_dict[f"sim_{aux}_{task_name}_matched_word_dist_1"] = res_dict[f"sim_{aux}_{task_name}_matched_word_dist_1"] + sim_list[f"sim_{aux}_nearest_dist"]
            # Can also break into several parts and concat
            # print(len(res_dict),len(df_matched))
            df_matched = pd.concat([
                partner_csv,
                pd.DataFrame(res_dict)
            ],axis=1)
            # Save the results
            df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_{match_method}.csv'))

            '''
            Here starts the accuracy calculation 
            '''
            for aux in ["cos","over","dice","jac"]:
                df_matched["matched_tk_path"] = df_matched[f"sim_{aux}_{task_name}_matched_path_1"]
                df_matched["source"] = df_matched["partner_path"]
                df_matched["distance"] = df_matched[f'sim_{aux}_{task_name}_matched_word_dist_1']
                df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{aux}_{task_name}_simstring.csv'))
                
                df_matched_for_nomatch = copy.deepcopy(df_matched)

                ## For simstring, within each method, we need to tune a bit
                if 'pr' not in task_name:
                    accuracy_dict[f"{task_name}_{match_method}_{aux}_match"] = calculate_matched_accuracy(matched_results = df_matched)
                    print('matched test accuracy:', calculate_matched_accuracy(matched_results = df_matched))
                
                else:
                    accuracy_dict[f"{task_name}_{match_method}_{aux}_match"] = calculate_pr_matched_accuracy(matched_results = df_matched)
                    print('matched test accuracy:', calculate_pr_matched_accuracy(matched_results = df_matched))


            with open(os.path.join(args.save_output,'japan_task_accuracy.json'),'w') as f:
                json.dump(accuracy_dict,f,ensure_ascii=False)
            ### Let's not worry about the no match accuracy first...

            # if "gcv" in task_name:
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_gcv_2_gcvtk_fuzzychinese_stroke.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_stroke_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_gcv_2_gcvtk_fuzzychinese_stroke.csv", levenshtein_match = False)
            # else:        
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_eff_2_efftk_fuzzychinese_stroke.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_stroke_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_eff_2_efftk_fuzzychinese_stroke.csv", levenshtein_match = False)
            # with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/accuracy_check.json','w') as f:
            #     json.dump(accuracy_dict,f,ensure_ascii=False)
                
        if args.fuzzychinese_stroke:
            match_method = 'fuzzychinese'
            raw_word = partner_list_for_match # This is the for matching
            title_dict = {x[0]:x[1] for x in title_list}
            for_test = list(title_dict.keys())
            test_dict = pd.Series(for_test) # The two lists are in the same length 
            # Divide it into several epochs...
            all_dict_list = []
            start_time = time.time()
            for i in range(0,len(raw_word),5000): # have to chunk it otherwise it get too big
                res = list_fuzzyChinese(raw_word[i:min(i+5000,len(raw_word))],test_dict,title_dict,task_name)
                res.to_csv(os.path.join(args.save_output,f'fuzzy_{i}.csv'))
                all_dict_list.append(res)

            time_span = time.time()-start_time
            store_time[f"{task_name}_fuzzychinese"] = time_span
            with open(os.path.join(args.save_output,'time_speed.json'),'w') as f:
                json.dump(store_time, f, ensure_ascii=False)

            for id, res in enumerate(all_dict_list):
                if id == 0:
                    df_match_result = res
                else:
                    df_match_result = pd.concat([df_match_result,res])
            # df_matched.to_csv(os.path.join(args.save_outputx,f'matched.csv'))
            df_match_result.reset_index(drop = True,inplace = True)
            df_matched = pd.concat([df_match_result,partner_csv],axis=1)

            for ana in ["stroke","char"]:
                df_matched["matched_tk_path"] = df_matched[f"fuzzychinese_{ana}_{task_name}_matched_path_1"]
                df_matched["source"] = df_matched["partner_path"]
                df_matched["distance"] = df_matched[f'fuzzychinese_{ana}_{task_name}_word_dist_1']
                df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{ana}_{task_name}_{match_method}.csv'))
                
                df_matched_for_nomatch = copy.deepcopy(df_matched)

                if 'pr' not in task_name:
                    accuracy_dict[f"{task_name}_{match_method}_{ana}_match"] = calculate_matched_accuracy(matched_results = df_matched)
                    print('matched test accuracy:', calculate_matched_accuracy(matched_results = df_matched))
                
                else:
                    accuracy_dict[f"{task_name}_{match_method}_{ana}_match"] = calculate_pr_matched_accuracy(matched_results = df_matched)
                    print('matched test accuracy:', calculate_pr_matched_accuracy(matched_results = df_matched))
           
            # if "gcv" in task_name:
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_gcv_2_gcvtk_fuzzychinese_stroke.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_stroke_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_gcv_2_gcvtk_fuzzychinese_stroke.csv", levenshtein_match = False)
            # else:        
            #     print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_eff_2_efftk_fuzzychinese_stroke.csv", levenshtein_match = False))
            #     accuracy_dict[f"{task_name}_stroke_nomatch"] = calculate_nomatch_accuracy(matched_results = df_matched_for_nomatch, file_name = "df_full_matched_eff_2_efftk_fuzzychinese_stroke.csv", levenshtein_match = False)
            
            print(accuracy_dict)
            with open(os.path.join(args.save_output,'japan_task_accuracy.json'),'w') as f:
                json.dump(accuracy_dict,f,ensure_ascii=False)
