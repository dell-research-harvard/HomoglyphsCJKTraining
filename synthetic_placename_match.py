# %%
# Import dependencies
import json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import pickle
from hyperopt import hp
import matplotlib.pyplot as plt
import random
from hyperopt import rand, tpe
from multiprocessing import Pool
import faiss
import heapq
from functools import partial
import argparse
import numpy as np
import textdistance as td
from fuzzychinese import FuzzyChineseMatch
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.measure.jaccard import JaccardMeasure
from simstring.measure.overlap import OverlapMeasure
from simstring.measure.dice import DiceMeasure
from simstring.searcher import Searcher
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
import heapq

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
                if str1[i-1] in cluster_dict:
                    if str2[j-1] in cluster_dict[str1[i-1]]:
                        dist=1*(1-cluster_dict[str1[i-1]][str2[j-1]]) # This is gamma actually, the substitution cost is the homoglyphic distance
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

def map_2_dist(x,dist_list):
    return float(dist_list[x])

def map_2_word(x,list2):
    return str(list2[x])


def map_2_word_dist(x,list2,dist_list):
    return [str(list2[x]),float(dist_list[x])]

def list_fd(word,list2):
    # try not to pass many things in pool, since it will copy everything, make things as lean as possible!
    dist_list = np.ones(len(list2))
    smallest_dist_10 = np.full((1,10),999) # Set the num to 999
    #print(list2)
    for id, word2 in tqdm(enumerate(list2)): # The returned value will keep the order as original
        #print(type(word2))# This is normal and right
        #print(type(word))
        if abs(len(str(word2))-len(str(word))) > np.max(smallest_dist_10):
            dist = 1000
        dist = custom_edit_distance(str(word),str(word2))
        #print(dist)
        dist_list[id] = dist
        if dist < np.max(smallest_dist_10):# If smaller than the 5 neighbors
            smallest_dist_10[np.argmax(smallest_dist_10)] = dist

    idx = np.argpartition(dist_list,10)

    #min_dist_10_list = list(map(lambda x:map_2_dist(x,dist_list),idx[:10]))
    #float(np.min(dist_list)) # Change the distance to Python native float type
    #min_dist_word_10_list = list(map(lambda x:map_2_word(x,list2),idx[:10])) # Which word in the ground truth dict get matched to

    word_dist = sorted(list(map(lambda x:map_2_word_dist(x,list2,dist_list),idx[:10])),key = lambda x:x[1])

    return word_dist
    #eturn [min_dist_10_list, min_dist_word_10_list]

def list_lev(word,list2):
    dist_list = np.ones(len(list2))
    smallest_dist_10 = np.full((1,10),999) # Set the num to 999
    #print(list2)
    for id, word2 in tqdm(enumerate(list2)): # The returned value will keep the order as original
        #print(type(word2))# This is normal and right
        #print(type(word))
        if abs(len(str(word2))-len(str(word))) > np.max(smallest_dist_10):
            dist = 1000
        dist = td.levenshtein(str(word),str(word2))
        #print(dist)
        dist_list[id] = dist
        if dist < np.max(smallest_dist_10):# If smaller than the 5 neighbors
            smallest_dist_10[np.argmax(smallest_dist_10)] = dist

    idx = np.argpartition(dist_list,10)

    #min_dist_10_list = list(map(lambda x:map_2_dist(x,dist_list),idx[:10]))
    #float(np.min(dist_list)) # Change the distance to Python native float type
    #min_dist_word_10_list = list(map(lambda x:map_2_word(x,list2),idx[:10])) # Which word in the ground truth dict get matched to

    word_dist = sorted(list(map(lambda x:map_2_word_dist(x,list2,dist_list),idx[:10])),key = lambda x:x[1])

    return word_dist


def list_simstring(list2, partners):
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    #print(list2)
    for title in list2:
        db.add(str(title))
        
    res_dict = {}
    thresh=0.01
    searcher_cos = Searcher(db, CosineMeasure())
    searcher_over = Searcher(db, OverlapMeasure())
    searcher_dice = Searcher(db, DiceMeasure())
    searcher_jac = Searcher(db, JaccardMeasure())

    mapper={"cos":searcher_cos,"over":searcher_over,"dice":searcher_dice,"jac":searcher_jac}

    #for aux in tqdm(["cos","over","dice","jac"]):
    for aux in tqdm(["cos","over","dice","jac"]):
        searcher=mapper[aux]
        nearest_1, dist_1, nearest_2, dist_2, nearest_3, dist_3, nearest_4, dist_4, nearest_5, dist_5, nearest_6, dist_6, nearest_7, dist_7, nearest_8, dist_8, nearest_9, dist_9, nearest_10, dist_10 = ([] for i in range(20))

        for partner in tqdm(partners):
            results=searcher.ranked_search(str(partner),0.01)
            #print(list(results.items())[:10])
            for sort_id in range(1,11):
                try:
                    locals()['nearest_'+str(sort_id)].append(list(results.items())[sort_id-1][0]) # Whether it can return top 10, it is ordered_dict, sorted well
                    locals()['dist_'+str(sort_id)].append(list(results.items())[sort_id-1][1])
                except:
                    locals()['nearest_'+str(sort_id)].append('')
                    locals()['dist_'+str(sort_id)].append(1)
        
        for sort_id2 in range(1,11):
            res_dict[f'sim_{aux}_nearest_{sort_id2}'] = locals()['nearest_'+str(sort_id2)]
            res_dict[f'sim_{aux}_nearest_dist_{sort_id2}'] = locals()['dist_'+str(sort_id2)]

    return res_dict 

def list_fuzzyChinese(raw_word,test_dict):# raw_word is the partner list/list 1, test_dict is the  title list/list 2
    #return all the results, just pass in two lists, don't pass in too many other things.
    '''
    You should actually delete task_name, but there is nothing here
    Input: test_dict, raw_word
    Output: nearest neighbor word list, nearest neighbor dist list  
    '''
    # def map_dict(a):
    #     return title_dict[a]

    fcm = FuzzyChineseMatch(ngram_range=(3,3),analyzer="stroke")
    fcm.fit(test_dict)
    top1_similar_stroke = fcm.transform(raw_word,n=10) # This is the nearest neighbor list - return top 10 similar


    fcm_char = FuzzyChineseMatch(ngram_range=(3,3),analyzer="char")
    fcm_char.fit(test_dict)
    top1_similar_char = fcm_char.transform(raw_word,n=10)

    res = pd.concat([
        pd.DataFrame(top1_similar_stroke,columns=[f'fuzzychinese_stroke_word_1', f'fuzzychinese_stroke_word_2',f'fuzzychinese_stroke_word_3', \
            f'fuzzychinese_stroke_word_4',f'fuzzychinese_stroke_word_5',f'fuzzychinese_stroke_word_6',f'fuzzychinese_stroke_word_7', \
                f'fuzzychinese_stroke_word_8',f'fuzzychinese_stroke_word_9',f'fuzzychinese_stroke_word_10']),
        pd.DataFrame(fcm.get_similarity_score(),columns=[f'fuzzychinese_stroke_word_dist_1', f'fuzzychinese_stroke_word_dist_2',f'fuzzychinese_stroke_word_dist_3', \
            f'fuzzychinese_stroke_word_dist_4',f'fuzzychinese_stroke_word_dist_5',f'fuzzychinese_stroke_word_dist_6',f'fuzzychinese_stroke_word_dist_7', \
                f'fuzzychinese_stroke_word_dist_8',f'fuzzychinese_stroke_word_dist_9',f'fuzzychinese_stroke_word_dist_10']), \
        pd.DataFrame(top1_similar_char,columns=[f'fuzzychinese_char_word_1', f'fuzzychinese_char_word_2',f'fuzzychinese_char_word_3', \
            f'fuzzychinese_char_word_4',f'fuzzychinese_char_word_5',f'fuzzychinese_char_word_6',f'fuzzychinese_char_word_7', \
                f'fuzzychinese_char_word_8',f'fuzzychinese_char_word_9',f'fuzzychinese_char_word_10']),
        pd.DataFrame(fcm_char.get_similarity_score(),columns=[f'fuzzychinese_char_word_dist_1', f'fuzzychinese_char_word_dist_2',f'fuzzychinese_char_word_dist_3', \
            f'fuzzychinese_char_word_dist_4',f'fuzzychinese_char_word_dist_5',f'fuzzychinese_char_word_dist_6',f'fuzzychinese_char_word_dist_7', \
                f'fuzzychinese_char_word_dist_8',f'fuzzychinese_char_word_dist_9',f'fuzzychinese_char_word_dist_10']), \
    ],axis = 1)

    # for i in range(1,11):
    #     res[f"fuzzychinese_char_{task_name}_matched_path_{i}"]=res.apply(lambda x:map_dict(x[f'fuzzychinese_char_{task_name}_matched_word_{i}']),axis=1)
    #     res[f"fuzzychinese_stroke_{task_name}_matched_path_{i}"]=res.apply(lambda x:map_dict(x[f'fuzzychinese_stroke_{task_name}_matched_word_{i}']),axis=1)

    return res
def same_matched(a,b):
    for ele in a:
        if ele == b: # If any of the ele in a equals b, return 1, after the iter, if nothing returns, just return 0
            return 1
    return 0

# %%
# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_entry", type=list, 
        default=["result","ground_truth"],#not necessary 
        help="The csv entry for use")
    parser.add_argument("--lang", type=str, 
        help="Choose from ja, zh, ko")
    parser.add_argument("--multilang_dir", type=str, 
        help="Choose from multilang_results or multilang_results_gcv")
    # Add optional comparisons here
    parser.add_argument("--homo", action="store_true", default=False, 
        help="Run homoglyph matching")
    parser.add_argument("--lev", action="store_true", default=False, 
        help="Compare to Levenstein Distance")
    parser.add_argument("--simstring", action="store_true", default=False, 
        help="Compare to simstring")
    parser.add_argument("--fuzzychinese", action="store_true", default=False, 
        help="Compare to fuzzychinese")
    parser.add_argument("--save_output", type=str, required=True, 
        help="Save output!")
    args = parser.parse_args()

    #abbrev_2_lang = {'zhs_80000':'s_chinese_expanded_easy','zhs_easy_paddle_80000':'s_chinese_expanded_easy','zhs_gcv_80000':'s_chinese_expanded_gcv','zhs_easy_80000':'s_chinese_expanded_easy','zhs_gcv_paddle_80000':'s_chinese_expanded_gcv','zhs_gcv_paddle_130047':'s_chinese_expanded_gcv','zhs_gcv_paddle_70091':'s_chinese_expanded_gcv','zht_paddle_easy':'t_chinese_expanded_easy','zh_paddle_easy':'s_chinese_expanded_easy','ja_paddle_easy':'japanese','ko_paddle_easy':'korean','zht_gcv_paddle_70091':'t_chinese_expanded_gcv','zht_gcv_paddle':'t_chinese_expanded_gcv','zhs_gcv_paddle':'s_chinese_expanded_gcv','ja_easy':'japanese','ko_easy':'korean','zhs_easy':'chinese','zht_easy':'t_chinese_expanded_easy','ja':'ja_expanded_easy','zh':'chinese','ko':'ko_expanded_easy','zht':'t_chinese_expanded_easy','zhs_paddle_on_gcv_image':'chinese','zht_paddle_on_gcv_image':'t_chinese','ja_paddle_on_gcv_image':'japanese','ko_paddle_on_gcv_image':'korean'}
    abbrev_2_lang = {'zhs_80000':'s_chinese_expanded_easy','zhs_easy_paddle_80000':'s_chinese_expanded_easy','zhs_gcv_80000':'s_chinese_expanded_gcv','zhs_easy_80000':'s_chinese_expanded_easy','zhs_gcv_paddle_80000':'s_chinese_expanded_gcv','zhs_gcv_paddle_130047':'s_chinese_expanded_gcv','zhs_gcv_paddle_70091':'s_chinese_expanded_gcv','zht_paddle_easy':'t_chinese_expanded_easy','zh_paddle_easy':'s_chinese_expanded_easy','ja_paddle_easy':'ja_expanded_easy','ko_paddle_easy':'ko_expanded_easy','zht_gcv_paddle_70091':'t_chinese_expanded_gcv','zht_gcv_paddle':'t_chinese_expanded_gcv','zhs_gcv_paddle':'s_chinese_expanded_gcv','ja_easy':'ja_expanded_easy','ko_easy':'ko_expanded_easy','zhs_easy':'chinese','zht_easy':'t_chinese_expanded_easy','ja':'ja_expanded_easy','zh':'chinese','ko':'ko_expanded_easy','zht':'t_chinese_expanded_easy','zhs_paddle_on_gcv_image':'chinese','zht_paddle_on_gcv_image':'t_chinese','ja_paddle_on_gcv_image':'japanese','ko_paddle_on_gcv_image':'korean'}
    pickle_name = abbrev_2_lang[args.lang]
    with open(f"/mnt/data01/homo/cjk_homoglyphs/char_char_dist_dict_800_{pickle_name}.pickle",'rb') as f:
        cluster_dict = pickle.load(f)

    # Convert csv to json list
    df = pd.read_csv(f"/mnt/data01/yxm/homo/{args.multilang_dir}/{args.lang}/error_df.csv")

    # Initialize the list 2
    try:
        list2 = df[args.csv_entry[1]].tolist()
    except:
        list2 = df['truth'].tolist()

    # Initialize the list 1
    result_list = df[args.csv_entry[0]].tolist() # The result_list is list 1
    assert len(list2) == len(result_list)

    list1 = [] # initialize the list1
    for res, truth in zip(result_list,list2):
        res_dict = {}
        res_dict["result"] = res
        res_dict["truth"] = truth
        list1.append(res_dict)
        
    # Save output and initialize the accuracy results storage
    os.makedirs(args.save_output, exist_ok=True)
    stored_accuracy = {} # Initialize the stored_accuracy
    for choice in ["homoglyph","lev","simstring_cos","simstring_dice","simstring_over","simstring_jac","fuzzychinese_stroke","fuzzychinese_char"]:
        stored_accuracy[choice] = {}
    ## add the df_matched
    df_matched = pd.DataFrame(list(zip(list2,result_list)), columns=['truth','result'])

    if args.homo:
        method_name = 'homo'
        with Pool(4) as p:
            word_dist_min_list = p.map(partial(list_fd,list2=list2),result_list)
        matched_list = []
        distance_list = []
        for id, (list1_ele, word_dist_min) in enumerate(zip(list1,word_dist_min_list)):
            #print(word_dist_min)
            list1[id]["matched_word"] = word_dist_min
            matched_list.append(word_dist_min)
            # list1[id]["matched_word_dist"] = word_dist_min[0]
            # distance_list.append(word_dist_min[0])

        with open(os.path.join(args.save_output,f'matched_{args.lang}_{method_name}.json'),'w') as f:
            json.dump(list1,f,ensure_ascii=False)
        
        matched_list_1, distance_list_1, matched_list_2, distance_list_2, matched_list_3, distance_list_3, matched_list_4, distance_list_4, matched_list_5, distance_list_5, matched_list_6, distance_list_6, matched_list_7, distance_list_7, matched_list_8, distance_list_8, matched_list_9, distance_list_9, matched_list_10, distance_list_10 = ([] for i in range(20))

        for result_homo in list1:
            for word_id, word in enumerate(result_homo["matched_word"]):
                locals()['matched_list_'+str(word_id+1)].append(word[0])
                locals()['distance_list_'+str(word_id+1)].append(word[1])

        # The result list is the OCRed text, the list2 is the ground truth text for matching
        df_matched = pd.DataFrame(list(zip(result_list,list2, \
            matched_list_1, distance_list_1, \
                matched_list_2, distance_list_2, \
                    matched_list_3, distance_list_3, \
                        matched_list_4, distance_list_4, \
                            matched_list_5, distance_list_5, \
                                matched_list_6, distance_list_6, \
                                    matched_list_7, distance_list_7, \
                                        matched_list_8, distance_list_8, \
                                            matched_list_9, distance_list_9, \
                                                matched_list_10, distance_list_10, )), \
                    columns=['result', 'truth', \
                    'homo_matched_1','homo_dist_1', \
                    'homo_matched_2','homo_dist_2', \
                    'homo_matched_3','homo_dist_3', \
                    'homo_matched_4','homo_dist_4', \
                    'homo_matched_5','homo_dist_5', \
                    'homo_matched_6','homo_dist_6', \
                    'homo_matched_7','homo_dist_7', \
                    'homo_matched_8','homo_dist_8', \
                    'homo_matched_9','homo_dist_9', \
                    'homo_matched_10','homo_dist_10', \
                                            ])
        
        print('number of data',len(df_matched))
        df_matched_small = df_matched.dropna(subset=["result"])
        print('number of data after refinement',len(df_matched_small))
        df_matched_small.to_csv(os.path.join(args.save_output,f'df_full_matched_small_{method_name}.csv'))

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["homo_matched_1"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["homoglyph"]["top 1"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["homo_matched_1"],x["homo_matched_2"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["homoglyph"]["top 2"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["homo_matched_1"],x["homo_matched_2"],x["homo_matched_3"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["homoglyph"]["top 3"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["homo_matched_1"],x["homo_matched_2"],x["homo_matched_3"],x["homo_matched_4"],x["homo_matched_5"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["homoglyph"]["top 5"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["homo_matched_1"],x["homo_matched_2"],x["homo_matched_3"],x["homo_matched_4"],x["homo_matched_5"],x["homo_matched_6"],x["homo_matched_7"],x["homo_matched_8"],x["homo_matched_9"],x["homo_matched_10"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["homoglyph"]["top 10"] =  accuracy
        
        df_matched_small.to_csv(os.path.join(args.save_output,f'df_full_matched_small_{method_name}.csv'))
        with open(os.path.join(args.save_output,f'accuracy_{args.lang}_small.json'),'w') as f:
            json.dump(stored_accuracy,f,ensure_ascii=False)
        print(stored_accuracy)
    # Levenshtein comparison!
    if args.lev:
        method_name = 'lev'
        print('Levenshtein comparison!')
        with Pool(32) as p:
            word_dist_min_list = p.map(partial(list_lev,list2=list2),result_list)
        # Save output
        matched_list = []
        distance_list = []
        for id, (list1_ele, word_dist_min) in enumerate(zip(list1,word_dist_min_list)):
            list1[id]["matched_word_lev"] = word_dist_min
            matched_list.append(word_dist_min)

        with open(os.path.join(args.save_output,f'matched_{args.lang}_{method_name}.json'),'w') as f:
            json.dump(list1,f,ensure_ascii=False)
        
        matched_list_1, distance_list_1, matched_list_2, distance_list_2, matched_list_3, distance_list_3, matched_list_4, distance_list_4, matched_list_5, distance_list_5, matched_list_6, distance_list_6, matched_list_7, distance_list_7, matched_list_8, distance_list_8, matched_list_9, distance_list_9, matched_list_10, distance_list_10 = ([] for i in range(20))

        for result_homo in list1:
            for word_id, word in enumerate(result_homo[f"matched_word_{method_name}"]): # Lev results are in matched_word_lev
                locals()['matched_list_'+str(word_id+1)].append(word[0])
                locals()['distance_list_'+str(word_id+1)].append(word[1])

        # The result list is the OCRed text, the list2 is the ground truth text for matching
        df_matched = pd.DataFrame(list(zip(result_list,list2, \
            matched_list_1, distance_list_1, \
                matched_list_2, distance_list_2, \
                    matched_list_3, distance_list_3, \
                        matched_list_4, distance_list_4, \
                            matched_list_5, distance_list_5, \
                                matched_list_6, distance_list_6, \
                                    matched_list_7, distance_list_7, \
                                        matched_list_8, distance_list_8, \
                                            matched_list_9, distance_list_9, \
                                                matched_list_10, distance_list_10, )), \
                    columns=['result', 'truth', \
                    'lev_matched_1','lev_dist_1', \
                    'lev_matched_2','lev_dist_2', \
                    'lev_matched_3','lev_dist_3', \
                    'lev_matched_4','lev_dist_4', \
                    'lev_matched_5','lev_dist_5', \
                    'lev_matched_6','lev_dist_6', \
                    'lev_matched_7','lev_dist_7', \
                    'lev_matched_8','lev_dist_8', \
                    'lev_matched_9','lev_dist_9', \
                    'lev_matched_10','lev_dist_10', \
                                            ])
        
        # Restrict to the results are not empty
        # Please use a separate thing here ...
        print(len(df_matched))
        df_matched_small = df_matched.dropna(subset=["result"])
        #print(df_matched["result"])
        #df_matched = df_matched[df_matched["result"].str.contains('')==False]
        print(len(df_matched_small))
        df_matched_small.to_csv(os.path.join(args.save_output,'df_full_matched_small_lev.csv'))
        
        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["lev_matched_1"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["lev"]["top 1"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["lev_matched_1"],x["lev_matched_2"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["lev"]["top 2"] =  accuracy

        # Store the accuracy, for top 3 just change it to any of them are the same as truth
        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["lev_matched_1"],x["lev_matched_2"],x["lev_matched_3"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["lev"]["top 3"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["lev_matched_1"],x["lev_matched_2"],x["lev_matched_3"],x["lev_matched_4"],x["lev_matched_5"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["lev"]["top 5"] =  accuracy

        df_matched_small["accuracy"]=df_matched_small.apply(lambda x:same_matched([x["lev_matched_1"],x["lev_matched_2"],x["lev_matched_3"],x["lev_matched_4"],x["lev_matched_5"],x["lev_matched_6"],x["lev_matched_7"],x["lev_matched_8"],x["lev_matched_9"],x["lev_matched_10"]],x["truth"]),axis=1)#The accuracy do not need to be stored
        accuracy=df_matched_small["accuracy"].mean() # Store these into a json file
        stored_accuracy["lev"]["top 10"] =  accuracy

        df_matched_small.to_csv(os.path.join(args.save_output,f'df_full_matched_small_{method_name}.csv'))
        with open(os.path.join(args.save_output,f'accuracy_{args.lang}_small.json'),'w') as f:
            json.dump(stored_accuracy,f,ensure_ascii=False)
        print(stored_accuracy)

    # Simstring comparison! (cosine measure, Dice, overlap, Jaccard: 2-grams)
    if args.simstring:
        method_name = 'simstring'
        partners = df_matched["result"].fillna('').astype(str)
        all_dict_list = []
        for i in range(0,len(partners),25):
            res_dict = list_simstring(list2, partners[i:min(i+25,len(partners))])
            all_dict_list.append(res_dict)
        # Initialize another return_dict
        res_dict = {}
        #for aux in ["cos","over","dice","jac"]:
        for aux in ["cos","over","dice","jac"]:
            for sort_id in range(1,11):
                res_dict[f"sim_{aux}_nearest_{sort_id}"] = []
                res_dict[f"sim_{aux}_nearest_dist_{sort_id}"] = []

        for sim_list in all_dict_list:
            for aux in ["cos","over","dice","jac"]:
                for sort_id in range(1,11):
                    res_dict[f"sim_{aux}_nearest_{sort_id}"] = res_dict[f"sim_{aux}_nearest_{sort_id}"] + sim_list[f"sim_{aux}_nearest_{sort_id}"]
                    res_dict[f"sim_{aux}_nearest_dist_{sort_id}"] = res_dict[f"sim_{aux}_nearest_dist_{sort_id}"] + sim_list[f"sim_{aux}_nearest_dist_{sort_id}"]

        # Can also break into several parts and concat
        df_matched = pd.concat([
            df_matched,
            pd.DataFrame(res_dict)
        ],axis=1)
        
        df_matched_small = df_matched.dropna(subset=["result"])

        for aux in ["cos","over","dice","jac"]:
            df_matched_small[f"accuracy_{aux}"]=df_matched_small.apply(lambda x:same_matched([x[f"sim_{aux}_nearest_1"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{aux}"].mean() # Store these into a json file
            stored_accuracy[f"simstring_{aux}"]["top 1"] =  accuracy

            df_matched_small[f"accuracy_{aux}"]=df_matched_small.apply(lambda x:same_matched([x[f"sim_{aux}_nearest_1"],x[f"sim_{aux}_nearest_2"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{aux}"].mean() # Store these into a json file
            stored_accuracy[f"simstring_{aux}"]["top 2"] =  accuracy

            df_matched_small[f"accuracy_{aux}"]=df_matched_small.apply(lambda x:same_matched([x[f"sim_{aux}_nearest_1"],x[f"sim_{aux}_nearest_2"],x[f"sim_{aux}_nearest_3"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{aux}"].mean() # Store these into a json file
            stored_accuracy[f"simstring_{aux}"]["top 3"] =  accuracy

            df_matched_small[f"accuracy_{aux}"]=df_matched_small.apply(lambda x:same_matched([x[f"sim_{aux}_nearest_1"],x[f"sim_{aux}_nearest_2"],x[f"sim_{aux}_nearest_3"],x[f"sim_{aux}_nearest_4"],x[f"sim_{aux}_nearest_5"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{aux}"].mean() # Store these into a json file
            stored_accuracy[f"simstring_{aux}"]["top 5"] =  accuracy

            df_matched_small[f"accuracy_{aux}"]=df_matched_small.apply(lambda x:same_matched([x[f"sim_{aux}_nearest_1"],x[f"sim_{aux}_nearest_2"],x[f"sim_{aux}_nearest_3"],x[f"sim_{aux}_nearest_4"],x[f"sim_{aux}_nearest_5"],x[f"sim_{aux}_nearest_6"],x[f"sim_{aux}_nearest_7"],x[f"sim_{aux}_nearest_8"],x[f"sim_{aux}_nearest_9"],x[f"sim_{aux}_nearest_10"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{aux}"].mean() # Store these into a json file
            stored_accuracy[f"simstring_{aux}"]["top 10"] =  accuracy
        print(stored_accuracy)
        df_matched_small.to_csv(os.path.join(args.save_output,f'df_full_matched_small_{method_name}.csv'))
        with open(os.path.join(args.save_output,f'accuracy_{args.lang}_{method_name}.json'),'w') as f:
            json.dump(stored_accuracy,f,ensure_ascii=False)   
            
    # FuzzyChinese comparison! (Stroke, Char)
    if args.fuzzychinese:
        method_name = 'fuzzychinese'
        raw_word = df_matched["result"].fillna('') # This is the for matching
        test_dict = pd.Series(list2) # The two lists are in the same length 
        raw_word = result_list # This is the for matching
        all_dict_list = []
        # The length of the list is len(list2)
        #raw_word = raw_word[:10]
        for i in range(0,len(raw_word),500): 
            res = list_fuzzyChinese(raw_word[i:min(i+500,len(raw_word))],test_dict)
            res.to_csv(os.path.join(args.save_output,f'check_{method_name}_{i}.csv'))
            all_dict_list.append(res)

        for id, res in enumerate(all_dict_list):
            if id == 0:
                df_matched = res
            else:
                df_matched = pd.concat([df_matched,res])
        df_matched.to_csv(os.path.join(args.save_output,f'matched.csv'))
        df_matched.reset_index(drop = True,inplace = True)

        df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{method_name}.csv'))
        df_matched["truth"] = list2
        df_matched["result"] = raw_word
        df_matched_small = df_matched.dropna(subset=["result"])
        # Store the accuracy to the accuracy json
        for mode in ["stroke","char"]:
            df_matched_small[f"accuracy_{mode}"]=df_matched_small.apply(lambda x:same_matched([x[f"fuzzychinese_{mode}_word_1"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{mode}"].mean() # Store these into a json file
            stored_accuracy[f"fuzzychinese_{mode}"]["top 1"] =  accuracy
    
            df_matched_small[f"accuracy_{mode}"]=df_matched_small.apply(lambda x:same_matched([x[f"fuzzychinese_{mode}_word_1"],x[f"fuzzychinese_{mode}_word_2"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{mode}"].mean() # Store these into a json file
            stored_accuracy[f"fuzzychinese_{mode}"]["top 2"] =  accuracy
            
            df_matched_small[f"accuracy_{mode}"]=df_matched_small.apply(lambda x:same_matched([x[f"fuzzychinese_{mode}_word_1"],x[f"fuzzychinese_{mode}_word_2"],x[f"fuzzychinese_{mode}_word_3"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{mode}"].mean() # Store these into a json file
            stored_accuracy[f"fuzzychinese_{mode}"]["top 3"] =  accuracy

            df_matched_small[f"accuracy_{mode}"]=df_matched_small.apply(lambda x:same_matched([x[f"fuzzychinese_{mode}_word_1"],x[f"fuzzychinese_{mode}_word_2"],x[f"fuzzychinese_{mode}_word_3"],x[f"fuzzychinese_{mode}_word_4"],x[f"fuzzychinese_{mode}_word_5"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{mode}"].mean() # Store these into a json file
            stored_accuracy[f"fuzzychinese_{mode}"]["top 5"] =  accuracy

            df_matched_small[f"accuracy_{mode}"]=df_matched_small.apply(lambda x:same_matched([x[f"fuzzychinese_{mode}_word_1"],x[f"fuzzychinese_{mode}_word_2"], \
                x[f"fuzzychinese_{mode}_word_3"],x[f"fuzzychinese_{mode}_word_4"],x[f"fuzzychinese_{mode}_word_5"], \
                    x[f"fuzzychinese_{mode}_word_6"],x[f"fuzzychinese_{mode}_word_7"],x[f"fuzzychinese_{mode}_word_8"], \
                        x[f"fuzzychinese_{mode}_word_9"],x[f"fuzzychinese_{mode}_word_10"]],x["truth"]),axis=1)#The accuracy do not need to be stored
            accuracy=df_matched_small[f"accuracy_{mode}"].mean() # Store these into a json file
            stored_accuracy[f"fuzzychinese_{mode}"]["top 10"] =  accuracy
        df_matched_small.to_csv(os.path.join(args.save_output,f'df_full_matched_small_{method_name}.csv'))
    print("matching accuracy",stored_accuracy)

    # Save the accuracy results
    with open(os.path.join(args.save_output,f'accuracy_{args.lang}_small.json'),'w') as f:
        json.dump(stored_accuracy,f,ensure_ascii=False)
