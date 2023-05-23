# %%
'''
Let's get all accuracy results here. It should include
1. Two versions of accuracy is needed - maybe a separate dictionary to keep the number of valid and exact matches
2. Separately show on synthetic data and real Japanese data
Paddle to Paddle Accuracy - Two Versions

Paddle to GCV Accuracy - both the previous dict and current dict

Easy to Easy Accuracy - waiting for the last one

GCV to GCV Accuracy - zht and Japanese

Paddle to Easy Accuracy - Running

In this script, you can also add paddle to paddle, easy to easy...

'''
import pandas as pd
import os
import json
from tqdm import tqdm
# Let's do a new folder... on the old dicts to synth results first

# This is no use
#file_name = ['df_full_matched_small_fuzzychinese.csv','df_full_matched_small_homo.csv','df_full_matched_small_lev.csv','df_full_matched_small_simstring.csv']

Easy_2_GT_ZHT_FOLDER_old_dict = '/mnt/data01/yxm/homoglyphic_matching/multi_lang/c_j_k_match_top_10_zht_easy'

## There is no new dict for this
# Paddle_2_GCV_ZHT_FOLDER_new_expanded_dict = '/mnt/data01/yxm/homoglyphic_matching/multi_lang/zht_gcv_paddle_homo_expanded_dict'

# This is the not good results
Easy_2_GT_ZHS_FOLDER_old_dict = '/mnt/data01/yxm/homoglyphic_matching/multi_lang/c_j_k_match_top_10_zhs_easy_80000_expand'# Not finished yet...

Easy_2_GT_JA_FOLDER = '/mnt/data01/yxm/homoglyphic_matching/multi_lang/c_j_k_match_top_10_ja_easy'

Easy_2_GT_KO_FOLDER = '/mnt/data01/yxm/homoglyphic_matching/multi_lang/c_j_k_match_top_10_ko_easy'

save_output = './aggregate_results_easy_2_GT_final'
os.makedirs(save_output,exist_ok=True)

'''
Let's exclude the exact matches first, only use the valid error df

Let's do a paddle to gcv series first... this is in the gcv folder

If you want to calculate several versions of accuracy, you need to keep the full df, valid df, and the error_df for match, so you can know everything

df_full, df_valid

'''
# same_matched function
def same_matched(a,b):
    for ele in a:
        if ele == b: # If any of the ele in a equals b, return 1, after the iter, if nothing returns, just return 0
            return 1
    return 0

# Let's define a function for calculate accuracy
def cal_acc(df_matched_small, method_name,stored_accuracy, aux=None):
    if aux==None:
        accuracy_name = 'accuracy'
        prefix = f"{method_name}_matched"
    
    elif method_name=="sim":
        assert aux!=None
        accuracy_name = f'accuracy_{aux}'
        prefix = f"{method_name}_{aux}_nearest"
        method_name = f"{method_name}_{aux}"
    elif method_name=="fuzzychinese":
        assert aux!=None
        accuracy_name = f'accuracy_{aux}'
        prefix = f"{method_name}_{aux}_word"    
        method_name = f"{method_name}_{aux}"    

    df_matched_small[accuracy_name]=df_matched_small.apply(lambda x:same_matched([x[f"{prefix}_1"]],x["truth"]),axis=1)#The accuracy do not need to be stored
    accuracy=df_matched_small[accuracy_name].sum() # Store these into a json file
    stored_accuracy[method_name]["top 1"] =  int(accuracy)

    df_matched_small[accuracy_name]=df_matched_small.apply(lambda x:same_matched([x[f"{prefix}_1"],x[f"{prefix}_2"]],x["truth"]),axis=1)#The accuracy do not need to be stored
    accuracy=df_matched_small[accuracy_name].sum() # Store these into a json file
    stored_accuracy[method_name]["top 2"] =  int(accuracy)

    df_matched_small[accuracy_name]=df_matched_small.apply(lambda x:same_matched([x[f"{prefix}_1"],x[f"{prefix}_2"],x[f"{prefix}_3"]],x["truth"]),axis=1)#The accuracy do not need to be stored
    accuracy=df_matched_small[accuracy_name].sum() # Store these into a json file
    stored_accuracy[method_name]["top 3"] = int(accuracy)

    df_matched_small[accuracy_name]=df_matched_small.apply(lambda x:same_matched([x[f"{prefix}_1"],x[f"{prefix}_2"],x[f"{prefix}_3"],x[f"{prefix}_4"],x[f"{prefix}_5"]],x["truth"]),axis=1)#The accuracy do not need to be stored
    accuracy=df_matched_small[accuracy_name].sum() # Store these into a json file
    stored_accuracy[method_name]["top 5"] = int(accuracy)

    df_matched_small[accuracy_name]=df_matched_small.apply(lambda x:same_matched([x[f"{prefix}_1"],x[f"{prefix}_2"],x[f"{prefix}_3"],x[f"{prefix}_4"],x[f"{prefix}_5"],x[f"{prefix}_6"],x[f"{prefix}_7"],x[f"{prefix}_8"],x[f"{prefix}_9"],x[f"{prefix}_10"]],x["truth"]),axis=1)#The accuracy do not need to be stored
    accuracy=df_matched_small[accuracy_name].sum() # Store these into a json file
    stored_accuracy[method_name]["top 10"] =  int(accuracy)

    return stored_accuracy


lang_2_folder = {'zht_easy':Easy_2_GT_ZHT_FOLDER_old_dict,'zhs_easy_80000':Easy_2_GT_ZHS_FOLDER_old_dict,'ko_easy':Easy_2_GT_KO_FOLDER,'ja_easy':Easy_2_GT_JA_FOLDER}


#for lang,folder in tqdm(lang_2_folder.items()):


for lang,folder in {'zht_easy':Easy_2_GT_ZHT_FOLDER_old_dict}.items():

    stored_accuracy = {} # Initialize the stored_accuracy
    for choice in ["homo","lev","sim_cos","sim_dice","sim_over","sim_jac","fuzzychinese_stroke","fuzzychinese_char"]:
        stored_accuracy[choice] = {}

    for method_name in ['homo','lev','sim','fuzzychinese']:
        if method_name == 'sim':
            file_name = "simstring"
        else:
            file_name = method_name
        df_matched_small = pd.read_csv(os.path.join(folder,f'df_full_matched_small_{file_name}.csv'))
        df_matched_small=df_matched_small.dropna(subset=['result'])
        df_matched_small = df_matched_small[df_matched_small["result"]!=""]
        
        ## Only drop result, no need to drop truth

        # df_matched_small=df_matched_small.dropna(subset=['truth'])
        # df_matched_small = df_matched_small[df_matched_small["truth"]!=""]       

        # No need to drop NA again, everything inside df_matched_small is already dropped - You still need to drop NA, but it is weird for levenshtein
        print('after drop NA',len(df_matched_small))

        if method_name =="homo" or method_name=="lev": 
            stored_accuracy = cal_acc(df_matched_small,method_name, stored_accuracy)
        elif method_name=="sim":
            for aux in ["cos","over","dice","jac"]:
                stored_accuracy = cal_acc(df_matched_small,method_name, stored_accuracy,aux)
        elif method_name=="fuzzychinese":
            for aux in ["stroke","char"]:
                stored_accuracy = cal_acc(df_matched_small,method_name, stored_accuracy,aux)

        with open(os.path.join(save_output,f'accuracy_{lang}_small_count.json'),'w') as f:
            json.dump(stored_accuracy,f,ensure_ascii=False)
        print(stored_accuracy)
    
    # No need get from there...

    df_full = pd.read_csv(f'/mnt/data01/yxm/homo/multilang_results/{lang}/df_full.csv')

    try:
        df_full=df_full.drop_duplicates(subset=['ground_truth'])
    except:
        df_full=df_full.drop_duplicates(subset=['truth'])

    df_valid = df_full.dropna(subset=['result'])
    # df_valid = pd.read_csv(f'/mnt/data01/yxm/homo/multilang_results/{lang}/df_valid.csv')#There is no direct valid df available, you need to calculate from results.csv

    df_error = pd.read_csv(f'/mnt/data01/yxm/homo/multilang_results/{lang}/error_df.csv')

    # You can have more functionalities for creating several versions of accuracy...
    # We want to keep two versions of top 1 accuracy, let alone others for now

    '''
    Think About How to Store this Information for Confluence Report...

    Maybe two tables - for the accuracy, let's have three columns? only keep top 1

    accuracy for only error df, include exact match but no empty string, include everything include the empty string
    
    save this to the save_output folder
    '''
    name = ["Paddle to GCV"]
    total_images = [len(df_full)]

    total_valid_OCR = [len(df_valid)]

    print('df_error before',len(df_error))
    df_error = df_error.dropna(subset=['result'])
    # Also need to drop where ground_truth is empty! Maybe that's also why lev doesn't match the full...
    # df_error = df_error.dropna(subset=['ground_truth'])

    print('df_error_after',len(df_error))
    total_error_df = [len(df_error)]

    total_exact_match = [len(df_valid)-len(df_error)]

    total_empty_string = [len(df_full)-len(df_valid)]
    
    total_empty_string_perc = [(len(df_full)-len(df_valid))/len(df_full)]
    
    total_exact_match_perc = [(len(df_valid)-len(df_error))/len(df_full)]
    
    total_error_df_perc = [len(df_error)/len(df_full)]

    df_stats_save = pd.DataFrame(list(zip(name,total_images,total_valid_OCR,total_error_df,total_exact_match, total_empty_string, total_empty_string_perc, total_exact_match_perc, total_error_df_perc)),columns=['name','Total #images','Total #Valid OCR','Total #Error df','Total #Exact Match', 'Total Empty String', 'total empty string perc','total exact match perc','total error df perc'])

    df_stats_save.to_csv(os.path.join(save_output,f'data_stats_{lang}.csv'))
    # Also store the exact count_from_json top 1

    method_name_list = []
    count_correct_match_list = []
    accuracy_error_list = []
    accuracy_valid_list = []
    accuracy_full_list = []

    for method_name in stored_accuracy:
        method_name_list.append(method_name)
        correct_match = stored_accuracy[method_name]["top 1"]

        count_correct_match_list.append(correct_match)

        accuracy_error_list.append(correct_match/len(df_error))

        accuracy_valid_list.append((correct_match+len(df_valid)-len(df_error))/len(df_valid))

        accuracy_full_list.append((correct_match+len(df_valid)-len(df_error))/len(df_full))

    df_accuracy_save = pd.DataFrame(list(zip(method_name_list,count_correct_match_list,accuracy_error_list,accuracy_valid_list,accuracy_full_list)),columns=['method_name','count','error','valid','full'])
    df_accuracy_save.to_csv(os.path.join(save_output,f"data_accuracy_{lang}.csv"))    

