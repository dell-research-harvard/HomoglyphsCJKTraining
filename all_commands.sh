#!/bin/bash
'''
Run Synthetic Matching
'''
cd /mnt/data01/yxm/homoglyphic_matching/multi_lang

# Easy to Paddle
python synthetic_placename_match.py --lang ko_paddle_easy --save_output ./c_j_k_match_top_10_ko_paddle_easy --homo --lev --fuzzychinese --simstring --multilang_dir multilang_results
python synthetic_placename_match.py --lang ja_paddle_easy --save_output ./c_j_k_match_top_10_ja_paddle_easy --homo --multilang_dir multilang_results --lev --fuzzychinese --simstring
python synthetic_placename_match.py --lang zhs_easy_paddle_80000 --save_output ./c_j_k_match_top_10_zhs_easy_paddle_80000_expand_final --homo --multilang_dir multilang_results_gcv --lev --fuzzychinese --simstring
python synthetic_placename_match.py --lang zht_paddle_easy --save_output ./c_j_k_match_top_10_zht_paddle_easy --homo --multilang_dir multilang_results --lev --fuzzychinese --simstring

# Paddle to GT
python synthetic_placename_match.py --lang zht --save_output ./zht_top10 --homo --lev --simstring --fuzzychinese --multilang_dir multilang_results
python synthetic_placename_match.py --lang zhs_80000 --save_output ./zhs_80000_results --homo --lev --simstring --fuzzychinese --multilang_dir multilang_results
python synthetic_placename_match.py --lang ja --save_output ./ja_paddle_top10_result --homo --lev --simstring --fuzzychinese --multilang_dir multilang_results
python synthetic_placename_match.py --lang ko --save_output ./c_j_k_match_top_10_test --homo --lev --simstring --fuzzychinese --multilang_dir multilang_results

# Easy to GT
python synthetic_placename_match.py --lang ja_easy --save_output ./c_j_k_match_top_10_ja_easy --homo --lev --fuzzychinese --simstring --multilang_dir multilang_results
python synthetic_placename_match.py --lang ko_easy --save_output ./c_j_k_match_top_10_ko_easy --homo --lev --fuzzychinese --simstring --multilang_dir multilang_results
python synthetic_placename_match.py --lang zhs_easy_80000 --save_output ./c_j_k_match_top_10_zhs_easy_80000_expand --homo --lev --fuzzychinese --simstring --multilang_dir multilang_results
python synthetic_placename_match.py --lang zht_easy --save_output ./c_j_k_match_top_10_zht_easy --homo --lev --fuzzychinese --simstring --multilang_dir multilang_results



# Replicate Synthetic Matching accuracy, including the path to all final matched files
python synthetic_accuracy_easy_2_paddle.py
python synthetic_accuracy_easy_2_gt.py
python synthetic_accuracy_paddle_2_gt.py

'''
Run TK and PR matching example
'''
python japan_supplier_match.py --homo --simstring --fuzzychinese --lev --match_task [['eff_2_efftk','effocr_partner','effocr_tk_title_dup_68352_clean_path.json','eff']] --save_output ./tk_title_partner_paddle_easy_dict_test_set

python japan_supplier_match.py --homo --simstring --fuzzychinese --lev --match_task [['easy_2_paddlepr','easy_ocr_partner','paddleocr_pr_title.json','cjk'], ['easy_2_easypr','easy_ocr_partner','easyocr_pr_title.json','cjk'], ['paddle_2_paddlepr','paddle_ocr_partner','paddleocr_pr_title.json','cjk'],]] --save_output ./pr_title_partner_paddle_easy_dict_debugged

## Future clean up, you can remove the cjk, eff...

