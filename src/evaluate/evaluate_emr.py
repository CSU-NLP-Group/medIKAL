# coding=utf-8
# 计算f1值等指标
# 参考文献: AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis

import xlrd
from fuzzywuzzy import process
import json
from tqdm import tqdm
import os

class HyperParams():
    def __init__(self):
        self.model_version = "qwen-7b-chat"
        self.input_ref_dir = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/processed_json_files/ref/"
        # self.input_pred_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/{self.model_version}/"
        self.input_pred_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/qwen-7b-chat/result/"
        self.ICD_database = '/home/myjia/Medical_LLM_task/EMR_diagnos/data/ICD/国际疾病分类ICD-10北京临床版v601.xls'
        self.top_n = 10
        self.threshold = 50
        # self.output_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/"
        self.output_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/qwen-7b-chat/"

args = HyperParams()

database = args.ICD_database

xls = xlrd.open_workbook(database)
sheet = xls.sheet_by_index(0)
disease_ids = sheet.col_values(colx = 0, start_rowx = 1)
disease_names = sheet.col_values(colx = 1, start_rowx = 1)
ICD_disease = {}
for disease_id, disease_name in zip(disease_ids, disease_names): 
    ICD_disease[disease_name] = disease_id


fout = open(args.output_dir + args.model_version + '_result.json', 'w', encoding="utf-8")

def set_match(pred, refs, matched):
    pred_set = [p[0] for p in pred]
    return_idx = None
    for idx, ref in enumerate(refs):
        ref_set = [r[0] for r in ref]
        for p in pred_set:
            for r in ref_set:
                if p == r and matched[idx] == 0:
                    return_idx = idx
    return return_idx

global_true_positive = 0
global_false_positive = 0
global_false_negitative = 0


for file_name in os.listdir(args.input_pred_dir):
    if file_name.endswith('.json'):

        fin_pred = open(args.input_pred_dir + file_name, 'r', encoding="utf-8")
        fin_ref = open(args.input_ref_dir + file_name.split('_')[0] + '_ref.json', 'r', encoding="utf-8")

        input_data_pred = []
        for line in fin_pred:
            input_data_pred.append(json.loads(line))
        
        input_data_ref = []
        for line in fin_ref:
            input_data_ref.append(json.loads(line))

        true_positive = 0.00001 # smooth
        false_positive = 0
        false_negitative = 0

        # output_record = []
        
        for idx in tqdm(range(len(input_data_ref)), desc=file_name.split("_res.json")[0] + "评估程序运行中"):
            line_ref = input_data_ref[idx]
            line_pred = input_data_pred[idx]

            assert line_ref["index"] == line_pred["index"]

            # references = line_ref['ref_response']
            predictions = [p for p in line_pred['pred_labels'] if p != '']

            if len(predictions) == 0:
                predictions = ['*']

            # index = line_ref['index']
            # ref_match = [process.extract(r, ICD_disease.keys(), limit=args.top_n) for r in references] # 模糊查询过程，得到的是一个列表，列表中的每个元素是一个元组，元组中的第一个元素是匹配到的icd列表里的疾病，第二个元素是匹配的分数，比如[('高血压', 100), ('1型糖尿病性高血压', 90), ('1型糖尿病性肥胖症性高血压', 90), ('2型糖尿病性高血压', 90), ('2型糖尿病性肥胖症性高血压', 90), ('糖尿病性高血压', 90), ('糖尿病性肥胖症性高血压', 90), ('糖耐量受损伴肥胖型高血压', 90), ('糖耐量受损伴高血压', 90), ('高血压所致精神障碍', 90)]
            # ref_match = [[(r[0], ICD_disease[r[0]], r[1]) for r in rr] for rr in ref_match]
            ref_match = line_ref['ref_match']

            pred_match = [process.extract(r, ICD_disease.keys(), limit=args.top_n) for r in predictions]
            pred_match = [[(r[0], ICD_disease[r[0]], r[1]) for r in rr] for rr in pred_match]

            refs = [[n for n in m if n[2] >= args.threshold] for m in ref_match]
            preds = [[n for n in m if n[2] >= args.threshold] for m in pred_match]
            set_matched = [0] * len(refs)
            for pred in preds:
                set_match_idx = set_match(pred, refs, set_matched)
                if set_match_idx is None: false_positive += 1 # do not match
                elif set_matched[set_match_idx] == 1: false_positive += 1 # can not match more than one times
                else: set_matched[set_match_idx] = 1 # first match

            true_positive += sum(set_matched)
            false_negitative += (len(refs) - sum(set_matched))

            # output_record.append({'index':index, 'ref_match':ref_match, 'ref_response':references})

        set_recall = true_positive / (true_positive + false_negitative)
        set_precision = true_positive / (true_positive + false_positive)
        set_f1 = set_precision * set_recall * 2 / (set_recall + set_precision)

        global_true_positive += true_positive
        global_false_positive += false_positive
        global_false_negitative += false_negitative
        
        fout.write(json.dumps({"dep": file_name.split("_pred_record.json")[0], "set_recall": set_recall, "set_precision": set_precision, "set_f1": set_f1}, ensure_ascii=False) + "\n")
        fout.flush()

        fin_pred.close()
        fin_ref.close()
    
global_set_recall = global_true_positive / (global_true_positive + global_false_negitative)
global_set_precision = global_true_positive / (global_true_positive + global_false_positive)
global_set_f1 = global_set_precision * global_set_recall * 2 / (global_set_recall + global_set_precision)

fout.write(json.dumps({"dep": "total_", "set_recall": global_set_recall, "set_precision": global_set_precision, "set_f1": global_set_f1}, ensure_ascii=False) + "\n")
fout.close()

    