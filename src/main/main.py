import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import torch
# import openpyxl
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# from transformers.generation.utils import GenerationConfig
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# import modelscope
# from neo4j import GraphDatabase, basic_auth
# from graphdatascience import GraphDataScience
# from retriv import SparseRetriever

# import zhipuai
# from langchain.llms import ChatGLM
from tqdm import tqdm
import numpy as np
import pandas as pd

import time
import sys
import re
import math
# sys.path.append('/home/myjia/Medical_LLM_task/MindMap/TextMatch')

"""导入自建模块"""
from utils import *
from ner import *
from kg_func import *
from models import *
from retriever import *
from doctor import *
from config import *

# ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->        

# ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-> 

def main(doctor, KG_Tools, chief_complaint, current_medical_history, past_disease_history, body_check, auxiliary_exam):

    # 对电子病历的基本信息部分进行总结
    fst_rd_summary, fst_rd_history = doctor.general_info_summary(chief_complaint, current_medical_history, past_disease_history)
    # 对输出进行处理
    # fst_rd_summary = "\n".join(fst_rd_summary.split("\n")[:5])

    scd_rd_summary, scd_rd_history = doctor.examination_summary(body_check, auxiliary_exam)
    # scd_rd_summary = "查体：\n" + body_check + "\n" + "辅助检查：\n" + auxiliary_exam
    
    ner_result, total_ner_dict, EMR2kg_entity_map = KG_Tools.get_ner_result(chief_complaint, fst_rd_summary, scd_rd_summary)

    # 把既往药物使用情况进行记录，后续需要通过判断drug_entity的长度来确定患者是否有药物使用史，进而确定这一组得分是否需要计算到总分中
    ner_dict_drug = ner_result[1]
    drug_entity = []
    for _ in ner_dict_drug['dru']:
        drug_entity.append(_['EMR_entity'])

    # 基于上述总结进行直接诊断
    direct_diagnos_result, direct_diagnos_history = doctor.direct_diagnos(fst_rd_summary, scd_rd_summary, args.direct_topn)
    # 将输出的疾病预测结果转换为列表
    fst_rd_can_dis = KG_Tools.process_output(direct_diagnos_result)

    # 获取患者病史信息和检查结果中的疾病信息
    past_dis, exam_dis= KG_Tools.get_past_dis(ner_result)

    direct_diagnos_dis = fst_rd_can_dis

    fst_rd_can_dis = list(set(fst_rd_can_dis + past_dis + exam_dis))

    # 获取知识图谱搜索结果
    reranked_candidate_disease, final_dis_entity_map, reranked_dis_path_dict = KG_Tools.get_candidate_disease_by_KG(total_ner_dict, fst_rd_can_dis)
       
    final_result = []
    # total_formatted_knowledge = ""
    # dis_cnt = 1
    prompt_list = []
    for i, disease in tqdm(enumerate(reranked_candidate_disease), desc="分析候选疾病...", total=len(reranked_candidate_disease)):
        cur_formatted_knowledge = ""
        cur_formatted_knowledge += f"疾病：{disease}\n相关信息：\n"
        # 主诉相关的信息
        chief_result = KG_Tools.check_chief(ner_result[0], final_dis_entity_map[disease])
        if chief_result[0] == "是":
            cur_formatted_knowledge += f"(1).是否符合患者主诉：\n[{chief_result[0]}, {chief_result[1]}]\n" # '是', ['左侧肢体无力']
        else:
            cur_formatted_knowledge += f"(1).是否符合患者主诉：\n[{chief_result[0]}]\n" # '是', ['左侧肢体无力']
        # cur_formatted_knowledge += f"主诉吻合度评估：[?]\n\n"

        # 与患者既往病史的关联程度
        history_result = KG_Tools.check_history(disease, ner_result[1], past_dis, reranked_dis_path_dict)
        cur_formatted_knowledge += f"(2).与患者既往病史的关联：\n{history_result}"
        # cur_formatted_knowledge += "既往病史关联程度评估：[?]\n\n"
        
        # 与患者既往药物使用情况相关的信息
        drug_result, drug_list = KG_Tools.check_drug(disease, drug_entity, reranked_dis_path_dict)
        cur_formatted_knowledge += f"(3).与患者既往药物使用情况关联：\n{drug_result}"
        # cur_formatted_knowledge += "药物使用情况评估：[?]\n\n"

        # 与患者检查指标关联程度
        exam_result = KG_Tools.check_exam(ner_result[2], final_dis_entity_map[disease])
        cur_formatted_knowledge += f"(4).与患者检查项目关联：\n[{exam_result[0]}, 吻合项目：{exam_result[1]}]\n"
        # cur_formatted_knowledge += "检查项目吻合程度评估：[?]\n\n"

        exam_result = auxiliary_exam if auxiliary_exam != "无" else body_check
        dis_history = past_dis if past_dis != [] else "既往无病史"
        drug_history = drug_list if drug_list != [] else "既往无用药史"
        dis_analysis, _ = doctor.analysis(disease, cur_formatted_knowledge, chief_complaint, dis_history, drug_history, exam_result)
        
        final_result.append((disease, dis_analysis))
        # total_formatted_knowledge += cur_formatted_knowledge
        # dis_cnt += 1
    
    # final_answer = doctor.analysis(total_formatted_knowledge, fst_rd_summary, scd_rd_summary)
    
    final_result.append([direct_diagnos_dis, past_dis, exam_dis, drug_entity])

    return final_result


if __name__ == "__main__":

    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    """初始化各个模块"""   

    # 知识图谱初始化 uri, username, password, kg_database, subgraph_name, kg_entity_path, entity_type_map_path
    kg_system = MyKnowledgeGraph(uri=args.uri, username=args.username, password=args.password, kg_database=args.kg_database_name, subgraph_name=args.subgraph_name, kg_entity_path=args.kg_entity_path, entity_type_map_path=args.entity_type_map_path)

    # NER模型初始化
    ner_model = NER_Model(args.ner_model_id, device='gpu')

    # 检索器初始化
    retriever = Retriever(args.retriever_type, args.retriever_version) 

    # LLM初始化
    chat_model = ChatModel(args.model_type, args.model_name_or_path, args.model_version)

    # prompt初始化
    doctor = Doctor(chat_model)

    # 其它工具模块初始化
    KG_Tools = KGTools(ner_model=ner_model, retriever=retriever, kg=kg_system, rerank_topn=args.rerank_topn, dis_topn=args.dis_topn, path_topn=args.path_topn, entity_weight_map_file = args.entity_weight_map_file)

    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # 用一个总的json文件记录各个数据集的输出结果
    if not os.path.exists(args.result_log_pred_dir):
        os.makedirs(args.result_log_pred_dir)
    total_record_file = open(args.result_log_pred_dir + "total_pred_record.json", "a", encoding="utf-8")

    # finished_list = ["肿瘤科", "口腔科"]

    # 对指定目录下的每个json文件进行读取
    for file_name in os.listdir(args.fin_directory):
        if file_name.endswith('.json'):
            # check：如果当前文件属于finished_list，说明之前已经运行完了；如果当前文件不属于task_list，说明不在任务列表里，不需要运行
            if file_name.split(".")[0] in args.finished_list or file_name.split(".")[0] not in args.task_list:continue

            # 用一个json文件记录当前数据集的输出结果
            record_file = open(args.result_log_pred_dir + file_name[:-5] + "_pred_record.json", "a", encoding="utf-8")

            with open(args.fin_directory + file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for i, EMR_dict in tqdm(enumerate(data), total = len(data), desc = f'开始运行{file_name.split(".")[0]}诊断程序'):

                    # check: cur_dep和cur_idx分别表示程序上一次运行停在了哪个科室的具体哪个病历上
                    if file_name.split(".")[0] == args.cur_dep and i < args.cur_idx: continue
                    if i == args.stop_idx: break

                    # 用于记录输出信息
                    pred_dict = {}

                    # 输入病历的各项内容
                    personal_info = EMR_dict["基本信息"] if "基本信息" in EMR_dict else "无"
                    chief_complaint = EMR_dict["主诉"] if "主诉" in EMR_dict else "无"
                    current_medical_history = EMR_dict["现病史"] if "现病史" in EMR_dict else "无"
                    past_disease_history = EMR_dict["既往史"] if "既往史" in EMR_dict else "无"
                    body_check = EMR_dict["查体"] if "查体" in EMR_dict else "无"
                    auxiliary_exam = EMR_dict["辅助检查"] if "辅助检查" in EMR_dict else "无"
                    label = EMR_dict["label"]

                    # 对病历内容进行预处理
                    if auxiliary_exam != "无":
                        auxiliary_exam = ae_preprocess(auxiliary_exam)

                    # 主函数
                    final_response = main(doctor, KG_Tools, chief_complaint, current_medical_history, past_disease_history, body_check, auxiliary_exam)

                    # 记录输出结果
                    pred_dict["index"] = i + 1
                    pred_dict["pred_response"] = final_response

                    # 将结果写入当前文件
                    record_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
                    record_file.flush()

                    # 将结果按行写入总文件
                    total_record_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
                    total_record_file.flush()

            record_file.close()
        # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
        
    # Calculate result
    print("diagnosis program over!!!")
    total_record_file.close()
    
    # 关闭运行的neo4j session
    kg_system.close()

    

            
               