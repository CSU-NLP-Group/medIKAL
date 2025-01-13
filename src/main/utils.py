from tqdm import tqdm
import re
import json
# from multiprocessing import Pool
import time
# from retriever import *
# from ner import *
# from config import *

class KGTools:
    """ 一些工具函数 """
    def __init__(self, ner_model, retriever, kg, rerank_topn, dis_topn, path_topn, entity_weight_map_file):
        self.ner_model = ner_model
        self.retriever = retriever
        self.kg = kg

        self.dis_topn = dis_topn
        self.rerank_topn = rerank_topn
        self.path_topn = path_topn

        with open(entity_weight_map_file, 'r', encoding="utf-8") as f:
            self.entity_weight_map = json.load(f)

    
    def process_output(self, text):
        pred_response = [t.split("：")[1].split("\n")[0] for t in text.replace(":", "：").split("预测疾病")[1:] if "：" in t]
        pred_response = [t for t in pred_response if t != ""]
        if pred_response == []:
            extract_dis = self.ner_model.ner(text=text)
            for extract_res in extract_dis:
                for out in extract_res['output']:
                    if out['type'] == 'dis':
                        pred_response.append(out['span'])
            if pred_response == []:
                return []
            
        res = [self.check_match(dis) for dis in pred_response]

        return res
    
    def check_match(self, entity):
        assert entity is not None
        if entity in self.kg.kg_entities:
            return entity
        else:
            retrieve_res = self.retriever.retrieve(query=entity, top_k=5)
            if len(retrieve_res) != 0:
                return retrieve_res[0]['text']
            else:
                return ""
        
    def get_past_dis(self, ner_result):
        
        past_dis = []
        exam_dis = [] # 辅助检查中出现的疾病

        past_dict = ner_result[1]
        exam_dict = ner_result[2]

        for dis in past_dict['dis']:
            past_dis.append(dis['kg_entity'])
        
        for dis in exam_dict['dis']:
            exam_dis.append(dis['kg_entity'])
        
        return past_dis, exam_dis
    
    def get_ner_result(self, chief_complaint, fst_rd_summary, scd_rd_summary):
        """ 获取实体识别结果，并保存在字典中 """
        total_ner_dict = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict1 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict2 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict3 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}

        EMR2kg_entity_map = {}

        extract_result1 = self.ner_model.ner(text=chief_complaint)
        extract_result2 = self.ner_model.ner(text=fst_rd_summary)
        extract_result3 = self.ner_model.ner(text=scd_rd_summary)

        for extract_res in extract_result1:
            for out in extract_res["output"]:
                cur_entity_dict = {}
                mapped_entity = self.check_match(out['span'])
                if mapped_entity == "":
                    continue
                EMR2kg_entity_map[out['span']] = mapped_entity
                cur_entity_dict["kg_entity"] = mapped_entity
                cur_entity_dict["kg_entity_type"] = self.kg.entity_type_map[mapped_entity]
                cur_entity_dict["EMR_entity"] = out['span']

                ner_dict1[out['type']].append(cur_entity_dict)
                total_ner_dict[out['type']].append(cur_entity_dict)
        
        for extract_res in extract_result2:
            for out in extract_res["output"]:
                cur_entity_dict = {}
                mapped_entity = self.check_match(out['span'])
                if mapped_entity == "":
                    continue
                EMR2kg_entity_map[out['span']] = mapped_entity
                cur_entity_dict["kg_entity"] = mapped_entity
                cur_entity_dict["kg_entity_type"] = self.kg.entity_type_map[mapped_entity]
                cur_entity_dict["EMR_entity"] = out['span']

                ner_dict2[out['type']].append(cur_entity_dict)
                total_ner_dict[out['type']].append(cur_entity_dict)
        
        for extract_res in extract_result3:
            for out in extract_res["output"]:
                cur_entity_dict = {}
                mapped_entity = self.check_match(out['span'])
                if mapped_entity == "":
                    continue
                EMR2kg_entity_map[out['span']] = mapped_entity
                cur_entity_dict["kg_entity"] = mapped_entity
                cur_entity_dict["kg_entity_type"] = self.kg.entity_type_map[mapped_entity]
                cur_entity_dict["EMR_entity"] = out['span']

                ner_dict3[out['type']].append(cur_entity_dict)
                total_ner_dict[out['type']].append(cur_entity_dict)

        # 对每个字典的键对应的value列表进行去重
        for key in total_ner_dict.keys():
            total_ner_dict[key] = [dict(t) for t in {tuple(d.items()) for d in total_ner_dict[key]}]
        
        for key in ner_dict1.keys():
            ner_dict1[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict1[key]}]
        
        for key in ner_dict2.keys():
            ner_dict2[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict2[key]}]

        for key in ner_dict3.keys():
            ner_dict3[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict3[key]}]
        
        self.EMR2kg_entity_map = EMR2kg_entity_map

        ner_result = [ner_dict1, ner_dict2, ner_dict3]

        return ner_result, total_ner_dict, EMR2kg_entity_map
    
    # def rerank_by_sym(self, candidate_disease, EMR_sym_list):
    #     """ 通过已有的症状列表，与候选疾病的症状做交集和并集，最终的分数为交集的个数/并集的个数 """
    #     rerank_dict = {}
    #     for disease in candidate_disease:
    #         disease_sym_list = self.kg.get_disease_sym(disease)
    #         intersection = list(set(EMR_sym_list) & set(disease_sym_list))
    #         # union = list(set(EMR_sym_list) | set(disease_sym_list))
    #         if len(EMR_sym_list) == 0:
    #             rerank_dict[disease] = 1
    #         else:
    #             rerank_dict[disease] = len(intersection) / len(EMR_sym_list)
    #         # 如果交集为空，则直接从候选疾病中删除

    #     result = rerank_dict.copy()
    #     for disease in rerank_dict.keys():
    #         if rerank_dict[disease] == 0:
    #             result.pop(disease)

    #     result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    #     return list(result.keys())[:self.rerank_topn]
    
    def rerank_by_path(self, candidate_disease, EMR_entity_dict, neighbor_dis_entity_map):
        """ 通过到目标实体的最短路径长度总和来对候选疾病进行排序"""

        length = len(candidate_disease)

        # 先统计所有实体的数量，后续计算路径得分时用来归一化
        entity_list = []
        for key, value in EMR_entity_dict.items():
            if value == [] or (key != "sym" and key != "dis" and key != "dru"):continue
            for entity in value:
                if entity["kg_entity_type"] not in self.kg.sub_graph_entity_list:
                    continue
                try: 
                    id_ = self.kg.gds.find_node_id([entity["kg_entity_type"]], {"name":entity["kg_entity"]})
                except:
                    try:
                        id_ = self.kg.gds.find_node_id([], {"name":entity["kg_entity"]})
                    except:
                        continue

                entity_list.append((entity["kg_entity"], id_))
        
        entity_list = list(set(entity_list))
        entity_cnt = len(entity_list)

        dis_rerank_score = {}
        dis_path_dict = {}

        time3 = time.time()

        for _, disease in tqdm(enumerate(candidate_disease), desc=f"Rerank by path, total disease: {length}", total=length):
            dis_rerank_score[disease] = 0
            dis_path_dict[disease] = {}
            try:
                source_id = self.kg.gds.find_node_id(["疾病"], {"name": disease})
            except:
                try:
                    source_id = self.kg.gds.find_node_id([], {"name": disease})
                except:
                    continue

            for entity in entity_list:
                dis_path_dict[disease][entity[0]] = []
                if entity[0] in neighbor_dis_entity_map[disease]:
                    dis_rerank_score[disease] += 1
                    continue
                result = self.kg.gds.shortestPath.dijkstra.stream(
                        self.kg.sub_graph,
                        sourceNode=source_id,
                        targetNode=entity[1],
                )
                try:
                    total_cost = result["totalCost"][0]
                except:
                    total_cost = 0
                if total_cost == 0:
                    dis_rerank_score[disease] += 0
                else:
                    dis_rerank_score[disease] += 1 / result["totalCost"][0]
                    dis_path_dict[disease][entity[0]] = self.get_path_str(result["nodeIds"][0])
        
        time4 = time.time()
        print(f"非并行计算时间：{time4 - time3}")

        # 对dis_rerank_score的每一项得分使用entity_cnt进行归一化
        if entity_cnt != 0:
            for key in dis_rerank_score.keys():
                dis_rerank_score[key] /= entity_cnt

        # 我们希望候选疾病与所有实体的距离之和越短越好
        dis_rerank_score = dict(sorted(dis_rerank_score.items(), key=lambda x: x[1], reverse=True)) 
        return list(dis_rerank_score.keys())[:self.rerank_topn], dis_path_dict
    
    def get_candidate_disease_by_KG(self, total_ner_dict, fst_rd_can_dis):
        """ 从知识图谱中获取候选疾病 """

        dis_score_dict = {}
        dis_entity_map = {}
        ner_list = []
        
        for key,value in total_ner_dict.items():
            if value == []:
                continue
            for entity in value:
                if self.entity_weight_map[key] == 0:
                    continue
                neighbor_list = list(set(self.kg.get_neighbor_disease(entity['kg_entity'])))
                if entity["kg_entity_type"] == "社会学": # 这个是知识图谱的标注错误，很多疾病的类型被标成了"社会学"
                    neighbor_list.append(entity['kg_entity'])
                ner_list.append(entity)
                for cur_dis in neighbor_list:
                    if cur_dis not in dis_score_dict:
                        dis_score_dict[cur_dis] = self.entity_weight_map[key]
                        dis_entity_map[cur_dis] = [entity['kg_entity']]
                    else:
                        dis_score_dict[cur_dis] += self.entity_weight_map[key]
                        dis_entity_map[cur_dis].append(entity['kg_entity'])

        # 这一步的目的是筛选初步选定的候选疾病，如果某个候选疾病没有出现在dis_entity_map,那说明它和病历中的任何一个描述都没有关系，直接删除
        fst_rd_can_dis_kg = [self.check_match(dis) for dis in fst_rd_can_dis]

        tmp_idx = 0
        while tmp_idx < len(fst_rd_can_dis_kg):
            if fst_rd_can_dis_kg[tmp_idx] not in dis_entity_map:
                fst_rd_can_dis_kg.remove(fst_rd_can_dis_kg[tmp_idx])
            else:
                tmp_idx += 1

        dis_score_dict = dict(sorted(dis_score_dict.items(), key=lambda x:x[1], reverse=True))

        cnt = self.dis_topn - len(fst_rd_can_dis_kg)
        if cnt < 0: cnt = 0
        # 考虑到采用实体得分的方式，有可能在计算某些疾病的分数时错误的把重要的疾病排除掉了，因此这里是直接把fst_rd_can_dis的疾病合并到neighbor_candidate_disease中
        neighbor_candidate_disease = fst_rd_can_dis_kg
        for dis in dis_score_dict.keys():
            if cnt == 0:
                break
            if dis not in neighbor_candidate_disease:
                neighbor_candidate_disease.append(dis)
                cnt -= 1

        neighbor_dis_entity_map = {k:dis_entity_map[k] for k in neighbor_candidate_disease}

        reranked_candidate_disease, reranked_dis_path_dict = self.rerank_by_path(neighbor_candidate_disease, total_ner_dict, neighbor_dis_entity_map)
        final_dis_entity_map = {k:neighbor_dis_entity_map[k] for k in reranked_candidate_disease}

        return reranked_candidate_disease, final_dis_entity_map, reranked_dis_path_dict
        
    # def check_link(self, entity, target_list):
    #     source_id = self.kg.gds.find_node_id([], {"name":entity})
    #     target_id_list = [self.kg.gds.find_node_id([ti["kg_entity_type"]], {"name":ti["kg_entity"]}) for ti in target_list]
    #     res = []
    #     for i, target_id in enumerate(target_id_list):
    #         query = f"""
    #         MATCH (n1)-[]-(n2) WHERE ID(n1) = {source_id} AND ID(n2) = {target_id} RETURN COUNT(*) > 0 AS isConnected
    #         """
    #         result = self.kg.session.run(query = query)
    #         for record in result:
    #             if record["isConnected"]: res.append(target_list[i]["kg_entity"])
    #     return res
    
    def check_chief(self, ner_dict_chief, dis_entity_list):
        chief_sym = []
        for _ in ner_dict_chief['sym']:
            chief_sym.append(_['EMR_entity'])
        for _ in ner_dict_chief['dis']:
            chief_sym.append(_['EMR_entity'])
        # 如果chief_sym和dis_entity_list有交集，那么返回它们的交集，同时返回评分(计算方式：交集的个数/chief_sym总的实体个数，计算时要防止分母为0，如果分母为0，那么直接返回0)
        if len(list(set(chief_sym) & set(dis_entity_list))) > 0:
            return "是", list(set(chief_sym) & set(dis_entity_list)), len(list(set(chief_sym) & set(dis_entity_list))) / len(chief_sym) if len(chief_sym) != 0 else 0
        else:
            return "否", [], 0
    
    def check_drug(self, disease, drug_entity, reranked_dis_path_dict):
        # 这里写的有问题，在写入最后的result时，应该用EMR中的实体，而不是用KG中的实体
        # drug_entity = []
        # for _ in ner_dict_drug['dru']:
        #     drug_entity.append(_['kg_entity'])
        result = ""
        if len(drug_entity) == 0:
            return "无药物使用史\n", []
        for drug in drug_entity:
            if drug not in reranked_dis_path_dict[disease]:
                continue
            if len(reranked_dis_path_dict[disease][drug]) != 0 and len(reranked_dis_path_dict[disease][drug]) < self.path_topn:
                result += f"与既往使用药物“{drug}”强相关。关联路径为：\n"
                for path in reranked_dis_path_dict[disease][drug]:
                    result += path + "\n"
            elif len(reranked_dis_path_dict[disease][drug]) >= self.path_topn:
                result += f"与既往使用药物“{drug}”弱相关\n"
            else:
                continue
        
        if result == "":
            return "与患者既往使用药物无关联\n", drug_entity

        return result, drug_entity
    
    def check_history(self, disease, ner_dict_dis, past_dis, reranked_dis_path_dict):
        
        past_dis_history = []
        for dis in ner_dict_dis['dis']:
            past_dis_history.append(dis['kg_entity'])

        if disease in past_dis_history:
            return "属于患者既往病史"
        else:
            result = ""
            cnt = 1
            for past_dis_kg in past_dis_history:
                if past_dis_kg not in reranked_dis_path_dict[disease]:
                    continue
                # past_dis_kg = self.EMR2kg_entity_map[past_dis]
                # if past_dis_kg not in reranked_dis_path_dict[disease]:
                if len(reranked_dis_path_dict[disease][past_dis_kg]) != 0 and len(reranked_dis_path_dict[disease][past_dis_kg]) < self.path_topn:
                    result += f"{cnt}. "
                    result += "与既往疾病“" + past_dis_kg + "”强相关。关联路径为：\n"
                    for path in reranked_dis_path_dict[disease][past_dis_kg]:
                        result += path + "\n"
                    cnt += 1
                elif len(reranked_dis_path_dict[disease][past_dis_kg]) >= self.path_topn:
                    result += f"{cnt}. "
                    result += "与既往疾病“" + past_dis_kg + "”弱相关\n"
                    cnt += 1

            if result == "":
                return "与患者既往病史无关联"    
            
            return result
        
    def check_exam(self, ner_dict_exam, dis_entity_list):
        exam_sym = []
        for _ in ner_dict_exam['sym']:
            exam_sym.append(_['kg_entity'])
        for _ in ner_dict_exam['dis']:
            exam_sym.append(_['kg_entity'])
        # 如果完全一致
        if len(list(set(exam_sym) & set(dis_entity_list))) == len(dis_entity_list):
            return "完全吻合", list(set(exam_sym) & set(dis_entity_list))
        elif len(list(set(exam_sym) & set(dis_entity_list))) > 0:
            return "部分吻合", list(set(exam_sym) & set(dis_entity_list))
        else:
            return "不吻合", []
        

    def get_path_str(self, node_id_list):
        path_str = ""
        node_list = [self.kg.gds.util.asNode(node_id)["name"] for node_id in node_id_list]
        # 现在已知node_list里的节点(已经转换为名称了)是依次连接成一条路径的，现在需要查出它们之间依次连接时的关系
        for i in range(len(node_list) - 1):
            path_str += node_list[i] + "->"
        path_str += node_list[-1]

        return path_str

def ae_preprocess(auxiliary_examination):
    """ 对辅助检查进行预处理 """
    auxiliary_examination = auxiliary_examination.replace("  - ", " ")
    auxiliary_examination = auxiliary_examination.replace("- ", " ")
    return auxiliary_examination


# def vote_module(direct_diagnos_result, diagnos_via_KG_result, history_disease, reranked_candidate_disease):
#     direct_dis_list = process_output(direct_diagnos_result)
#     diagnos_via_KG_list = process_output(diagnos_via_KG_result)

#     disease_dict = {}
#     # 投票过程
#     for dis in direct_dis_list:
#         if dis not in disease_dict:
#             disease_dict[dis] = 1
#         else:
#             disease_dict[dis] += 1
    
#     for dis in diagnos_via_KG_list:
#         if dis not in disease_dict:
#             disease_dict[dis] = 1
#         else:
#             disease_dict[dis] += 1
    
#     for dis in history_disease:
#         if dis not in disease_dict:
#             disease_dict[dis] = 1
#         else:
#             disease_dict[dis] += 1
    
#     for dis in reranked_candidate_disease:
#         if dis not in disease_dict:
#             disease_dict[dis] = 1
#         else:
#             disease_dict[dis] += 1
    
#     # 对结果进行排序，并返回top-5的列表
#     sorted_disease_dict = dict(sorted(disease_dict.items(), key=lambda x:x[1], reverse=True))
#     final_result = list(sorted_disease_dict.keys())[:args.final_topn]

#     return final_result