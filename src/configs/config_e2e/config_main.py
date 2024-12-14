config_dict = {}

# 把下面注释里的参数都放到config_dict里


# 模型的版本，通过是否带有"api"来区别本地模型和api
config_dict["model_version"] = "qwen-7b-chat"
# 模型的路径(如果用的是modelscope则此参数不需要填，默认都存在.cache里)
config_dict["model_name_or_path"] = "/home/myjia/Medical_LLM_task/LLMs_base_model/{}".format(config_dict["model_version"])
# 模型的类型，用于区分不同类型模型的chat接口
config_dict["model_type"] = None

if "glm" in config_dict["model_version"]:
    config_dict["model_type"] = "glm"
elif "baichuan" in config_dict["model_version"]:
    config_dict["model_type"] = "baichuan"
elif "qwen" in config_dict["model_version"]:
    if "api" in config_dict["model_version"]:
        config_dict["model_type"] = "qwen_api"  
    else:
        config_dict["model_type"] = "qwen"

# 数据相关：输入数据的文件夹路径
config_dict["fin_directory"] = f'/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/processed_json_files/zh/'
# config_dict["fin_directory"] = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/multi-choice_QA/CMB/CMB-Clin/"

# NER模型，默认为RaNER模型，TODO:尝试其它ner模型
config_dict["ner_model_id"] = "/home/myjia/Medical_LLM_task/PLMs_base_model/nlp_raner_named-entity-recognition_chinese-base-cmeee"

# retriever
config_dict["retriever_type"] = "dense" # 检索器的类型：["sparse", "dense", "hybrid"]
config_dict["retriever_version"] = "dr_corom_emb" # 检索器的基模型：sr: ["bm25_merge_kg"]; dr: ["dr_corom_emb",  "dr_bge_emb"]; hr: ["hr_bm25_corom", "hr_bm25_bge"]

# neo4j，知识图谱相关参数
config_dict["uri"] = "bolt://localhost:7687" # neo4j连接ip
config_dict["username"] = "neo4j" # neo4j连接用户名
config_dict["password"] = "123456" # neo4j连接密码
config_dict["kg_database_name"] = "mergekg" # 图数据库的名字
config_dict["kg_type"] = "merge" # 类型，这个参数主要是为了加载下面几个文件
config_dict["kg_entity_path"] = '/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_{}.txt'.format(config_dict["kg_type"]) # 知识图谱的全部实体
config_dict["entity_type_map_path"] = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/entity_type_maps/entity_type_map_{}.json".format(config_dict["kg_type"]) # 知识图谱全部的实体->实体类型的映射关系
config_dict["subgraph_name"] = "subgraph" # gds子图的名字

# neo4j， 实体权重相关参数
config_dict["entity_weight_map_file"] = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/entity_weight_map/entity_weight_emr.json"

# 候选疾病相关参数
config_dict["direct_topn"] = 2 # LLM直接进行预测的疾病个数
config_dict["dis_topn"] = 2 * config_dict["direct_topn"] # rerank之前保留的候选疾病个数
config_dict["rerank_topn"] = config_dict["direct_topn"] # rerank之后保留的候选疾病的个数
config_dict["final_topn"] = config_dict["direct_topn"] # 最终保留的候选疾病的个数
config_dict["path_topn"] = 3 # 路径长度阈值，超过此阈值则视为弱相关

# 总的科室列表：["儿科", "耳鼻咽喉科", "妇产科", "护理科", "急诊科", "精神科", "康复科", "口腔科", "麻醉疼痛科", "内科", "皮肤性病科", "外科", "眼科", "肿瘤科"]

# 已经完成的数据集/科室列表
config_dict["finished_list"] = []

# 本次运行需要完成的任务列表
config_dict["task_list"] = ["CMB-Clin"]
# 上次运行终止的任务/科室/数据集
config_dict["cur_dep"] = "None"
# 上次运行终止的断点位置索引
config_dict["cur_idx"] = -1
# 本次运行终止的断点位置索引
config_dict["stop_idx"] = 3200

# 输出相关：通常来说，输出需要记录-1.用了哪个模型；2.检索器；3.候选疾病的数量；4.(TODO)ner模型的版本
config_dict["result_log_pred_dir"] = f"/home/myjia/Medical_LLM_task/EMR_diagnos/data/multi-choice_QA/CMB/CMB-Clin/{config_dict['model_version']}_{config_dict['retriever_version']}_{config_dict['direct_topn']}/"
