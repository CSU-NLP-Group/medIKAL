from neo4j import GraphDatabase, basic_auth
from graphdatascience import GraphDataScience
import json

entity_list = [
    "None",
    "其他",
    "其他治疗",
    "手术治疗",
    "检查",
    "流行病学",
    "疾病",
    "症状",
    "社会学",
    "药物",
    "部位",
    "预后",
]


class MyKnowledgeGraph:
    def __init__(self, uri, username, password, kg_database, subgraph_name, kg_entity_path, entity_type_map_path):
        """ neo4j driver和session """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self.session = self.driver.session()

        """ gds以及子图加载 """
        self.gds = GraphDataScience.from_neo4j_driver(uri, auth=(username, password), database=kg_database)
        # sub_graph就是原来的G
        self.sub_graph = self.gds.graph.get(subgraph_name)
        self.sub_graph_entity_list = entity_list

        """ 知识图谱实体集合加载 """
        kg_entities = []
        kg_entity_filepath = kg_entity_path
        with open(kg_entity_filepath, 'r', encoding='utf-8') as kg_entity_file:
            kg_entity_lines = kg_entity_file.readlines()
            for kg_entity_line in kg_entity_lines:
                kg_entities.append(kg_entity_line.strip().split('\t')[0])
        kg_entity_file.close()
        self.kg_entities = kg_entities

        """ 知识图谱实体类型映射加载 """
        fin_entity_type_map = open(entity_type_map_path, "r", encoding="utf-8")
        entity_type_map = json.load(fin_entity_type_map)
        self.entity_type_map = entity_type_map

    def close(self):
        self.session.close()
        self.driver.close()

    def get_neighbor_disease(self, entity_name):
        # 按照关系类型查询实体的邻居实体
        query = """
        MATCH (e)-[r]-(n:`疾病`)
        WHERE e.name = $entity_name
        RETURN collect(n.name) AS neighbor_entities
        """
        result = self.session.run(query, entity_name=entity_name)

        neighbor_list = []
        try:
            for record in result:
                neighbors = record["neighbor_entities"]
                neighbor_list.extend(neighbors)
        except:
            neighbor_list = []

        return neighbor_list   

    def get_disease_sym(self, disease_name):
        # 按照关系类型查询实体的邻居实体
        query = """
        MATCH (e)-[r:`临床表现`]-(n)
        WHERE e.name = $disease_name
        RETURN collect(n.name) AS neighbor_entities
        """
        result = self.session.run(query, disease_name=disease_name)

        neighbor_list = []
        for record in result:
            neighbors = record["neighbor_entities"]
            neighbor_list.extend(neighbors)

        return neighbor_list 

    def get_disease_info(self, disease_name):
        # 按照关系类型查询实体的邻居实体
        gender_list = []
        pop_list = []
        age_list = []
        bod_list = []

        query1 = """
        MATCH (e)-[r:`发病性别倾向`]-(n)
        WHERE e.name = $disease_name
        RETURN collect(n.name) AS neighbor_entities
        """
        result1 = self.session.run(query1, disease_name=disease_name)
        for record in result1:
            neighbors1 = record["neighbor_entities"]
            gender_list.extend(neighbors1)

        # query2 = """
        # MATCH (e)-[r:`多发群体`]-(n)
        # WHERE e.name = $disease_name
        # RETURN collect(n.name) AS neighbor_entities
        # """
        # result2 = session.run(query2, disease_name=disease_name)
        # for record in result2:
        #     neighbors2 = record["neighbor_entities"]
        #     pop_list.extend(neighbors2)

        query3 = """
        MATCH (e)-[r:`多发群体`]-(n)
        WHERE e.name = $disease_name
        RETURN collect(n.name) AS neighbor_entities
        """
        result3 = self.session.run(query3, disease_name=disease_name)
        for record in result3:
            neighbors3 = record["neighbor_entities"]
            age_list.extend(neighbor