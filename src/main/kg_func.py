from neo4j import GraphDatabase, basic_auth
from graphdatascience import GraphDataScience
import json

entity_list = [
    "None",
    "其他",
    "其他治疗",
    "手术治疗",
    # "检查",
    "流行病学",
    "疾病",
    "症状",
    "社会学",
    "药物",
    "部位",
    # "预后",
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

    # def get_disease_sym(self, disease_name):
    #     # 按照关系类型查询实体的邻居实体
    #     query = """
    #     MATCH (e)-[r:`临床表现`]-(n)
    #     WHERE e.name = $disease_name
    #     RETURN collect(n.name) AS neighbor_entities
    #     """
    #     result = self.session.run(query, disease_name=disease_name)

    #     neighbor_list = []
    #     for record in result:
    #         neighbors = record["neighbor_entities"]
    #         neighbor_list.extend(neighbors)

    #     return neighbor_list
    
    def find_shortest_path(self, start_entity_name, end_entity_name):
    
        query = """
        MATCH (start_entity), (end_entity)
        WHERE start_entity.name = $start_entity_name AND end_entity.name = $end_entity_name
        MATCH p = shortestPath((start_entity)-[*..10]-(end_entity))
        RETURN p
        """
        result = self.session.run(
            query,
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        # 用paths记录路径的字符串表示和对应的路径长度，方便后续排序并输出
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            path_len = len(path.relationships)
            entities = []
            relations = []
            if path is not None:
                for i in range(len(path.nodes)):
                    node = path.nodes[i]
                    entity_name = node["name"]
                    entities.append(entity_name)
                    if i < len(path.relationships):
                        relationship = path.relationships[i]
                        relation_type = relationship.type
                        relations.append(relation_type)
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i]
                    path_str += "->" + relations[i] + "->"
            paths.append((path_str, path_len))

        if len(paths) != 0:
            # 按照长度排序
            paths.sort(key=lambda x: x[1])

            # 取最短的那条路径输出
            return paths[0]
        else:
            return ("", 0)