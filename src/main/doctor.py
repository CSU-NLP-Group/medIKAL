
class Doctor:
    def __init__(self, chat_model):
        self.chat_model = chat_model

    def general_info_summary(self, chief_complaint, current_medical_history, disease_history):
        prompt = "你是一名优秀的AI医学专家，你可以基于患者的病历内容总结用于诊断的关键信息。以下是一个真实的某患者的电子病历的部分内容，请你仔细阅读以下内容，了解患者的基本情况。\n" \
                   + '"""\n' \
                   + f"【主诉】：{chief_complaint}\n"\
                   + f"【现病史】：{current_medical_history}\n" \
                   + f"【既往史】：{disease_history}\n" \
                   + '"""\n' \
                   + "##任务：\n{请你根据上述内容，总结出对于诊断和治疗有用的关键信息，并生成一份总结报告，报告有具体的格式要求}。\n"\
                   + "##报告格式要求：\n{请按照以下格式，填写“[]”处的内容，完成报告。语言请尽可能地简洁。}\n"\
                   + "1.出现症状：[]\n"\
                   + "2.近期就诊经历：[](没有则填无)\n"\
                   + "3.既往疾病史：[](没有则填无)\n" \
                   + "4.既往手术史：[](没有则填无)\n" \
                   + "5.药物使用情况：[](没有则填无)\n" \
                   + "你的总结：\n"

        fst_rd_summary, fst_rd_history = self.chat_model.chat_([prompt])
        return fst_rd_summary, fst_rd_history
    
    def examination_summary(self, body_check, auxiliary_exam):
        prompt =  "你是一名优秀的AI医学专家，你可以基于患者的检查结果总结用于诊断的关键信息。" \
                + "患者的检查结果如下所示：\n"\
                + '"""\n' \
                + f"【查体结果】：{body_check}\n" \
                + f"【辅助检查结果】：{auxiliary_exam}\n" \
                + '"""\n' \
                + "目前患者的检查结果有很多冗余的内容，请你对上述检查结果进行总结。\n"\
                + "##任务：\n{请你对上述患者检查结果进行总结和概括，保留对诊断有用的信息，删除那些对诊断意义不大的内容。}\n"\
                + "##要求：\n{请按照以下格式，填写“[]”处的内容，完成报告。语言清尽可能地简洁。}\n"\
                + "1.查体结果：[]\n"\
                + "2.辅助检查结果：[]\n"\
                + "你的总结：\n"
    
        scd_rd_summary, scd_rd_history = self.chat_model.chat_([prompt])

        physical_exam = scd_rd_summary.split("查体结果：").split("辅助检查结果：")[0]
        if physical_exam:
            scd_rd_summary = physical_exam
        
        aug_exam = scd_rd_summary.split("辅助检查结果：").split("\n")[0]
        if aug_exam:
            scd_rd_summary += aug_exam

        return scd_rd_summary, scd_rd_history

    def direct_diagnos(self, fst_rd_summary, scd_rd_summary, topn=5):
        # diagnos_template = "预测疾病1：\n预测疾病2：\n预测疾病3：\n预测疾病4：\n预测疾病5：\n"
        diagnos_template = ""
        for i in range(topn):
            diagnos_template += f"预测疾病{i+1}：\n"
        prompt = "你是一名优秀的AI医学专家。你可以根据患者的病情进行疾病诊断。\n" \
                + f"患者的基本情况：{fst_rd_summary}\n" \
                + f"患者的检查结果：{scd_rd_summary}\n" \
                + f"###任务说明：请你根据患者的症状、就诊经历、既往病史和检查结果，预测患者可能患有哪些疾病(你可以给出top-{topn}个可能的预测)，格式要求如下。请只输出预测结果，不要输出其它内容。\n"\
                + f"###格式要求：{diagnos_template}\n"
        
        direct_diagnos_result, direct_diagnos_history = self.chat_model.chat_([prompt])
        return direct_diagnos_result, direct_diagnos_history
    
    def analysis(self, disease, formatted_knowledge, chief_complaint, dis_history, drug_history, auxiliary_exam):
        analysis_template = f"候选疾病*{disease}*可能性评估：\n" \
                          + "1.与患者主诉吻合度得分：[?](满分10分)\n" \
                          + "2.与患者既往病史关联程度得分：[?](满分10分)\n" \
                          + "3.与患者既往药物使用关联程度得分：[?](满分10分)\n"\
                          + "4.与患者检查结果关联程度得分：[?](满分10分)\n" \
                          + "5.请分析所提供的关联性信息中是否存在错误或者误导性的信息? [?]\n"\
                          + "6.该疾病是否能作为诊断结果：[?](是/否)\n"

        prompt = "你是一名优秀的AI医学专家。你可以根据患者的症状、就诊经历、既往病史和检查结果，对现有的诊断结果的合理性进行分析。请你仔细阅读以下内容，了解患者的基本情况。\n"\
                + f"患者的基本情况：{chief_complaint}\n" \
                + f"患者的既往病史：{dis_history}\n" \
                + f"患者的既往用药史：{drug_history}\n" \
                + f"患者的检查结果：{auxiliary_exam}\n" \
                + f"根据患者的情况，一名医生的初步诊断为：{disease}。\n"\
                + "你需要考虑这一诊断的合理性。为此，你查询了医学知识图谱并获得了以下的关联信息：\n" \
                + f"{formatted_knowledge}\n" \
                + "##任务：{请你根据我们提供给你的知识图谱知识，对该疾病进行评估。}\n" \
                + f"##格式要求：{analysis_template}\n" \
                + "你的分析：\n"

        analysis_result, analysis_history = self.chat_model.chat_([prompt])
        return analysis_result, analysis_history
