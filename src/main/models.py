from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import modelscope
import torch
import asyncio
from http import HTTPStatus
import platform
import time

from dashscope import Generation
from dashscope.aigc.generation import AioGeneration

class ChatModel:
    def __init__(self, model_type, model_name_or_path, model_version):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.model_version = model_version
        self.load_model(model_type, model_name_or_path, model_version)
        

    def load_model(self, model_type, model_name_or_path, model_version):
        """ 加载模型 """

        # baichuan-13b量化加载
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
        # chat_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code

        if model_type == "glm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.chat_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            # if args.model_version == "-6b":
            #     chat_model = chat_model.half().cuda()
            self.chat_model = self.chat_model.cuda()
            self.chat_model.eval()
        elif model_type == "baichuan":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
            self.chat_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.chat_model.generation_config.do_sample = False
            # Generation_Config = GenerationConfig.from_pretrained(model_name_or_path)
            # Generation_Config.do_sample = False
            # self.chat_model.generation_config = Generation_Config
            self.chat_model = self.chat_model.cuda().eval()
        elif model_type == "qwen":
            if model_version == "qwen-7b-chat":
                tokenizer = modelscope.AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
                chat_model = modelscope.AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto",    torch_dtype=torch.bfloat16,    trust_remote_code=True).eval()
                chat_model.generation_config.do_sample = False
                self.chat_model = chat_model
                self.tokenizer = tokenizer
            elif model_version == "qwen-14b-chat":
                tokenizer = modelscope.AutoTokenizer.from_pretrained("qwen/Qwen-14B-Chat-Int4", trust_remote_code=True)
                chat_model = modelscope.AutoModelForCausalLM.from_pretrained("qwen/Qwen-14B-Chat-Int4", device_map="auto",  trust_remote_code=True).eval()
                chat_model.generation_config.do_sample = False
                self.chat_model = chat_model
                self.tokenizer = tokenizer
        elif model_type == "qwen_api":
            self.seed = 1234
    
    def chat_(self, messages):
        """ 自定义chat接口 """
        if self.model_type == "glm" or self.model_type == "qwen":
            return self.chat_glm(messages)
        elif self.model_type == "baichuan":
            return self.chat_baichuan(messages)
        elif self.model_type == "qwen_api":
            api_result = self.chat_qwenapi(messages)
            if api_result[0] == "error, no correct response":
                # 再次尝试调用api
                time.sleep(1)
                retry_result = self.chat_qwenapi(messages)
                # 如果再次调用api还是失败，则抛出异常
                if retry_result[0] == "error, no correct response":
                    raise ValueError("api调用失败")
                else:
                    return retry_result
            else:
                return api_result
        else:
            raise ValueError("model_type must be in ['glm', 'baichuan', 'qwen']")
        
    def chat_glm(self, messages):
        """ glm调用chat，注意qwen的调用方式和glm是一样的(估计是都参考了llama) """
        # if len(prompt) > 2048:
        #     prompt = prompt[:2048]
        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        if old_history is not None:
            response, history = self.chat_model.chat(self.tokenizer,
                                           prompt,
                                           do_sample=False,
                                           temperature=1.0,
                                           top_p = 1.0,
                                           repetition_penalty = 1.1,
                                           history = old_history
                                           )
        else:
            response, history = self.chat_model.chat(self.tokenizer,
                                           prompt,
                                           do_sample=False,
                                           temperature=1.0,
                                           top_p = 1.0,
                                           repetition_penalty = 1.1,
                                           history = old_history
                                           )
        return response, history
    
    def chat_baichuan(self, texts):
        """ baichuan调用chat """
        if len(texts) == 1:
            prompt = texts[0]
            messages = []
            messages.append({"role": "user", "content": prompt})
        else:
            prompt = texts[0]
            # messages = []
            messages = texts[1]
            messages.append({"role": "user", "content": prompt})
        response = self.chat_model.chat(self.tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        return response, messages
    
    def chat_qwenapi(self, messages):
        """ qwenapi调用chat """

        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        messages = [{'role': 'system', 'content': 'You are an experienced medical expert.'},
                {'role': 'user', 'content': prompt}]
        
        # qwen1.5-72b-chat-api, 把后面的-api去掉，才是模型的名字
        response = Generation.call(model=self.model_version[:-4],
                                   messages=messages,
                                   seed=self.seed,
                                   result_format='message')
        
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'], response['output']['choices'][0]['message']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return "error, no correct response", "error, no history"

        # return response
