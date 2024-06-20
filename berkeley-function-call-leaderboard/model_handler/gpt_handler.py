from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    convert_to_function_call,
    augment_prompt_by_languge,
    language_specific_pre_processing,
    ast_parse,
)
from model_handler.constant import (
    GORILLA_TO_OPENAPI,
    GORILLA_TO_PYTHON,
    USER_PROMPT_FOR_CHAT_MODEL,
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
)
from openai import OpenAI
import os, time, json
from typing import List, Dict, Union


class OpenAIHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def inference(
        self, prompt: str, functions: Union[List[Dict], List[str]], test_category: str
    ):

        if (
            "FC" not in self.model_name
        ):  # 对于没有fc的模型，只能使用chat模式，从prompt传入tools，name不用额外处理删除FC
            # 非fc版本--prompt TODO justin
            prompt = augment_prompt_by_languge(
                prompt, test_category
            )  # 对于java等语言，需要在prompt中加入this is java ....

            # function 初始传入是个list，每个元素是一个dict，经过language_specific_pre_processing后，变成一个dict
            # 原始： [{"name": "geometry.circumference", "description": "Calculate the circumference of a circle given the radius.", "parameters": {"type": "dict", "properties": {"radius": {"type": "integer", "description": "The radius of the circle."}, "units": {"type": "string", "description": "Units for the output circumference measurement. Default is 'cm'."}}, "required": ["radius"]}]
            functions = language_specific_pre_processing(
                functions, test_category, False
            )  # 对于java等语言，把上面的dict的properties改一下，不重要,返回的还是List[Dict]形式
            message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_FOR_CHAT_MODEL,
                },
                {
                    "role": "user",
                    "content": "Questions:"
                    + USER_PROMPT_FOR_CHAT_MODEL.format(
                        user_prompt=prompt, functions=str(functions)
                    ),  # 这里的functions是一个dict，变成str
                },
            ]
            start_time = time.time()
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            latency = time.time() - start_time
            result = response.choices[0].message.content
        else:  # 对于自带fc功能的模型，把tools传入接口，如gpt4preview， 要看有没有tools可用，但无论如何都要去掉-FC才是valid model name
            prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(functions, test_category, True)
            if type(functions) is not list:
                functions = [functions]
            message = [{"role": "user", "content": "Questions:" + prompt}]
            oai_tool = convert_to_tool(
                functions, GORILLA_TO_OPENAPI, self.model_style, test_category, True
            )
            start_time = time.time()
            if len(oai_tool) > 0:  # 有fc能力的模型，且有tools可用
                # response 格式： https://platform.openai.com/docs/api-reference/chat/object
                response = self.client.chat.completions.create(
                    messages=message,
                    model=self.model_name.replace("-FC", ""),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    tools=oai_tool,
                )
            else:  # 有fc能力的模型，但没有tools可用
                response = self.client.chat.completions.create(
                    messages=message,
                    model=self.model_name.replace("-FC", ""),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            latency = time.time() - start_time
            try:  # 这是return list of func name and arguments dict,
                # 如 [{'func1': "{'arg1': 'val1', 'arg2': 'val2'}""}, {'func2': "{'arg1': 'val1'}"}]
                result = [
                    {func_call.function.name: func_call.function.arguments}
                    for func_call in response.choices[0].message.tool_calls
                ]  # https://platform.openai.com/docs/api-reference/chat/create?lang=python  格式参考这个，arguments是一个str
            except:
                result = response.choices[0].message.content
        metadata = {}
        metadata["input_tokens"] = response.usage.prompt_tokens
        metadata["output_tokens"] = response.usage.completion_tokens
        metadata["latency"] = latency
        return result, metadata

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result, language)
        else:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                if language == "Python":
                    pass
                else:
                    # all values of the json are casted to string for java and javascript
                    for key in params:
                        params[key] = str(params[key])
                decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result)
            execution_list = []
            for function_call in decoded_output:
                for key, value in function_call.items():
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                    )
            return execution_list
        else:
            function_call = convert_to_function_call(result)
            return function_call
