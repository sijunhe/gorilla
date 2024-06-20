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
import erniebot
import re
import os, time, json


# def clean_string(result):
#     # 使用正则表达式去除开头和结尾的 ```json, ```python, ``` 等字符，以及换行符和空白字符
#     cleaned_result = re.sub(
#         r"^\s*```(?:json|python)?\s*|\s*```\s*$", "", result, flags=re.DOTALL
#     )
#     # 去除首尾的空白字符
#     cleaned_result = cleaned_result.strip()
#     return cleaned_result


# def clean_result_string(result):
#     # 去除开头的换行符和空白字符以及可能的`json`或`python`标记
#     cleaned_result = re.sub(r"^\s*```(?:json|python)?\s*", "", result, flags=re.DOTALL)
#     # 去除结尾的换行符和空白字符以及可能的```标记
#     cleaned_result = re.sub(r"\s*```\s*$", "", cleaned_result, flags=re.DOTALL)
#     # 去除中间的换行符和多余空白字符
#     cleaned_result = re.sub(r"\s*\n\s*", "", cleaned_result)
#     # 去除外部的引号并保持内部的内容
#     cleaned_result = re.sub(r"\[\"(.*?)\"\]", r"[\1]", cleaned_result)
#     return cleaned_result


def clean_result_string(result):
    """
    清理开头和结尾的无关字符。
    保留 [] 内部的内容不变，无论其中是否包含转义字符。
    Args:
        result (str): 需要处理的结果字符串。

    Returns:
        str: 处理后的结果字符串。

    """
    # 处理开头和结尾的```json或```python标记（如果存在）
    cleaned_result = re.sub(
        r"^\s*```(?:json|python)?\s*|\s*```\s*$", "", result, flags=re.DOTALL
    )
    # 去除中间的换行符和多余空白字符
    cleaned_result = re.sub(r"\s*\n\s*", "", cleaned_result)
    # 去除外部的引号并保持内部的内容
    cleaned_result = re.sub(r"\[\"(.*?)\"\]", r"[\1]", cleaned_result)
    return cleaned_result


class ErnieHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OpenAI

    def inference(self, prompt, functions, test_category):
        if "FC" not in self.model_name:
            prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(
                functions, test_category, False
            )
            # if "multiple" in test_category:
            #     functions = functions
            # elif
            # functions = functions[0]
            # 去除list，直接以dict形式传入， 加入message后下面变成str(dict)不能去除list，就算是list是一个也得带着
            message = [
                {
                    "role": "user",
                    "content": "Questions:"
                    + USER_PROMPT_FOR_CHAT_MODEL.format(
                        user_prompt=prompt, functions=str(functions)
                    ),
                },
            ]
            start_time = time.time()
            response = erniebot.ChatCompletion.create(
                messages=message,
                system=SYSTEM_PROMPT_FOR_CHAT_MODEL,  # ernie从这里传入system
                model=self.model_name,  # ernie-3.5-8k-0205
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            latency = time.time() - start_time
            result = response.get_result()

            # 没有使用function显示传入，因此返回str，比较脏：'```\n[geometry.calculate_area_circle(radius=5)]\n```'
            # print(f"before clearn:", result)
            result = clean_result_string(result)
            # print(f"after clearn:", result)
            # print(type(result))  # 输出: <class 'str'>

        else:
            prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(functions, test_category, True)
            if type(functions) is not list:
                functions = [functions]
            message = [{"role": "user", "content": "Questions:" + prompt}]
            oai_tool = convert_to_tool(
                functions, GORILLA_TO_OPENAPI, self.model_style, test_category, True
            )
            oai_tool = [i["function"] for i in oai_tool]
            start_time = time.time()
            if len(oai_tool) > 0:
                response = erniebot.ChatCompletion.create(
                    messages=message,
                    model=self.model_name.replace("-FC", ""),
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p,
                    functions=oai_tool,
                )
            else:
                response = erniebot.ChatCompletion.create(
                    messages=message,
                    model=self.model_name.replace("-FC", ""),
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p,
                )  # 返回的是ChatCompletionResponse对象
            latency = time.time() - start_time
            result = response.get_result()  # 这里就是返回一个dict
            # eg:{'name': 'get_current_temperature',
            #       'thoughts': '用户想知道深圳市今天的温度，我可以使用get_current_temperature工具来获取信息。',
            #         'arguments': '{"location":"深圳","unit":"摄氏度"}'}
            # except:
            #     result = response.choices[0].message.content
        metadata = {}
        metadata["input_tokens"] = response.usage["prompt_tokens"]
        metadata["output_tokens"] = response.usage["completion_tokens"]
        metadata["latency"] = latency
        return result, metadata

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(
                result, language
            )  # result输入的时候是str，返回的是list
        else:
            decoded_output = []
            name = result["name"]
            params = json.loads(result["arguments"])
            if language == "Python":
                pass
            else:
                # all values of the json are casted to string for java and javascript
                for key in params:
                    params[key] = str(params[key])
            decoded_output.append(
                {name: params}
            )  # 变成 [{name: {params}}] ， 如[{'math_triangle_area_heron': {'side1': 3, 'side2': 4, 'side3': 5}}]，把key去掉了，直接function_name: {params}
            print(decoded_output)
        return decoded_output

    def decode_execute_0(self, result):
        """sijun版"""
        if type(result) == dict:
            result = [result]
        execution_list = []
        for function_call in result:
            try:
                function_name = function_call["name"]
                function_args = function_call["arguments"]
                execution_list.append(
                    f"{function_name}({','.join([f'{k}={repr(v)}' for k,v in json.loads(function_args).items()])})"
                )
            except Exception as e:
                print(function_call, str(e))
        return execution_list

    def decode_execute(self, result):
        # for executable categoryde
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
