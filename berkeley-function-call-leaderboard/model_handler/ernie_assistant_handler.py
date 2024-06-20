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
import os, time, json
import requests


def run_thread(assistant_id, messages, tools):
    url = f"{os.environ['ASSISTANTS_API_HOST']}/v1/threads/runs"
    headers = {"appId": "123456", "Content-Type": "application/json"}
    data = {
        "blocking": True,
        "instructions": SYSTEM_PROMPT_FOR_CHAT_MODEL,
        "assistant_id": assistant_id,
        "thread": {"messages": messages},
        "tools": tools,
    }

    # 传入多条数据
    if len(messages) > 1:
        data["metadata"]["option.allow.create.multiple.messages"] = "true"

    response = requests.post(
        url=url, headers=headers, json=data
    ).json()  # 这里json了， 注意这初始返回的是appbuilder的RunResult,没有token消耗的信息
    if "status" in response and response["status"] == "failed":
        print(response, data)

    if "final_answer" in response and response["final_answer"]:
        return response["final_answer"]["message"]["content"]["text"]
    elif "required_action" in response and response["required_action"]:
        return response["required_action"]["submit_tool_outputs"]["tool_calls"][0][
            "function"
        ]  # https://github.com/baidubce/app-builder/blob/b92c0d2a14b944f2b13f0b6ada805284508c7da5/appbuilder/core/assistant/type/thread_type.py#L199
    # 直接是一个class FunctionCall(BaseModel)的结构： {"name": "function_name", "arguments": "json_string", "output": "json_string"}
    return response


class ErnieAssistantHandler(BaseHandler):
    """用于EBT 3.5和4.0的助手模型"""

    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OpenAI

    def inference(self, prompt, functions, test_category):
        prompt = augment_prompt_by_languge(prompt, test_category)
        functions = language_specific_pre_processing(functions, test_category, True)
        message = [{"role": "user", "content": "Questions:" + prompt}]
        oai_tool = convert_to_tool(
            functions, GORILLA_TO_OPENAPI, self.model_style, test_category, True
        )  # 这里跟ernie-3.5
        start_time = time.time()
        # Hard-code assistant id for appropriate model
        if self.model_name == "EBT-3.5-Assistant-FC":
            assistant_id = "asst_e0e486df95b74ad7baba67c9ebf4e905"
        elif self.model_name == "EBT-4.0-Assistant-FC":
            assistant_id = "asst_1aec940065fb4f81b66d89fc04011ab4"
        else:
            raise NotImplementedError("Invalid Model")

        result = run_thread(
            assistant_id=assistant_id, messages=message, tools=oai_tool
        )  # DICT
        latency = time.time() - start_time

        metadata = {}
        metadata["input_tokens"] = 0
        metadata["output_tokens"] = 0
        metadata["latency"] = latency
        return result, metadata

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result, language)
        else:  # 走这个

            # {"idx": 0, "result": {"name": "calculate_triangle_area", "arguments": "{\"base\":10,\"height\":5,\"unit\":\"units\"}"}, "input_token_count": 0, "output_token_count": 0, "latency": 5.1565101146698}
            #

            # {"result": {"name": "calculate_triangle_area", "arguments": "{\"base\":10,\"height\":5,\"unit\":\"units\"}"}}
            decoded_output = []
            name = result["name"]
            if isinstance(result["arguments"], str):
                params = json.loads(result["arguments"])
            else:
                params = result["arguments"]

            # params = json.loads(str(result["arguments"]))
            # params = result["arguments"]
            if language == "Python":
                pass
            else:
                # all values of the json are casted to string for java and javascript
                for key in params:
                    params[key] = str(params[key])
            decoded_output.append({name: params})
            # 变成 [{'calculate_triangle_area': {'base': 10, 'height': 5, 'unit': 'units'}}]
        return decoded_output

    def decode_execute(self, result):
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
