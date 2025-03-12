import json
import os
import time
import requests

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import platform
from pathlib import Path
import time

class TCPKeepAliveHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        
        # Platform-specific TCP keepalive settings
        system = platform.system()
        if system == 'Linux':
            socket_options.extend([
                (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 45),
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)
            ])
        elif system == 'Darwin':  # macOS
            # TCP_KEEPALIVE is Apple's equivalent of TCP_KEEPIDLE
            TCP_KEEPALIVE = 0x10
            socket_options.extend([
                (socket.IPPROTO_TCP, TCP_KEEPALIVE, 45),
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)
            ])
        
        kwargs['socket_options'] = socket_options
        return super().init_poolmanager(*args, **kwargs)

def create_session():
    """Create a session with retry strategy and keepalive settings"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504, 104],
        allowed_methods=["POST"],
        raise_on_status=False,
        respect_retry_after_header=True
    )
    
    # Mount the custom adapter
    adapter = TCPKeepAliveHTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Define a retry decorator with exponential backoff and a maximum number of attempts
@retry(
    stop=stop_after_attempt(3),  # Maximum number of retry attempts
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    retry=retry_if_exception_type(ConnectionResetError),  # Retry only on ConnectionResetError
)
def run_thread(messages, tools):
    url = f"{os.environ["ASSISTANT_API_HOST"]}/v1/threads/runs"
    headers = {
        "appId": "123456",
        "Content-Type": "application/json",
        "Authorization": "testldx/1722333193666/9c2285103869105c43191fd48abb420eadf6286de45c7576e3bd3130168a606f"
    }
    data = {
        "blocking": True,
        "model": "ERNIE-4.5-single-custom-yiyanweb-eval",
        "assistant_id": "asst_8402576dad614c2ca66b75f7f951e1b9",
        "thread": {
            "messages": messages
        },
        "tools": tools,
        "parallel_tool_calls": True,
    }

    # Add metadata if there are multiple messages
    if len(messages) > 1:
        data["metadata"] = {"option.allow.create.multiple.messages": "true"}

    try:
        session = create_session()
        response = session.post(url=url, headers=headers, json=data, timeout=(30, 600)).json()
    except ConnectionResetError as e:
        print(f"Connection reset by peer: {e}")
        raise  # Re-raise the exception to trigger the retry
    
    if "status" in response and response["status"] == "failed":
        return response["last_error"]["code"]
    if "final_answer" in response and response["final_answer"]:
        return response['final_answer']["message"]["content"][0]["text"]['value']
    elif "required_action" in response and response['required_action']:
        return response['required_action']["submit_tool_outputs"]["tool_calls"]
    
    return response

def convert_dict_to_object(data):
    """
    Recursively converts all instances of 'type': 'dict' to 'type': 'object' in a nested structure.
    
    Args:
        data: The data structure to process (can be dict, list, or other primitive types)
        
    Returns:
        The processed data structure with all 'dict' types replaced with 'object'
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'type' and value == 'dict':
                data[key] = 'object'
            else:
                data[key] = convert_dict_to_object(value)
        return data
    elif isinstance(data, list):
        return [convert_dict_to_object(item) for item in data]
    else:
        return data

class ErnieHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.is_fc_model = True

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        start_time = time.time()
        
        
        end_time = time.time()


        if len(tools) > 0:
            api_response = run_thread(messages=message, tools=tools)
        else:
            api_response = run_thread(messages=message, tools=[])
        
        return api_response, end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        tools = convert_dict_to_object(tools)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call["function"]["name"]: func_call["function"]["arguments"]}
                for func_call in api_response
            ]
            tool_call_ids = [func_call["id"] for func_call in api_response]
        except:
            model_responses = api_response
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response
        
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": 0,
            "output_token": 0,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        # OpenAI reasoning models don't support temperature parameter
        # Beta limitation: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        if "o1" in self.model_name or "o3-mini" in self.model_name:
            return self.generate_with_backoff(
                messages=inference_data["message"],
                model=self.model_name,
            )
        else:
            return self.generate_with_backoff(
                messages=inference_data["message"],
                model=self.model_name,
                temperature=self.temperature,
            )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data
