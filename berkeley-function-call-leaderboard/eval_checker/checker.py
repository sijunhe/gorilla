from js_type_converter import js_type_converter
from java_type_converter import java_type_converter
from model_handler.constant import (
    UNDERSCORE_TO_DOT,
    JAVA_TYPE_CONVERSION,
    JS_TYPE_CONVERSION,
)
from eval_checker_constant import REAL_TIME_MATCH_ALLOWED_DIFFERENCE
from custom_exception import NoAPIKeyError
import re
import requests  # Do not remove this import even though it seems to be unused. It's used in the executable_checker_rest function.
import time
import json

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# This is the list of types that we need to recursively check its values
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]


EVAL_GROUND_TRUTH_PATH = (
    "./rest-eval-response_v5.jsonl"  # Ground truth file for v5 for rest execution
)
with open(EVAL_GROUND_TRUTH_PATH, "r") as f:
    EVAL_GROUND_TRUTH = f.readlines()


#### Helper functions for AST ####
def find_description(func_descriptions, name):
    #  [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........
    # If func_descriptions is a list, this is the multiple or multiple_parallel case
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if "name" in func_description:  # 有可能是内置函数，没有name 这个key
                if func_description["name"] in name:
                    return func_description
        return None
    else:
        # This is the parallel case, there is no need to loop through the list, as there is only one function
        return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def convert_func_name(function_name, model_name: str):
    model_name_escaped = model_name.replace("_", "/")
    if "." in function_name:
        if model_name_escaped in UNDERSCORE_TO_DOT:
            # OAI does not support "." in the function name so we replace it with "_". ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
            # This happens for OpenAI, Mistral, and Google models
            return re.sub(r"\.", "_", function_name)
    return function_name


def type_checker(
    param: str,  # 当前检查的 param的名字
    value,  # 模型输出的 arg的值
    possible_answer: list,  # ["单双曲铝单板价格"] or [[1, 2, 3]]
    expected_type_description: str,  # arg原始定义的type，
    expected_type_converted,  # arg原始定义的type，转换python后的type
    nested_type_converted,
):
    # NOTE: This type checker only supports nested type checking for one level deep.
    # We didn't implement recursive type checking for nested types, as it's not needed for the current use case and it's very complex.

    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False
    # check for the case where a variable is used instead of a actual value.
    # use the type in possible_answer as the expected type
    possible_answer_type = get_possible_answer_type(
        possible_answer
    )  # -> for single item
    # if possible_answer only contains optional parameters, we can't determine the type
    if possible_answer_type != None:
        # we are being precise here.
        # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
        if possible_answer_type != expected_type_converted:
            is_variable = True

    # value is the same type as in function description
    if type(value) == expected_type_converted:
        # We don't need to do recursive check for simple types
        if nested_type_converted == None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True  # Each parameter should match to at least one possible answer type.
                # Here, we assume that each item should be the same type. We could also relax it.
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break

                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. Expected outer type {expected_type_description} with inner type {str(nested_type_converted)}. Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    # value is not as expected, check for the case where a variable is used instead of a actual value
    # use the type in possible_answer as the expected type
    possible_answer_type = get_possible_answer_type(possible_answer)
    # if possible_answer only contains optional parameters, we can't determine the type
    if possible_answer_type != None:
        # we are being precise here.
        # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, got {type(value).__name__}. Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str):
    # This function standardizes the string by removing all the spaces, ",./-_*^" punctuation, and converting it to lowercase
    # It will also convert all the single quotes to double quotes
    # This is used to compare the model output with the possible answers
    # We don't want to punish model for answer like April 1, 2024 vs April 1,2024, vs April 1 2024
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def string_checker(param: str, model_output: str, possible_answer: list):
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }

    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list):
    # Convert the tuple to a list

    standardize_model_output = list(model_output)

    # If the element in the list is a string, we need to standardize it
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer = []
    # We also need to standardize the possible answers
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(
                    standardize_string(possible_answer[i][j])
                )
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }

    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list):
    # This function works for simple dictionaries, as well as dictionaries with nested dictionaries
    # model_output: {'field': 'aaaa', 'operation': '>', 'value': '25'}
    # possible_answers : [{'field': ['age'], 'operation': ['>'], 'value': ['25']}]
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        # [{'field': [...], 'operation': [...], 'value': [...]}]
        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

        flag = True

        possible_answer = possible_answers[i]
        # {'field': ['age'], 'operation': ['>'], 'value': ['25']}
        # possible_anwer is a single dictionary
        if len(model_output.keys()) != len(possible_answer.keys()):
            result["valid"] = False
            result["error"].append("Wrong number of parameters for dictionary.")
            result["error_type"] = "value_error:dict_items"
            flag = False
            continue

        for key, value in model_output.items():
            # model_output: {'field': 'age', 'operation': '>', 'value': '25'}
            # 第一轮key vale = filed, 'aaaa' from model
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            expected_values = possible_answer[key]
            # expected_values: ['age']
            if isinstance(expected_values, dict):
                result = dict_checker(param, value, [expected_values])
                if not result["valid"]:
                    flag = False
                    break
            else:
                standardize_value = value  # 这是model的value 如 'aaa' / '>' / '25'
                # If the value is a string, we need to standardize it
                if type(value) == str:
                    standardize_value = standardize_string(
                        value
                    )  # 不变，还是 'aaa' / '>' / '25'
                # We also need to standardize the possible answers
                standardize_possible_answer = []
                for i in range(
                    len(possible_answer[key])
                ):  # 当前key是field， possilbe_answer= {'field': ['age'], 'operation': ['>'], 'value': ['25']}
                    if type(possible_answer[key][i]) == str:
                        standardize_possible_answer.append(
                            standardize_string(possible_answer[key][i])
                        )
                    else:
                        standardize_possible_answer.append(possible_answer[key][i])
                # field完事后，standardize_possible_answer= ['age'] , 只有一个value
                # if standardize_value not in standardize_possible_answer:
                #     result["valid"] = False
                #     result["error"].append(
                #         f"Invalid value for parameter {repr(key)}: {repr(value)}. Expected one of {standardize_possible_answer}."
                #     )
                #     result["error_type"] = "value_error:dict_value"
                #     flag = False
                #     break

                for possible_answer_item in standardize_possible_answer:
                    if type(standardize_value) != type(possible_answer_item):
                        result["valid"] = False
                        result["error"].append(
                            f"Invalid value's type for parameter {repr(key)}: {repr(value)}. Expected one of {standardize_possible_answer}."
                        )
                        result["error_type"] = "value_error:dict_value's_type"
                        flag = False
                        break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list):
    # This function takes in a list of dictionaries and checks if each dictionary is valid
    # The order of the dictionaries in the list must match the order of the possible answers

    # param: conditions
    # value(model_output): [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]
    # possible_answer_item[param](possible_answers:
    #           [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]

    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True  # True means so far, all dictionaries are valid

        # Only proceed if the number of dictionaries in the list matches the number of dictionaries in the possible answers
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(
                param,
                model_output[
                    dict_index
                ],  # {'field': 'aaaa', 'operation': '>', 'value': '25'}
                [
                    possible_answers[answer_index][dict_index]
                ],  # [{'field': ['age'], 'operation': ['>'], 'value': ['25']}]
            )
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}

    return result


def simple_function_checker(
    func_description_item: dict,
    model_result_item: dict,
    possible_answer_item: dict,
    language: str,
    model_name: str,
):
    """
    func_description_item,   即list of tool pool，
    这里是simple function，只可能有一个func description，没有很多doc
    [{"name": "ExpertSearch", "description": "私......... ,"parameters": {"type": "object", "properties": {"query": {"type": ..
    [{'name': 'database_query', 'description': 'Query the database based on certain conditions.', 'parameters': {'type': 'dict', 'properties': {'table': {'type': 'string', 'description': 'Name of the table to query.'}, 'conditions': {'type': 'array', 'items': {'type': 'dict', 'properties': {'field': {'type': 'string', 'description': 'The field to apply the condition.'}, 'operation': {'type': 'string', 'description': 'The operation to be performed.'}, 'value': {'type': 'string', 'description': 'The value to be compared.'}}, 'required': ['field', 'operation', 'value']}, 'description': 'Conditions for the query.'}}, 'required': ['table', 'conditions']}}]

    model_result_item,  上一步变成了 单个： dict，
    {"ExpertSearch": {"query": "单双曲铝单板价格"}}
    {'database_query': {'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}}

    possible_answer_item, 不可能是内置函数，
    # 已经从上一步转换了：{"name": "ExpertSearch", "arguments": {"query": "单双曲铝单板价格"}}
     变成：      {"ExpertSearch": {"query": ["单双曲铝单板价格"]}
     传入的时候是
     {'database_query': {'table': ['user'], 'conditions': [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]}}

    """
    # Extract function name and parameters details

    ## 这里要转换一下！！！
    # possible_answer_item = possible_answer_item["arguments"]
    possible_answer_item = list(possible_answer_item.values())[
        0
    ]  # 提取 values 后： {'table': ['user'], 'conditions': [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]}

    # 因为是simple function ， 只会有一个true tgt function
    if isinstance(func_description_item, list):
        func_name = func_description_item[0]["name"]
        param_details = func_description_item[0]["parameters"]["properties"]
        required_params = func_description_item[0]["parameters"][
            "required"
        ]  # 所有能用的param：
    else:
        func_name = func_description_item["name"]
        param_details = func_description_item["parameters"]["properties"]
        required_params = func_description_item["parameters"]["required"]

    # func_name = func_description_item[0]["name"]

    # 所有必须的param,

    # Initialize a result dictionary
    result = {
        "valid": True,
        "error": [],
        "error_type": "simple_function_checker:unclear",
    }

    func_name = convert_func_name(func_name, model_name)

    # 1. Check if function name matches
    if (
        func_name not in model_result_item
    ):  # true function not in model output, infact for simple, function description is only one tool
        result["valid"] = False
        result["error"].append(
            f"Function name {repr(func_name)} not found in model output."
        )
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_result_item[func_name]
    # 拿出模型输出的params:{'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}

    # 2. Check for required parameters in model output
    for param in required_params:  # iterate over list of str
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    # 3. Validate types and values for each parameter in model output
    for (
        param,
        value,
    ) in (
        model_params.items()
    ):  # 模型的参数和值：{'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}

        if (
            param not in param_details or param not in possible_answer_item
        ):  # {"query": ["单双曲铝单板价格"]
            result["valid"] = False
            result["error"].append(
                f"Unexpected parameter: {repr(param)}."
            )  # hallucination 了， 出现了不可用的param
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]  # 拿出当前arg1的详细信息
        # {"query": {"type": "string", "description": "搜索问题"}}
        #  复杂： "preferences": {"type": "array", "items": {"type": "string", "enum": ["A", "T", "C", "G"]}, "description": "Preferred nucleotides to include more frequently in the DNA sequence."}
        expected_type_description = full_param_details[
            "type"
        ]  # 当前arg1的定义的type ,array

        is_variable = False
        nested_type_converted = None

        if language == "Java":
            expected_type_converted = JAVA_TYPE_CONVERSION[expected_type_description]

            if expected_type_description in JAVA_TYPE_CONVERSION:
                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JAVA_TYPE_CONVERSION[nested_type]
                    value = java_type_converter(
                        value, expected_type_description, nested_type
                    )
                    print(f"after covert java NESTED_CONVERSION_TYPE_LIST: {value}")
                else:
                    value = java_type_converter(value, expected_type_description)
                    print(f"after covert java NOT NESTED_CONVERSION_TYPE_LIST: {value}")

        elif language == "JavaScript":
            expected_type_converted = JS_TYPE_CONVERSION[expected_type_description]

            if expected_type_description in JS_TYPE_CONVERSION:
                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JS_TYPE_CONVERSION[nested_type]
                    value = js_type_converter(
                        value, expected_type_description, nested_type
                    )
                    print(f"after covert js NESTED_CONVERSION_TYPE_LIST: {value}")
                else:
                    value = js_type_converter(value, expected_type_description)
                    print(f"after covert js NOT NESTED_CONVERSION_TYPE_LIST: {value}")

        elif language == "Python":  # 走这个
            expected_type_converted = PYTHON_TYPE_MAPPING[
                expected_type_description
            ]  # 从string -> str 而已

            if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                # ["array", "tuple"]， 有的param是nested
                # {"type": "array", "items": {"type": "string"
                # 比如"stops": {"type": "array", "items": {"type": "string"}, 这个应该是list(string)
                nested_type = param_details[param]["items"]["type"]
                nested_type_converted = PYTHON_TYPE_MAPPING[
                    nested_type
                ]  # string -> str也转换

        if expected_type_description == "tuple":
            try:
                value = list(value)
            except ValueError:
                print(
                    f"param: {param}, the expected type for its value is tuple, but model outputed value is: {value}, cannot be converted using list()"
                )
                result["valid"] = False
                result["error"].append(
                    f"Param is: {repr(param)}, the expected type for its value is tuple, but the model outputed value is: {repr(value)}, cannot be converted using list()"
                )
                result["error_type"] = (
                    "simple_function_checker:model_output_value_cannot_convert_to_list"
                )
                return result

        if expected_type_description == "float":
            try:
                # print(f"before float: {value}")
                value = float(value)
                # print(f"after float: {value}")
            except ValueError:
                print(
                    f"param: {param}, the expected type for its value is float, but model outputed value is: {value}, cannot be converted using float()"
                )
                result["valid"] = False
                result["error"].append(
                    f"Param is: {repr(param)}, the expected type for its value is float, but the model outputed value is: {repr(value)}, cannot be converted using float()"
                )  #
                result["error_type"] = (
                    "simple_function_checker:model_output_value_cannot_convert_to_float"
                )
                return result

        # Type checking
        # In fact, we only check for Python here.
        # Type check for other languages are handled by the type converter, and so their value (after conversion) is always correct.
        # 此处仅检查普通type，或者list（list）之类
        type_check_result = type_checker(
            param,  # model产生的每个param, conditions
            value,  # model 产生的每个param 的值 ， 要么 简单的type 如int, float, str, bool, 要么是nested的type，如list, tuple
            # [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]
            possible_answer_item[param],  # ["单双曲铝单板价格"] or [[1, 2, 3]]
            # [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]
            # 这个是一个list，里面是当前param所有可能的值， 比如 [10] , ['yes', ""]
            expected_type_description,  # array
            expected_type_converted,  #  list
            nested_type_converted,  # dict
        )
        is_variable = type_check_result["is_variable"]  # bool
        if not type_check_result[
            "valid"
        ]:  # 某一个param 不满足了，那全都是false了 , 包括嵌套list不满足的情况
            return type_check_result

        # 剩余list(dict) 没检查
        # It doesn't make sense to special handle dictionaries and list of dictionaries if the value is a variable.
        # We can just treat the variable as a string and use the normal flow.
        if not is_variable:
            # Special handle for dictionaries
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer_item[param])
                if not result["valid"]:
                    return result
                continue

            # Special handle for list of dictionaries
            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer_item[param])
                # param: conditions
                # value: [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]
                # possible_answer_item[param]: [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]
                if not result["valid"]:
                    return result
                continue

            # Special handle for strings
            elif expected_type_converted == str:
                # We don't check for case sensitivity for string, as long as it's not a variable
                result = string_checker(param, value, possible_answer_item[param])
                if not result["valid"]:
                    return result
                continue

            # elif expected_type_converted == list:
            #     result = list_checker(param, value, possible_answer_item[param])
            #     if not result["valid"]:
            #         return result
            #     continue

        # Check if the value is within the possible answers
        # if value not in possible_answer_item[param]:
        #     result["valid"] = False
        #     result["error"].append(
        #         f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer_item[param]}."
        #     )
        #     result["error_type"] = "value_error:others"
        #     return result

    # Check for optional parameters not provided but allowed
    for param in possible_answer_item:  # {"query": ["单双曲铝单板价格"]}
        # 遍历possible_answer_item的所有key , 此处已经转换了：{"query": "单双曲铝单板价格"}
        if param not in model_params and "" not in possible_answer_item[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_enforce_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: dict,
    language: str,
    model_name: str,
):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_enforce_order:wrong_count",
        }

    func_name_list = list(possible_answers.keys())
    possible_answers_list = []

    for key, value in possible_answers.items():
        possible_answers_list.append({key: value})

    for i in range(len(possible_answers_list)):
        func_description = find_description(func_descriptions, func_name_list[i])
        if func_description is None:
            return {
                "valid": False,
                "error": [
                    f"Function doc description not found for function name: {repr(func_name_list[i])}."
                ],
                "error_type": "parallel_function_checker_enforce_order:cannot_find_description",
            }
        result = simple_function_checker(
            func_description,
            model_output[i],
            possible_answers_list[i],
            language,
            model_name,
        )
        if not result["valid"]:
            return result

    return {"valid": True, "error": []}


def parallel_function_checker_no_order(
    func_description_item: list,
    model_result_item: list,
    possible_answer_item: dict,  # {"ExpertSearch": {"query": ["单双曲铝单板价格"]}}；； 他想要的是 {"geometry.area_circle":{"radius":[10],"units":["","meters"]}} ，咱们不是
    language: str,
    model_name: str,
):
    """for multiple
    func_description_item,  # 即list of tool pool，即使上面检查了是内置函数的possible
    但这里tool pool 中还会带着，因为是pool 中的函数，但无所谓：
    [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........


    model_result_item,  已经变成了 list of dict，这里不可能有内置函数，所以不会报错：
    [{"ExpertSearch": {"query": "单双曲铝单板价格"}}]

    possible_answer_item, 已经转换过了
    {"ExpertSearch": {"query": ["单双曲铝单板价格"]}}
    """

    # 我们这里先转换一下
    # possible_answer_item = {
    #     possible_answer_item["name"]: possible_answer_item["arguments"]
    # }

    if len(model_result_item) != len(possible_answer_item):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    func_name_list = list(possible_answer_item.keys())  # ["ExpertSearch"]
    possible_answer_item_list = []

    for key, value in possible_answer_item.items():
        possible_answer_item_list.append(
            {key: value}
        )  # [{"ExpertSearch": {"query": ["单双曲铝单板价格"]}}]

    matched_indices = []

    # We go throught the possible answers one by one, and eliminate the model output that matches the possible answer
    # It must be this way because we need ground truth to fetch the correct function description

    for i in range(len(possible_answer_item_list)):
        # [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........
        func_description = find_description(
            func_description_item, func_name_list[i]
        )  # 返回的还是dict，是单个函数的description
        # 返回 {"name": "ExpertSearch", "description": "私......... ,"parameters": {"type": "object", "properties": {"query": {"type": ..}

        # This should not happen. As possible_answer_item is the ground truth, and it should have the correct function name.
        if func_description is None:
            return {
                "valid": False,
                "error": [
                    f"Function doc description not found for function name: {repr(func_name_list[i])}."
                ],
                "error_type": "parallel_function_checker_no_order:cannot_find_description",
            }

        all_errors = []

        for index in range(len(model_result_item)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description,
                model_result_item[
                    index
                ],  # {"ExpertSearch": {"query": "单双曲铝单板价格"}}
                possible_answer_item_list[
                    i
                ],  # {"ExpertSearch": {"query": ["单双曲铝单板价格"]}}
                language,
                model_name,
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_result_item": model_result_item[index],
                            "possible_answer_item": possible_answer_item_list[i],
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [
                i for i in range(len(model_result_item)) if i not in matched_indices
            ]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def patten_matcher(exec_output, expected_result, function_call, is_sanity_check):
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}

    if type(exec_output) != type(expected_result):
        return {
            "valid": False,
            "error": [
                f"Wrong execution result type for {repr(function_call)}. Expected type: {type(expected_result)}, but got: {type(exec_output)}."
            ],
            "error_type": "executable_checker:wrong_result_type",
            "model_executed_output": exec_output,
        }
    if type(exec_output) == dict:
        # We loose the requirement for the sanity check as the expected result used in the sanity check might not be the most up-to-date one.
        # This happens when the key is a timestamp or a random number.
        if is_sanity_check:
            if len(exec_output) != len(expected_result):
                return {
                    "valid": False,
                    "error": [
                        f"Wrong execution result pattern for {repr(function_call)}. Expect type Dict, but wrong number of elements in the output. Expected length: {len(expected_result)}, but got: {len(exec_output)}."
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_length",
                    "model_executed_output": exec_output,
                }
            else:
                return result

        for key, value in expected_result.items():
            if key not in exec_output:
                return {
                    "valid": False,
                    "error": [
                        f"Wrong execution result pattern for {repr(function_call)}. Expect type Dict, but key {repr(key)} not found in the model output."
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_key_not_found",
                    "model_executed_output": exec_output,
                }
        for key, value in exec_output.items():
            if key not in expected_result:
                return {
                    "valid": False,
                    "error": [
                        f"Wrong execution result pattern for {repr(function_call)}. Expect type Dict, but key {repr(key)} not expected in the model output."
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_extra_key",
                    "model_executed_output": exec_output,
                }
    if type(exec_output) == list:
        if len(exec_output) != len(expected_result):
            return {
                "valid": False,
                "error": [
                    f"Wrong execution result pattern for {repr(function_call)}. Expect type list, but wrong number of elements in the output. Expected length: {len(expected_result)}, but got: {len(exec_output)}."
                ],
                "error_type": "executable_checker:wrong_result_type:list_length",
                "model_executed_output": exec_output,
            }
    return result


#### Helper functions for Exec ####
def executable_checker_simple(
    function_call: str,
    expected_result,
    expected_result_type: str,
    is_sanity_check=False,
):
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}

    exec_dict = {}

    try:
        exec(
            "from executable_python_function import *" + "\nresult=" + function_call,
            exec_dict,
        )
        exec_output = exec_dict["result"]
    except NoAPIKeyError as e:
        raise e
    except Exception as e:
        print(
            f"Error in executable_checker_simple: {repr(function_call)}. Error: {str(e)}"
        )
        result["valid"] = False
        result["error"].append(
            f"Error in execution: {repr(function_call)}. Error: {str(e)}"
        )
        result["error_type"] = "executable_checker:execution_error"
        return result

    # We need to special handle the case where the execution result is a tuple and convert it to a list
    # Because when json is stored, the tuple is converted to a list, and so the expected result is a list when loaded from json
    if isinstance(exec_output, tuple):
        exec_output = list(exec_output)

    if expected_result_type == "exact_match":
        if exec_output != expected_result:
            result["valid"] = False
            result["error"].append(
                f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}."
            )
            result["error_type"] = "executable_checker:wrong_result"
            result["model_executed_output"] = exec_output
            return result

    elif expected_result_type == "real_time_match":
        # Allow for 5% difference
        if (type(expected_result) == float or type(expected_result) == int) and (
            type(exec_output) == float or type(exec_output) == int
        ):
            if not (
                expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                <= exec_output
                <= expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            ):
                result["valid"] = False
                result["error"].append(
                    f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}. {REAL_TIME_MATCH_ALLOWED_DIFFERENCE * 100}% difference allowed."
                )
                result["error_type"] = "executable_checker:wrong_result_real_time"
                result["model_executed_output"] = exec_output
                return result
        else:
            result["valid"] = False
            result["error"].append(
                f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}. Type needs to be float or int for real time match criteria."
            )
            result["error_type"] = "executable_checker:wrong_result_real_time"
            result["model_executed_output"] = exec_output
            return result

    else:
        # structural match
        pattern_match_result = patten_matcher(
            exec_output, expected_result, function_call, is_sanity_check
        )
        if not pattern_match_result["valid"]:
            return pattern_match_result

    return result


def executable_checker_parallel_no_order(
    decoded_result: list, expected_exec_result: list, expected_exec_result_type: list
):

    if len(decoded_result) != len(expected_exec_result):
        return {
            "valid": False,
            "error": [
                f"Wrong number of functions provided. Expected {len(expected_exec_result)}, but got {len(decoded_result)}."
            ],
            "error_type": "value_error:exec_result_count",
        }

    matched_indices = []
    for i in range(len(expected_exec_result)):
        all_errors = []
        for index in range(len(decoded_result)):
            if index in matched_indices:
                continue

            result = executable_checker_simple(
                decoded_result[index],
                expected_exec_result[i],
                expected_exec_result_type[i],
                False,
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_executed_output": (
                                result["model_executed_output"]
                                if "model_executed_output" in result
                                else None
                            ),
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [
                i for i in range(len(decoded_result)) if i not in matched_indices
            ]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "executable_checker:cannot_find_match",
            }

    return {"valid": True, "error": [], "error_type": "executable_checker:unclear"}


#### Main function ####
def executable_checker_rest(func_call, idx):
    if "https://geocode.maps.co" in func_call:
        time.sleep(2)
    if "requests_get" in func_call:
        func_call = func_call.replace("requests_get", "requests.get")
    try:
        response = eval(func_call)
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Execution failed. {str(e)}"],
            "error_type": "executable_checker_rest:execution_error",
        }

    try:
        if response.status_code == 200:

            eval_GT_json = json.loads(EVAL_GROUND_TRUTH[idx])
            try:
                if isinstance(eval_GT_json, dict):
                    if isinstance(response.json(), dict):
                        if set(eval_GT_json.keys()) == set(response.json().keys()):
                            return {"valid": True, "error": [], "error_type": ""}
                        return {
                            "valid": False,
                            "error": ["Key inconsistency"],
                            "error_type": "executable_checker_rest:wrong_key",
                        }
                    return {
                        "valid": False,
                        "error": [
                            f"Expected dictionary, but got {type(response.json())}"
                        ],
                        "error_type": "executable_checker_rest:wrong_type",
                    }

                elif isinstance(eval_GT_json, list):
                    if isinstance(response.json(), list):
                        if len(eval_GT_json) != len(response.json()):
                            return {
                                "valid": False,
                                "error": [f"Response list length inconsistency."],
                                "error_type": "value_error:exec_result_rest_count",
                            }

                        else:
                            for i in range(len(eval_GT_json)):
                                if set(eval_GT_json[i].keys()) != set(
                                    response.json()[i].keys()
                                ):
                                    return {
                                        "valid": False,
                                        "error": [f"Key inconsistency"],
                                        "error_type": "executable_checker_rest:wrong_key",
                                    }

                            return {"valid": True, "error": []}
                    else:
                        return {
                            "valid": False,
                            "error": [
                                f"Expected list, but got {type(response.json())}"
                            ],
                            "error_type": "executable_checker_rest:wrong_type",
                        }
                return {
                    "valid": False,
                    "error": [
                        f"Expected dict or list, but got {type(response.json())}"
                    ],
                    "error_type": "executable_checker_rest:wrong_type",
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": [
                        f"Error in execution and type checking. Status code: {response.status_code}. Error: {str(e)}"
                    ],
                    "error_type": "executable_checker_rest:response_format_error",
                }
        else:
            return {
                "valid": False,
                "error": [
                    f"Execution result status code is not 200, got {response.status_code}"
                ],
                "error_type": "executable_checker_rest:wrong_status_code",
            }
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Cannot get status code of the response. Error: {str(e)}"],
            "error_type": "executable_checker_rest:cannot_get_status_code",
        }


def ast_checker(
    func_description_item,
    model_result_item,
    possible_answer_item,
    language,
    test_category,
    model_name,
):
    """
    func_description_item,  # 即list of tool pool，
    但这里tool pool 中还可能带着内置函数，因为是pool 中的函数，但无所谓：
    [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........

    model_result_item,  已经变成了 list of dict，这里不可能有内置函数，所以不会报错：
    [{"ExpertSearch": {"query": "单双曲铝单板价格"}}]

    possible_answer_item,
    {"ExpertSearch": {"query": ["单双曲铝单板价格"]}}
    """
    if "multiple" in test_category or "parallel" in test_category:
        # multiple functions + parallel functions
        # Some formatting issues that needs to be handled
        if test_category == "parallel_function":
            func_description_item = [func_description_item]

        return parallel_function_checker_no_order(
            func_description_item,
            model_result_item,
            possible_answer_item,
            language,
            model_name,
        )
    else:  # simple function
        if len(model_result_item) != 1:  # 就是 上面拿出来的model_result_item
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }

        model_result_item = model_result_item[
            0
        ]  # dict  -{"ExpertSearch": {"query": "单双曲铝单板价格"}}
        return simple_function_checker(
            func_description_item,
            model_result_item,
            possible_answer_item,
            language,
            model_name,
        )


def exec_checker(decoded_result: list, func_description: dict, test_category: str):
    if "multiple" in test_category or "parallel" in test_category:
        return executable_checker_parallel_no_order(
            decoded_result,
            func_description["execution_result"],
            func_description["execution_result_type"],
        )

    else:
        if len(decoded_result) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_exec_checker:wrong_count",
            }
        return executable_checker_simple(
            decoded_result[0],
            func_description["execution_result"][0],
            func_description["execution_result_type"][0],
            False,
        )
