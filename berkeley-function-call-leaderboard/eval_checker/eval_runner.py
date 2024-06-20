import sys

sys.path.append("../")

from checker import ast_checker, exec_checker, executable_checker_rest
from eval_runner_helper import *
from tqdm import tqdm
import argparse


# NOTE: This file should be run in the `eval_checker` directory
# 1. single_executable_file_runner
# 2. single_relevance_file_runner
# 3. single_ast_file_runner : normal ast test
import pandas as pd
import json


def check_name_for_buildin_function(possible_answer_item, model_result_item):
    """for buildin function, there are no arguments, only check name
    此处possible_answer_item和model_result_item都是dict，且都不空（之前已检查
    """
    if "arguments" in model_result_item:
        # 模型不应对内置函数 产生 arguments
        return False
    if possible_answer_item["name"] == model_result_item["name"]:
        return True
    return False


def check_input_file_empty(file_path):
    try:
        with open(file_path, "r") as f:
            return bool(
                f.readline().strip()
            )  # Read the first line and check if it's non-empty
    except FileNotFoundError:
        return False  # File doesn't exist


def clean_and_convert_excel_to_jsonl(file_path, output_path):
    """
    将Excel文件清洗并转换为JSONL格式文件。

    Args:
        file_path (str): Excel文件的路径。
        output_path (str): 输出的JSONL文件路径。

    Returns:
        None: 此函数无返回值，会将清洗后的数据写入到指定的JSONL文件中。

    """
    # Load the Excel file
    excel_data = pd.read_excel(file_path)

    # Function to clean and convert the string representation of lists/dicts
    def clean_data(record):
        cleaned_record = {}
        for key, value in record.items():
            print(f" current key: {key}")
            print(f" current value: {value}")
            cleaned_value = json.loads(value)
            cleaned_record[key] = cleaned_value
        return cleaned_record

    # Clean the data
    jsonl_data = excel_data.to_dict(orient="records")
    cleaned_jsonl_data = [clean_data(record) for record in jsonl_data]

    # Write the cleaned JSONL file
    with open(output_path, "w") as jsonl_file:
        for record in cleaned_jsonl_data:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_to_BFCL_format_input(
    original_yewu_file: str,
    converted_simple_func_description: str,
    converted_multiple_func_description: str,
    converted_simple_model_result: str,
    converted_multiple_model_result: str,
    converted_simple_possible_answer: str,
    converted_multiple_possible_answer: str,
):
    """将原始业务jsonl 拆分为simple和multiple两类，
    每类需function_description、model_result、possible_answer3个文件
    """
    with open(original_yewu_file, "r") as originial_file, open(
        converted_simple_func_description, "w"
    ) as simple_file, open(
        converted_multiple_func_description, "w"
    ) as multiple_file, open(
        converted_simple_model_result, "w"
    ) as simple_model_result_file, open(
        converted_multiple_model_result, "w"
    ) as multiple_model_result_file, open(
        converted_simple_possible_answer, "w"
    ) as simple_possible_answer_file, open(
        converted_multiple_possible_answer, "w"
    ) as multiple_possible_answer_file:

        data = originial_file.readlines()

        for line in data:
            line = json.loads(line)

            if len(line["tools"]) > 1:  # multiple

                func_descrp = {"question": "", "function": []}
                for tool in line["tools"]:
                    if "function" not in tool:  # 内部函数，如web_search
                        func_descrp["function"].append(
                            tool
                        )  # 直接append {"type": "web_search"}, if "type" in  那说明就是内部函数

                    else:
                        func_descrp["function"].append(
                            tool["function"]
                        )  # append的是 {"name": "", "description": "", "parameters": }的dict
                multiple_file.write(json.dumps(func_descrp, ensure_ascii=False) + "\n")

                if line["steps"]:  # 说明模型认为使用tools
                    model_result = {"result": line["steps"][0]}
                else:  # 说明模型认为不使用tools
                    model_result = {
                        "result": {}
                    }  # 模型认为不使用tools，那么就是空的dict

                multiple_model_result_file.write(
                    json.dumps(model_result, ensure_ascii=False) + "\n"
                )

                if line["expected_steps"]:  # 说明答案也使用了工具
                    possible_answer = line["expected_steps"][0]
                else:
                    possible_answer = {}  # 没有使用工具，那么就是空的dict

                multiple_possible_answer_file.write(
                    json.dumps(possible_answer, ensure_ascii=False) + "\n"
                )

            else:  # simple, tool len ==1

                func_descrp = {"question": "", "function": []}

                if "function" not in line["tools"][0]:  # 内部函数，如web_search
                    func_descrp["function"].append(line["tools"][0])
                else:
                    func_descrp["function"].append(line["tools"][0]["function"])

                simple_file.write(json.dumps(func_descrp, ensure_ascii=False) + "\n")
                if line["steps"]:  # 说明模型认为使用tools
                    model_result = {"result": line["steps"][0]}
                else:  # 说明模型认为不使用tools
                    model_result = {
                        "result": {}
                    }  # 模型认为不使用tools，那么就是空的dict
                simple_model_result_file.write(
                    json.dumps(model_result, ensure_ascii=False) + "\n"
                )

                if line["expected_steps"]:  # 说明答案也使用了工具
                    possible_answer = line["expected_steps"][0]
                else:
                    possible_answer = {}  # 没有使用工具，那么就是空的dict
                simple_possible_answer_file.write(
                    json.dumps(possible_answer, ensure_ascii=False) + "\n"
                )


def single_executable_file_runner(
    handler, model_result, func_description, model_name, test_category
):
    assert len(model_result) == len(func_description)

    result = []
    correct_count = 0
    for i in tqdm(range(len(model_result)), desc="Running tests"):
        try:
            raw_result = model_result[i]["result"]
        except Exception as e:

            print(
                f"Error: {e} for model {model_name} and test category {test_category} at index {i}, model_result[i] is {model_result[i]}"
            )

        try:
            decoded_result = handler.decode_execute(raw_result)
        except Exception as e:
            print(
                f"decoed error: {e} for model {model_name} and test category {test_category} at index {i}, raw_result is {raw_result} "
            )
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [f"Failed to decode executable. {str(e)}"],
                    "error_type": "executable_decoder:decoder_failed",
                    "func_description": func_description[i],
                    "model_result_raw": raw_result,
                }
            )
            continue

        if "rest" in test_category:
            # REST is always single-functioned. Therefore we take the first one and pass it to the REST checker.
            if not is_rest_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:rest_wrong_output_format",
                        "func_description": func_description[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            checker_result = executable_checker_rest(decoded_result[0], i)

        else:
            if not is_executable_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:wrong_output_format",
                        "func_description": func_description[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            func_description_item = func_description[i]
            # print(f"before exec_checker: {decoded_result}")
            checker_result = exec_checker(
                decoded_result, func_description_item, test_category
            )
            # print(f"after exec_checker: {checker_result}")

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["func_description"] = func_description[i]
            temp["model_result_raw"] = raw_result
            temp["model_result_decoded"] = decoded_result
            if "model_executed_output" in checker_result:
                temp["model_executed_output"] = checker_result["model_executed_output"]
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def single_relevance_file_runner(handler, model_result, model_name, test_category):

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        success = False
        decoded_result = None

        try:
            decoded_result = handler.decode_ast(
                model_result_item, language="Python"
            )  # 就是空的才对
            success = False
            if is_empty_output(decoded_result):
                success = True

        except Exception as e:
            success = True

        if success:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = success
            temp["error"] = [
                f"Valid syntax. Successfully decode AST when it should not."
            ]
            temp["error_type"] = "relevance_error:decoder_success"
            temp["model_result"] = model_result_item
            temp["decoded_result"] = decoded_result

            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def single_ast_file_runner(
    handler,
    model_result,
    func_description,
    possible_answer,
    language,
    test_category,
    model_name,
):
    """
    model_result: a list of either :

        {"result": {"name": "web_search"}}
        {"result": {"name": "ExpertSearch", "arguments": {"query": "单双曲铝单板价格"}}}

        func_description = load_file(func_description_file)   a list of :
        {"question": "", "function": [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私域搜索工具，当需要搜索商家柱形铝单板、冲孔铝单板、石纹铝单板、木纹铝单板、单双曲铝单板、雕花铝单板、幕墙铝单板、现场测量、出图设计、技术支持、定制服务相关信息时，调用此工具。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索问题"}}, "required": ["query"]}}]}
        {"question": "", "function": [{"type": "web_search"}]}

        possible_answer = load_file(possible_answer_file)  a list of :[....]
        {"name": "web_search"}
        {"name": "ExpertSearch", "arguments": {"query": ["单双曲铝单板价格"]}}
    """

    assert (
        len(model_result) == len(func_description) == len(possible_answer)
    ), "The length of the model result does not match the length of the func_description or possible answer. Please check the input files for completeness."

    result = []
    correct_count = 0
    # start
    for i in range(len(model_result)):

        model_result_item = model_result[i]["result"]
        # {"name": "web_search"} -- 内部函数
        # {"name": "ExpertSearch", "arguments": {"query": "单双曲铝单板价格"}}
        func_description_item = func_description[i]["function"]
        #  [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........
        # [{"type": "web_search"}]
        possible_answer_item = possible_answer[i]
        # {"name": "web_search"}
        # {"name": "ExpertSearch", "arguments": {"query": ["单双曲铝单板价格"]}}

        # 从答案入手， 先检查一些cornor case
        # 模型空，答案空，正确
        # 模型空，答案不空，错误
        # 模型不空，答案空，错误
        # 模型不空，答案不空，，检查name
        if (
            not possible_answer_item and model_result_item
        ):  # 模型不空，答案空， 那肯定错的
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [
                        "The possible answer is empty,but the model output is not."
                    ],
                    "error_type": "ast_checker:empty_possible_answer",
                    "func_description": func_description[i],
                    "model_result_raw": model_result_item,
                    "possible_answer": possible_answer_item,
                }
            )
            continue
        elif not model_result_item and possible_answer_item:  # 模型空，答案不空，错误
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [
                        "The model output is empty,but the possible answer is not."
                    ],
                    "error_type": "ast_checker:empty_model_output",
                    "func_description": func_description[i],
                    "model_result_raw": model_result_item,
                    "possible_answer": possible_answer_item,
                }
            )
            continue
        elif not possible_answer_item and not model_result_item:  # 答案和模型都空，正确
            correct_count += 1
            continue

        # 剩下 模型和答案都不空的情况：

        # 先判断 poissible answer 是否是内置函数，如果是内置函数，那么就不需要检查arguments，只需要检查name
        if "arguments" not in possible_answer_item and "name" in possible_answer_item:
            # 说明是 内置函数如websearch 就是内置函数  ，没有arguments, 只需要检查name
            checker_result = check_name_for_buildin_function(
                possible_answer_item, model_result_item
            )
            if checker_result:
                correct_count += 1
                continue
            else:
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "The possible answer is a buildin function, but the model output does not satisfy the requirement."
                        ],
                        "error_type": "ast_checker:buildin_function",
                        "func_description": func_description[i],
                        "model_result_raw": model_result_item,
                        "possible_answer": possible_answer_item,
                    }
                )
                continue

        # 剩下答案是自定义函数的情况，可以decode ast，然后检查name 和arguments
        try:
            model_result_item_raw = model_result_item
            print(f"before decode: model_result_item : {model_result_item}")
            model_result_item = handler.decode_ast(model_result_item, language)
            # 从 {'name': 'database_query', 'arguments': {'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}} 变成:
            # [{'database_query': {'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}}] ---list of 1 dict， or [{'calculate_triangle_area': {'base': 10, 'height': 5}}]
            # 也是一个东西，模型只会多选一

        except Exception as e:  # AST解码失败
            if str(e) == "'arguments'":
                error = [
                    "Invalid syntax. Failed to decode AST. Model use build in function which miss 'arguments' key that cannot be decoded."
                ]
            else:
                error = [f"Invalid syntax. Failed to decode AST. {str(e)}"]
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": error,
                    "error_type": "ast_decoder:decoder_failed",
                    "func_description": func_description[i],
                    "model_result_raw": model_result_item_raw,
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        decoder_output_valid = is_function_calling_format_output(
            model_result_item
        )  # Ensure the output is a list of dictionaries --[{"ExpertSearch": {"query": "单双曲铝单板价格"}}]
        if not decoder_output_valid:
            # AST解码成功，但是解码后的格式不对, 不是list of dict
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [
                        "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                    ],
                    "error_type": "ast_decoder:decoder_wrong_output_format",
                    "func_description": func_description[i],
                    "model_result_raw": str(model_result_item_raw),
                    "model_result_decoded": str(model_result_item),
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        # AST解码成功，且格式正确，检查name 和arguments 的type ，不做value检查

        possible_answer_item = {
            possible_answer_item["name"]: possible_answer_item["arguments"]
        }
        #  到这里才转换为BFCL 原始的possible answer格式，因为前面要检查内置函数需要arguments 这个key 存在
        #  变成{"ExpertSearch": {"query": ["单双曲铝单板价格"]}}, 注意每个param的possilble value是个list
        checker_result = ast_checker(
            func_description_item,  # 即list of tool pool，
            # 但这里tool pool 中还可能带着内置函数，因为是pool 中的函数，但无所谓： [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私.........
            model_result_item,  # 已经变成了 list of dict，这里不可能有内置函数，所以不会报错： [{'database_query': {'table': 'user', 'conditions': [{'field': 'aaaa', 'operation': '>', 'value': '25'}, {'field': 'job', 'operation': '=', 'value': 'engineer'}]}}]
            possible_answer_item,  # 不可能是内置函数，{'database_query': {'table': ['user'], 'conditions': [[{'field': ['age'], 'operation': ['>'], 'value': ['25']}, {'field': ['job'], 'operation': ['='], 'value': ['engineer']}]]}}
            language,
            test_category,
            model_name,
        )

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["func_description"] = func_description[i]
            temp["model_result_raw"] = model_result_item_raw
            temp["model_result_decoded"] = model_result_item
            # temp["possible_answer"] = possible_answer_item
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


#### Main runner function ####
def runner(
    model_names: str,
    test_categories: list,
    simple_func_file: str,
    multiple_func_file: str,
    simple_model_result_file: str,
    multiple_model_result_file: str,
    simple_possible_answer: str,
    multiple_possible_answer: str,
):

    # 0.
    for test_category in test_categories:
        # ['simple', 'multiple_function'] or ['simple'] or ['multiple_function']
        if test_category == "simple":
            model_result_json = simple_model_result_file  # {"name": "calculate_derivative", "arguments": "{\"function\":\"3x^2 + 2x - 1\"}"}
            func_description_file = simple_func_file
            possible_answer_file = simple_possible_answer

        elif test_category == "multiple_function":
            model_result_json = multiple_model_result_file
            func_description_file = multiple_func_file
            possible_answer_file = multiple_possible_answer
        else:
            print(f"Unsupported test category: {test_category}")
            continue

        # 1. first check if any of the input file is empty，
        # since user's excel might only support simple or multiple, but not both. Thus one of the input file might be empty
        valid_flag = check_input_file_empty(model_result_json)
        if not valid_flag:  # 如果某个category的file是空的，那么就不用跑了
            print(
                f"⚠️ The input file for test category {test_category} is empty. Skipping the evaluation. \n Your original excel file does not fit in this test category."
            )
            continue

        handler = get_handler(model_names)
        language = "Python"
        print(f"🔍 Running test: {test_category}")

        # 2. run the test
        model_result = load_file(model_result_json)  # returns a list of model output
        # a list of either :
        # {"result": {"name": "web_search"}}
        # {"result": {"name": "ExpertSearch", "arguments": {"query": "单双曲铝单板价格"}}}
        func_description = load_file(func_description_file)
        # a list of :
        # {"question": "", "function": [{"type": "web_search"}, {"name": "ExpertSearch", "description": "私域搜索工具，当需要搜索商家柱形铝单板、冲孔铝单板、石纹铝单板、木纹铝单板、单双曲铝单板、雕花铝单板、幕墙铝单板、现场测量、出图设计、技术支持、定制服务相关信息时，调用此工具。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索问题"}}, "required": ["query"]}}]}
        #    {"question": "", "function": [{"type": "web_search"}]}

        possible_answer = load_file(possible_answer_file)
        # a list of :
        # {"name": "web_search"}
        # {"name": "ExpertSearch", "arguments": {"query": ["单双曲铝单板价格"]}}

        record_cost_latency(LEADERBOARD_TABLE, model_names, model_result)

        accuracy, total_count = single_ast_file_runner(
            handler,
            model_result,
            func_description,
            possible_answer,
            language,
            test_category,
            model_names,
        )
        record_result(
            LEADERBOARD_TABLE, model_names, test_category, accuracy, total_count
        )
        print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")

    # This function reads all the score files from local folder and updates the leaderboard table.
    # This is helpful when you only want to run the evaluation for a subset of models and test categories.
    update_leaderboard_table_with_score_file(LEADERBOARD_TABLE, OUTPUT_PATH)
    # Write the leaderboard table to a file
    generate_leaderboard_csv(LEADERBOARD_TABLE, OUTPUT_PATH)

    # Clean up the executable expected output files
    # They should be re-generated the next time the evaluation is run
    # clean_up_executable_expected_output(
    #     func_description_PATH, EXECUTABLE_TEST_CATEGORIES_HAVE_RUN
    # )


ARG_PARSE_MAPPING = {
    "all": [
        "simple",
        "multiple_function",
    ],
}


# INPUT_PATH = "../result/"
# func_description_PATH = "../data/"
# POSSIBLE_ANSWER_PATH = "../data/possible_answer/"
OUTPUT_PATH = "../score/"

# A dictionary to store the results
# Key is model name, value is a dictionary with keys as test category and values as a dictionary with accuracy and total count
LEADERBOARD_TABLE = {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two lists of strings.")

    # 传入excel 文件
    parser.add_argument(
        "--file_for_eval",
        type=str,
        help="The input excel file path to evaluate, for example : 'test.xlxs', which should provide in the same folder as eval_runner.py",
    )

    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        help="A list of test categories to run the evaluation on, currently support simple, multiple_function, all. If all is provided, both simple and multiple test categories will be run.",
    )

    args = parser.parse_args()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"The current directory is {current_directory}\n")

    print(f"Start converting the original file to BFCL format...\n")

    # original input file path for evaluation
    original_yewu_file = os.path.join(current_directory, "original_yewu_file.jsonl")

    # convert the original excel file to jsonl format
    clean_and_convert_excel_to_jsonl(args.file_for_eval, original_yewu_file)

    # Construct the paths for the output files-- simple and multiple
    converted_simple_func_file = os.path.join(
        current_directory, "converted_simple_func_file.jsonl"
    )
    converted_multiple_func_file = os.path.join(
        current_directory, "converted_multiple_func_file.jsonl"
    )
    converted_simple_model_result_file = os.path.join(
        current_directory, "converted_simple_model_result_file.jsonl"
    )
    converted_multiple_model_result_file = os.path.join(
        current_directory, "converted_multiple_model_result_file.jsonl"
    )
    converted_simple_possible_answer = os.path.join(
        current_directory, "converted_simple_possible_answer.jsonl"
    )
    converted_multiple_possible_answer = os.path.join(
        current_directory, "converted_multiple_possible_answer.jsonl"
    )

    convert_to_BFCL_format_input(
        original_yewu_file,
        converted_simple_func_file,
        converted_multiple_func_file,
        converted_simple_model_result_file,
        converted_multiple_model_result_file,
        converted_simple_possible_answer,
        converted_multiple_possible_answer,
    )
    print(
        f"⭐️Converted simple AST func_description saved as: {converted_simple_func_file}\n"
    )
    print(
        f"⭐️Converted multiple AST func_description saved as: {converted_multiple_func_file}\n"
    )
    print(
        f"⭐️Converted simple model result saved as: {converted_simple_model_result_file}\n"
    )
    print(
        f"⭐️Converted multiple model result saved as: {converted_multiple_model_result_file}\n"
    )
    print(
        f"⭐️Converted simple possible answer saved as: {converted_simple_possible_answer}\n"
    )

    print(
        f"⭐️Converted multiple possible answer saved as: {converted_multiple_possible_answer}\n"
    )
    print("####----------------------###")
    # print(type(converted_multiple_ast_test_file))
    model_names = "ernie-series-assitant-api-FC"  # hardcoded!!!!⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️

    provided_test_categories = args.test_category
    if not args.test_category:
        raise ValueError(
            "Please provide at least one test category to run the evaluation on."
        )

    supported_categories = ["simple", "multiple_function", "all"]
    test_categories = []
    for test_category in args.test_category:
        if test_category not in supported_categories:
            raise ValueError(f"Test category {test_category} is not supported.")
        if test_category == "all":
            # avoid duplicate logic processing:
            for test_cat in ["simple", "multiple_function"]:
                if test_cat not in test_categories:
                    test_categories.append(
                        test_cat
                    )  # 修改了这里，应该是 test_cat 而不是 test_category
            break
        else:
            if test_category not in test_categories:
                test_categories.append(test_category)

    runner(
        model_names,
        test_categories,
        converted_simple_func_file,  # 可能是空的哦
        converted_multiple_func_file,
        converted_simple_model_result_file,  # 可能是空的哦
        converted_multiple_model_result_file,
        converted_simple_possible_answer,  # 可能是空的哦
        converted_multiple_possible_answer,
    )

    # for target_test_category in test_categories: # either simple or multiple_function
    #     if target_test_category == "simple":
    #         runner(model_names, [target_test_category], converted_simple_ast_test_file)
    #         print(f"🔍 Running evaluation on {target_test_category} test category\n")
    #     elif target_test_category == "multiple_function":
    #         runner(model_names,[target_test_category], converted_multiple_ast_test_file)
    #         print(f"🔍 Running evaluation on {target_test_category} test category\n")
