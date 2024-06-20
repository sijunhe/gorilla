import json





# read a jsonl file, and return a jsonl file, 
# the single line for a json in that original jsonl file looks like :

# {
#     "steps":[
#        {
#           "name":"calculate_triangle_area",
#           "arguments":"{\"base\":10,\"height\":5,\"unit\":\"units\"}"
#        }
#     ],
#     "tools":[
#        {
#           "type":"function",
#           "function":
#                  {
#                     "name":"calculate_triangle_area",
#                     "description":"Calculate the area of a triangle given its base and height.",
#                     "parameters":{
#                         "type":"dict",
#                         "properties":{
#                         "base":{
#                             "type":"integer",
#                             "description":"The base of the triangle."
#                         },
#                         "height":{
#                             "type":"integer",
#                             "description":"The height of the triangle."
#                         },
#                         "unit":{
#                             "type":"string",
#                             "description":"The unit of measure (defaults to 'units' if not specified)"
#                         }
#                         },
#                         "required":[
#                         "base",
#                         "height"
#                         ]
#                     }
#                 }
#        }
#     ]
#  }

# read it and convert to the following format :
#  with "question" key always be empty, "function" key be the everything from tools key , if value of tools key is 1, then it's simple function, so we just extract the whole thing from function key, else, we need to process multiple functions 
# {"question": "Find the area of a triangle with a base of 10 units and height of 5 units.", "function": {"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "dict", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}}


def convert_yewu_jsonl_to_BFCL_jsonl(originial_yewu_file, coverted_simple_ast_test, coverted_multiple_ast_test):
    with open(originial_yewu_file, 'r') as originial_file, \
         open(coverted_simple_ast_test, 'w') as simple_file, \
         open(coverted_multiple_ast_test, 'w') as multiple_file:
        
        data = originial_file.readlines()

        for line in data:
            line = json.loads(line) 
            if len(line['tools']) > 1: # multiple
                this_line = {"question": "", "function": []}
                for tool in line["tools"]:
                    this_line["function"].append(tool["function"])
                multiple_file.write(json.dumps(this_line) + "\n")
            else: # simple
                this_line = {"question": "", "function": line["tools"][0]["function"]}
                simple_file.write(json.dumps(this_line) + "\n")



