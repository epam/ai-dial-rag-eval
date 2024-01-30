import re
import json
import pandas as pd

from universal_api_utils import merge


def parse_log(log_file_name, /, combined_search_stage_regexp='Combined search'):
    with open(log_file_name, "r") as file:
        log_lines = []
        system_prompt = None

        line_index = 0
        for line_str in file:
            line_index = line_index + 1

            line = json.loads(line_str)

            request = line["request"]
            request_body = json.loads(request["body"])
            request["body"] = request_body
            stream = request_body.get("stream", False)

            response = line.get('response', '')
            response_body = None
            if stream:
                body = response["body"]
                chunks = body.split("\n\ndata: ")

                chunks = [chunk.strip() for chunk in chunks]

                chunks[0] = chunks[0][chunks[0].find("data: ") + 6:]
                if chunks[-1] == "[DONE]":
                    chunks.pop(len(chunks) - 1)

                response_body = json.loads(chunks[-1])
                for chunk in chunks[0: len(chunks) - 1]:
                    chunk = json.loads(chunk)
                    response_body["choices"] = merge(response_body["choices"], chunk["choices"])

                response_body_choices = response_body["choices"]
            else:
                response_body = json.loads(response["body"])
            response["body"] = response_body

            model_name = request_body.get("model")
            if model_name is None:
                re_match = re.match('/openai/deployments/(.*?)/chat/', request.get('uri', ''))
                if re_match is not None:
                    model_name = re_match.group(1)

            request_body_messages = request_body.get('messages', [])
            user_prompt = None
            for msg in request_body_messages:
                msg_role = msg.get('role', '').lower()
                msg_content = msg.get('content')
                if msg_role == '' or msg_content is None:
                    continue
                if msg_role == 'system':
                    if system_prompt is not None:
                        raise Exception(
                            f"Line {line_index}: Ambiguous system request prompt '{system_prompt}' vs '{msg_content}'")
                    system_prompt = pd.DataFrame({'log_line_number': line_index,
                                                  'request_time': request.get('time'),
                                                  'request_model': model_name,
                                                  'system_prompt': msg.get('content'),
                                                  }, index=[0])
                elif msg_role == 'user':
                    if user_prompt is None or user_prompt == '':
                        user_prompt = msg_content.strip()
                    elif msg_content is not None:
                        raise Exception(
                            f"Line {line_index}: Ambiguous user request prompt '{user_prompt}' vs '{msg_content}'")

            response_body_choices = response_body.get('choices', [])
            if len(response_body_choices) != 1:
                raise Exception(f"Line {line_index}: Response choices len(='{len(response_body_choices)}') != 1")
            response_body_choices_delta = response_body_choices[0].get('delta')
            if response_body_choices_delta is None:
                raise Exception(f"Line {line_index}: There is no 'delta' field in response")
            response_body_choices_delta_content = response_body_choices_delta.get('content')
            if response_body_choices_delta_content is None:
                raise Exception(f"Line {line_index}: There is no 'delta.content' field in response")

            rbcdcc = response_body_choices_delta.get('custom_content', {})
            rbcdccs = list(filter(
                lambda stage: re.search(combined_search_stage_regexp, stage.get('name', '')) is not None,
                rbcdcc.get('stages', [])))
            if len(rbcdccs) > 1:
                raise Exception(f"Line {line_index}: combined search stage: {rbcdccs}")
            rbcdccsa = None
            if len(rbcdccs) == 1:
                rbcdccsa = rbcdccs[0].get('attachments')

            log_line = pd.DataFrame({'log_line_number': line_index,
                                     'request_time': request.get('time'),
                                     'request_model': model_name,
                                     'request_message': user_prompt,
                                     'request_history': json.dumps(request_body_messages),

                                     'response_message': response_body_choices_delta_content,
                                     'response_search_attachments': json.dumps(rbcdccsa),
                                     'response_attachments': json.dumps(rbcdcc.get('attachments')),
                                     }, index=[0])
            log_lines.append(log_line)

        log_lines = pd.concat(log_lines)

        return log_lines, system_prompt


[example_log_data, example_system_prompt] = parse_log("logs/example.log")
print(example_log_data)

[example2_log_data, example2_system_prompt] = parse_log("logs/example2.log")
print(example2_log_data)
