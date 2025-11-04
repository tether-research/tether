import pickle
import time
from pathlib import Path
import base64
import json
import re
import shelve
from PIL import Image
from io import BytesIO
import gevent
import threading
import copy
from contextlib import contextmanager
import concurrent.futures
import cv2
import hashlib

from google import genai


class VlmWrapper:
    def __init__(self, model, api_key, default_temperature=0.0, thinking_budget=None, use_cache=None):
        self.model = model
        self.api_key = api_key
        self.default_temperature = default_temperature
        self.thinking_budget = thinking_budget
        self.use_cache = use_cache

        self.init_model(model, api_key)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        if self.use_cache is not None:
            self.cache_lock = threading.Lock()
            self.cache_path = str(self.use_cache / "cache")
        
    def init_model(self, model, api_key):
        self.api_key = api_key
        org, name = model.split("/")
        self.org, self.name = org, name
        if org == "google":
            print(f"Configured Gemini model {name}")
        else:
            print(f"Organization {org} unknown, skipping model initialization")
    
    @contextmanager
    def locked_cache(self):
        with self.cache_lock:
            with shelve.open(self.cache_path) as cache:
                yield cache

    def close(self):
        self.executor.shutdown()

    def query(self, messages, retries=10, temperature=None, system_instruction=None, save_path=None, use_cache=True):        
        temperature = temperature if temperature is not None else self.default_temperature
        original_messages = copy.deepcopy(messages)

        response = None
        if self.use_cache is not None and use_cache:
            messages_hash = str(hash(pickle.dumps("\n".join([str(i) for i in original_messages]))))
            with self.locked_cache() as cache:
                if messages_hash in cache:
                    print("Retrieved response from cache")
                    response = cache[messages_hash]
        
        if response is None:
            response = self.run_query(messages, retries=retries, temperature=temperature, system_instruction=system_instruction)

        if self.use_cache is not None and use_cache:
            messages_hash = str(hash(pickle.dumps("\n".join([str(i) for i in original_messages]))))
            with self.locked_cache() as cache:
                cache[messages_hash] = response

        self.save_query(original_messages, response, save_path)
        return response

    def query_async(self, messages, retries=10, temperature=None, system_instruction=None, save_path=None, future_id=None):
        return self.executor.submit(lambda: (future_id, self.query(messages, retries, temperature, system_instruction, save_path)))
     
    def run_query(self, messages, retries=10, temperature=None, system_instruction=None):
        if self.org == "google":
            filepaths = []
            for i, message in enumerate(messages):
                if isinstance(message, Path):
                    filepaths.append(str(message))
                    messages[i] = len(filepaths) - 1

            messages = self.upload_files(messages, filepaths)
            messages = {"role": "user", "parts": messages}
        else:
            raise NotImplementedError
        
        def call_api(self):
            if self.org == "google":
                client = genai.Client(api_key=self.api_key)
                config = genai.types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=self.thinking_budget),
                )
                response = client.models.generate_content(
                    model=self.name,
                    contents=messages["parts"],
                    config=config
                )
            else:
                raise NotImplementedError
            return response

        response, exception = None, None
        for _ in range(retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_api, self)
                    response = future.result(timeout=90)
                if self.org == "google":
                    response = response.text
                break
            except concurrent.futures.TimeoutError:
                print("Timeout (1 min) while querying, retrying...")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(f"Exception while querying: {e}")
                exception = e
                time.sleep(1)
        if not response:
            print(f"Query failed {retries} times!")
            raise exception
        
        return response

    def upload_files(self, messages, filepaths):
        def upload_file(path):
            with open(path, "rb") as file:
                cache_key = hashlib.sha256(file.read()).hexdigest()[:16]

            model = genai.Client(api_key=self.api_key)
            
            if self.use_cache is not None:
                with self.locked_cache() as cache:
                    if cache_key in cache:
                        try:
                            file = model.files.get(name=cache[cache_key])
                            if file.state.name == "ACTIVE":
                                print(f"Retrieved file from cache ({path})")
                                return file
                        except:
                            print(f"Failed to retrieve cached file from Gemini API, probably using another API key ({path})")

            if not Path(path).exists():
                raise ValueError(f"File {path} does not exist, VLM upload failed!")
            file = model.files.upload(file=path)
            while file.state.name == "PROCESSING":
                time.sleep(1)
                file = model.files.get(name=file.name)
            if file.state.name == "FAILED":
                raise ValueError(file.state.name)
            
            if self.use_cache is not None:
                with self.locked_cache() as cache:
                    cache[cache_key] = file.name

            return file

        messages = [upload_file(filepaths[message]) if isinstance(message, int) else message for message in messages]
        return messages

    ##### Response Post Processing #####

    def parse_response_json(self, response, save_path=None):
        patterns = [
            r'```json(.*?)```',
            r'```(.*?)```',
            r'^(.*?)$',
        ]
        for pattern in patterns:
            string = re.findall(pattern, response, re.DOTALL)
            if string:
                parsed_response = string[-1].strip()
                parsed_response = parsed_response.replace("\n", "")
                parsed_response = parsed_response.replace('’', "'")
                try:
                    response = json.loads(parsed_response)
                    break
                except:
                    print(f"Failed to parse response!")
                    return None
        if save_path:
            with open(save_path, "w") as f:
                json.dump(response, f, indent=4)
        return response

    def format_query(self, messages, response):
        messages = [f"![{message.name}]({message})" if isinstance(message, Path) else message for message in messages]
        ret = "\n\n".join(messages)
        ret += "\n" + "*"*100 + "\n\n"
        ret += response
        return ret
    
    def save_query(self, messages, response, save_path):
        if save_path is None:
            return
        with open(save_path, "w") as f:
            f.write(self.format_query(messages, response))

