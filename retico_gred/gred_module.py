import glob
import itertools
import os
import re
import sys
import time
from typing import Union

import random

from retico_vision import ObjectPermanenceIU

os.environ['CORE'] = 'retico-core'
sys.path.append(os.environ['CORE'])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import retico_core
from retico_core import abstract, UpdateType
# from retico_chatgpt.chatgpt import GPTTextIU
from retico_core.text import TextIU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bsu-slim/gred-cozmo"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

class GREDTextIU(TextIU):
    @staticmethod
    def type():
        return TextIU.type()
    def __repr__(self):
        # show the full payload without truncation
        return f"{self.type()} - ({self.creator.name()}): {self.get_text()}"


class GREDActionGenerator(abstract.AbstractModule):
    @staticmethod
    def name():
        return "GRED Action Generator"
    @staticmethod
    def description():
        return "Generate robot‐behavior sequences from emotion labels."
    @staticmethod
    def input_ius():
        # return GPTTextIU # TODO: I didn't want to deal with extra dependencies, but could easily make this support either
        return [ObjectPermanenceIU]
    @staticmethod
    def output_iu():
        return GREDTextIU

    def __init__(self, model, tokenizer, device, behaviours_to_pregenerate=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.current_text = ""
        self.behaviours_to_pregenerate = behaviours_to_pregenerate
        self.pregenerated_behaviours = {}


    def process_update(self, update_message):
        received_update = False
        for iu, update_type in update_message:
            if update_type not in [UpdateType.ADD, UpdateType.COMMIT]:
                continue
            if not isinstance(iu, ObjectPermanenceIU): # TODO: re-add chatgpt iu
                continue
            
            received_update = True
            # extract the emotion label from the IU payload
            if isinstance(iu, ObjectPermanenceIU): # Different behaviour for obj permanence IU than chatgpt IU
                num_objects_seen = len(iu.payload)
                self.current_text = 'interest_desire' if num_objects_seen > 0 else 'confusion_sorrow_boredom'
            else:
                self.current_text = iu.payload.strip()

        if received_update:
            print(f"Received emotion label: {self.current_text}")
            if self.current_text in dict(self.behaviours_to_pregenerate).keys():
                behavior = random.choice(self.pregenerated_behaviours[self.current_text])
            else:
                # run the model once per iteration
                start = time.time()
                behavior = self.predict(self.current_text)
                end = time.time()
                print(f"Behaviour prediction took {end - start} seconds")

            # IAC I can't have the head move, it messes with the obj depth function
            behavior_without_head_movement = []
            for val in behavior.split(" "):
                if val.startswith("set_head_angle"):
                    continue
                else:
                    behavior_without_head_movement.append(val)
            behavior = " ".join(behavior_without_head_movement)
            print(f"Generated behavior for {self.current_text}: {behavior}")

            payload = f"{behavior}"
            # prepare result update
            output_iu = self.create_iu(iu)
            output_iu.payload = payload
            output_iu = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            return output_iu

    def predict(self, emotion_label: str, num_predictions=1) -> Union[list[str], str]:
        prompt = f"<|startoftext|>Emotion: {emotion_label} <|endoftext|> Behaviors:"
        prompts = [prompt] * num_predictions
        inputs = tokenizer(prompts, return_tensors="pt").to(device)

        # Generate behaviors
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        if num_predictions == 1:
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text.split("Behaviors:")[1].strip()
        else:
            generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
            return [generated_text.split("Behaviors:")[1].strip() for generated_text in generated_texts]


    def change_words(self, predicted_strings, behaviour):
        """
        Some of the predicted words are not compatible with Cozmo.
        - "mm" is spoken as "millimeter"
        - "blarghhhh" becomes "b" "l" "a" "r" "g" "h" "h" "h" "h"
        - "zoo" and "yawn" sound too much like actual words instead of spoken words

        This replaces all "say_text_" strings with strings from the training data (modified to adjust "yawn" and "zoo" into "aawwhn" and "tsoo")
        which guarantees Cozmo won't accidentally say a word that sounds more intelligent than we are expecting.
        """
        possible_new_words = []
        training_data_directory_path = "../../retico-gred/cozmo_training_data/cozmo_saved_behaviors"
        if behaviour == 'interest_desire':
            # Create the search pattern
            file_name_substrings = ['understanding', 'joy', 'interest']
        elif behaviour == 'confusion_sorrow_boredom':
            file_name_substrings = ['boredom', 'confusion', 'frustration']

        search_patterns = [os.path.join(training_data_directory_path, f"*{substring}*") for substring in file_name_substrings]

        matching_files = [glob.glob(search_pattern) for search_pattern in search_patterns]
        matching_files = list(itertools.chain.from_iterable(matching_files))
        for filepath in matching_files:
            if os.path.isfile(filepath): # Ensure it's a file and not a directory
                with open(filepath, 'r') as f:
                    say_text_command = [re.sub("[\s+'()]", '', "_".join(sub_list.split(","))) for sub_list in f.read().split('\n') if 'say_text' in sub_list]
                    possible_new_words.extend(say_text_command)
        regex = re.compile(r'(say_text_([^ ]+)_([^ ]+)_([^ ]+))')
        updated_predicted_strings = [regex.sub(lambda match: random.choice(possible_new_words), predicted_string) for predicted_string in predicted_strings]
        return updated_predicted_strings


    def setup(self):
        if self.behaviours_to_pregenerate:
            for behavior, num_to_generate in self.behaviours_to_pregenerate:
                results = self.predict(behavior, num_to_generate)
                updated_predicted_strings = self.change_words(results, behavior)
                self.pregenerated_behaviours[behavior] = updated_predicted_strings