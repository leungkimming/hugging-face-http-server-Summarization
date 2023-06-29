# Copyright (c) Microsoft. All rights reserved.
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from . import InferenceGenerator
# The model used to get the tokenizer can be a little arbitrary
# since the tokenizers are common within the same model type


class QnaGenerator(InferenceGenerator.InferenceGenerator):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def perform_inference(self, prompt, context, max_tokens):
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        model.to(self.device)

        inputs = prompt.split('|')
        text = inputs[0]
        question = inputs[1]
        print("1 {0}, {1}".format(question, text), flush=True)

        inputs = self.tokenizer(question, text, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
        input_ids  = inputs["input_ids"].tolist()[0]
        print("2 {0}".format(inputs), flush=True)


        outputs = model(**inputs)
        print("3 {0}".format(outputs), flush=True)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        print("4 {0}, {1}".format(answer_start_scores, answer_end_scores), flush=True)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        print(answer_start, flush=True)
        print(answer_end, flush=True)
        print("5 {0}, {1}".format(answer_start, answer_end), flush=True)

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        print("6 {0}".format(answer), flush=True)

        return (
            answer,
            0,
            0,
        )
###############################################
        # model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        # model.to(self.device)

        # encodings = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        # generated_ids = model.generate(encodings, min_length=80, max_length=120)
        # # summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        # return (
        #     self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        #     encodings.numel(),
        #     len(generated_ids[0]),
        # )
        
        # encodings = self.tokenizer.encode_plus(
        #     text=prompt, text_pair=context, truncation=True, return_tensors="pt"
        # )

        # generated_ids = model.generate(
        #     encodings.input_ids,
        #     max_length=1020,
        #     # num_beams = 5,
        #     # temperature = 0.8,
        #     no_repeat_ngram_size=4,
        #     early_stopping=True,
        #     max_tokens=max_tokens,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        # )

        # return (
        #     self.tokenizer.decode(generated_ids[0]),
        #     encodings.input_ids.numel(),
        #     len(generated_ids[0]),
        # )
