# Copyright (c) Microsoft. All rights reserved.

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
#from transformers import AutoModelWithLMHead, AutoTokenizer

from . import InferenceGenerator

# The model used to get the tokenizer can be a little arbitrary
# since the tokenizers are common within the same model type


class SummarizationGenerator(InferenceGenerator.InferenceGenerator):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def perform_inference(self, prompt, context, max_tokens):
        # model = AutoModelWithLMHead.from_pretrained(self.model_name)
        # model.to(self.device)

        # encodings = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        # print("1 {0}".format(encodings), flush=True)
        # outputs = model.generate(encodings, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        # print("2 {0}".format(outputs), flush=True)
        # summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("3 {0}".format(summary), flush=True)
        # return (
        #     summary,
        #     0,
        #     0,
        # )
#################################################        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)

        encodings = self.tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
        generated_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        generated_answers_encoded = model.generate(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            min_length=64,
            max_length=256,
            do_sample=False, 
            early_stopping=True,
            num_beams=8,
            temperature=0,
            top_k=0.2,
            top_p=0.2,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1)
        
        return (
            self.tokenizer.decode(generated_answers_encoded[0], 
                skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
            0,
            0,
        )
            # temperature=1.0,
            # top_k=None,
            # top_p=None,
#####################################################
        # model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # model.to(self.device)

        # encodings = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        # generated_ids = model.generate(encodings, min_length=80, max_length=120)
        # return (
        #     self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        #     encodings.numel(),
        #     len(generated_ids[0]),
        # )
#######################################################
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
