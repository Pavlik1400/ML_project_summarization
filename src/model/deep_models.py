from transformers import GPT2LMHeadModel, OpenAIGPTLMHeadModel, BertLMHeadModel, RobertaForMaskedLM, XLNetLMHeadModel, \
    AlbertForMaskedLM

MODELS = {
    "gpt2": GPT2LMHeadModel,
    "openai-gpt": OpenAIGPTLMHeadModel,
    "bert-base-uncased": BertLMHeadModel,
    "bert-base-cased": BertLMHeadModel,
    "roberta-base": RobertaForMaskedLM,
    "xlnet-base-cased": XLNetLMHeadModel,
    "albert-base-v2": AlbertForMaskedLM,
    "albert-large-v2": AlbertForMaskedLM,
    "albert-xlarge-v2": AlbertForMaskedLM
}
