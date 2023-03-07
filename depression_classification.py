import torch
import pickle
import fastparquet

import numpy as np
import pandas as pd

from general_configuration import GeneralConfig
from transformers import RobertaModel, BertTokenizer, BertModel, BertConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline


config = GeneralConfig()
loaded_tokenizer = AutoTokenizer.from_pretrained(config.deproberta_model_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(config.deproberta_model_path)

nlp = TextClassificationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)

all_subreddit_submissions = pd.read_parquet(config.all_subreddits_parquet_path, engine="fastparquet")

depression_severity_res_path = nlp(all_subreddit_submissions.selftext.tolist()[:10], truncation=True)
with open(config.depression_severity_res_path, 'wb+') as f_res:
    pickle.dump(depression_severity_res_path, f_res)
# this is bonkers
