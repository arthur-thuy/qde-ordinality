"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from model.build import ORDINAL_MODEL_REGISTRY, build_model
from model.distilbert_ordinal import build_distilbert_ordinal
from model.bert_ordinal import build_bert_ordinal
from model.modernbert_ordinal import build_modernbert_ordinal
from model.roberta_ordinal import build_roberta_ordinal
