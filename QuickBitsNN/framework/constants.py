################################################################
import sys, os, os.path



#  Batch Assignment for Training/Evaluations
#----------------------------------------------------------------
multiplier          = 12
BATCH_SIZE          = 128 * multiplier
BATCH_SAMPLES       = 128 * multiplier
VALIDATION_SAMPLES  = 128 * multiplier
TEST_SAMPLES        = 128 * multiplier



#  Tokenizer() Constants
#----------------------------------------------------------------
START_TOKEN     = f'>'
PADDING_TOKEN   = f'_'
END_TOKEN       = f'|'
HASHEX_VOCABULARY   = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
VOCABULARY          = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f', START_TOKEN, PADDING_TOKEN, END_TOKEN]
HASHEX_TO_INDEX = {v:k for k,v in enumerate(VOCABULARY)}
INDEX_TO_HASHEX = {k:v for k,v in enumerate(VOCABULARY)}
PADDING_TOKEN_INDEX   = HASHEX_TO_INDEX[PADDING_TOKEN]
START_TOKEN_INDEX   = HASHEX_TO_INDEX[START_TOKEN]
END_TOKEN_INDEX   = HASHEX_TO_INDEX[END_TOKEN]

MAX_INT         = 4294967295
