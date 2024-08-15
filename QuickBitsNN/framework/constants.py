################################################################
import sys, os, os.path

#  Batch Assignment for Training/Evaluations
#----------------------------------------------------------------
multiplier          = 32 #128 #128 #128#//2
BATCH_SIZE          = multiplier
BATCH_SAMPLES       = BATCH_SIZE
VALIDATION_SAMPLES  = BATCH_SIZE 
TEST_SAMPLES        = BATCH_SIZE



#  Tokenizer() Constants
#----------------------------------------------------------------
START_TOKEN     = f'>'
PADDING_TOKEN   = f'_'
END_TOKEN       = f'|'
HASHEX_VOCABULARY   = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']


MAX_INT         = 4294967295
