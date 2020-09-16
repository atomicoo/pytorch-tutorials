# %%
import requests
from pathlib import Path

# %%
BIG_FILE_URL = 'https://norvig.com/big.txt'
DATA_PATH = Path('data')
FILENAME = 'big.txt'
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
if not (DATA_PATH / FILENAME).exists():
    with open((DATA_PATH / FILENAME), 'wb') as big_f:
        response = requests.get(BIG_FILE_URL, )
        
        if response.status_code == 200:
            big_f.write(response.content)
        else:
            print("Unable to get the file: {}".format(response.reason))

# %%
from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# %%
# 创建一个包含空白的BPE model的tokenizer
tokenizer = Tokenizer(BPE())

# %%
# 加入normalizer
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase(),
])

# 加入pre-tokenizer
tokenizer.pre_tokenizer = ByteLevel()

# 加入Decoder
tokenizer.decoder = ByteLevelDecoder()

# %%
from tokenizers.trainers import BpeTrainer

# %%
# 创建BPE Trainer
trainer = BpeTrainer(
            vocab_size=25000, 
            show_progress=True, 
            initial_alphabet=ByteLevel.alphabet())

# 训练BPE model
tokenizer.train(trainer, ['data/big.txt'])
print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

# %%
SAVE_PATH = Path('tokenizers')
PATH = SAVE_PATH / 'bytelevel-bpe-tokenizer-model'
if not PATH.exists():
    PATH.mkdir(parents=True, exist_ok=True)

# %%
# 保存模型
tokenizer.model.save(str(PATH))

# %%
# 在需要时重新载入使用（可与transformers无缝衔接配合使用）
# 注意，实践中这里需要按训练时的情况重新构建好tokenizer再载入model
tokenizer.model = BPE(vocab=str(PATH/'vocab.json'), 
                      merges=str(PATH/'merges.txt'))

# %%
# 编码/解码
encoded = \
    tokenizer.encode("This is a simple input to be tokenized.")
print("Encoded string: {}".format(encoded.tokens))

decoded = \
    tokenizer.decode(encoded.ids)
print("Decoded string: {}".format(decoded))

# %%
from tokenizers import ByteLevelBPETokenizer
# tokenizer提供了一些经典tokenization算法的高级封装
# 譬如可以用`ByteLevelBPETokenizer`简单地重写上面的内容
# 
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
            files=['data/big.txt'], 
            vocab_size=25000, 
            show_progress=True)

SAVE_PATH = Path('tokenizers')
PATH = SAVE_PATH / 'bytelevel-bpe-tokenizer-model'
if not PATH.exists():
    PATH.mkdir(parents=True, exist_ok=True)

tokenizer.save_model(str(PATH))

tokenizer = ByteLevelBPETokenizer(
                    vocab_file=str(PATH/'vocab.json'), 
                    merges_file=str(PATH/'merges.txt'))

encoded = \
    tokenizer.encode("This is a simple input to be tokenized.")
print("Encoded string: {}".format(encoded.tokens))

decoded = \
    tokenizer.decode(encoded.ids)
print("Decoded string: {}".format(decoded))

# %%
# 与transformers搭配使用时，Encoding structure中有用的properties
#   - normalized_str
#   - original_str
#   - tokens
#   - input_ids
#   - attention_mask
#   - special_token_mask
#   - type_ids
#   - overflowing

# %%
