from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import tqdm
class Tokenizer():
    def __init__(self, pairs):
        self.pairs = pairs
        
    def train_tokenizer(self):
        # WordPiece tokenizer
        if not Path("./data").exists():
            Path("./data").mkdir(exist_ok=True)
            text_data = []
            file_count = 0
            for sample in tqdm.tqdm(x[0] for x in self.pairs):
                text_data.append(sample)

                if len(text_data) == 10000:
                    with open(f"./data/text{file_count}.txt" , "w" , encoding = "utf-8") as f:
                        f.write("\n".join(text_data))
                    text_data= []
                    file_count +=1
            paths = [str(s) for s in Path("./data").glob("**/*.txt")]
        else:
            print("tokenizer 학습 데이터 존재")

        ## training tokenizer
        if not Path("./bert-it-1").exists():
            tokenizer = BertWordPieceTokenizer(
                clean_text = True,
                handle_chinese_chars = False,
                strip_accents = False,
                lowercase = True
            )
            tokenizer.train(
                files = paths,
                vocab_size = 30000,
                min_frequency = 5,
                limit_alphabet = 1000,
                wordpieces_prefix = "##",
                special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            )
            Path("./bert-it-1").mkdir()
            tokenizer.save_model("./bert-it-1" , 'bert-it')
        else:
            print("Tokenizer 학습 파일 존재 하므로 모델을 불러옵니다.")
        
        tokenizer = self.load_tokenizer()
        return tokenizer
        
    def load_tokenizer(self):
        file =  "bert-it-1/bert-it-vocab.txt"
        tokenizer = BertTokenizer.from_pretrained(file ,local_files_only=True)
        return tokenizer
    