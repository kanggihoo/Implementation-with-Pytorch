import random
import torch
import itertools
import torch.utils.data as data
class BERTDataset(torch.utils.data.Dataset):
  def __init__(self , pairs , tokenizer , seq_len=64):
    self.lines = pairs
    self.tokenizer = tokenizer
    self.seq_len = seq_len
    self.corpus_lines = len(pairs)
  def __len__(self):
    return self.corpus_lines

  def __getitem__(self , idx):
    # Step 1 get random sentence pair
    t1, t2, is_next_label = self.get_sent(idx)


    # Step 2: randomly replace each token to MASK or random token or Not replace
      # => using tokenizer, split sentence to tokens and then encode each token to interger value
    t1_random, t1_label = self.random_word(t1)
    t2_random, t2_label = self.random_word(t2)

    # Step 3: adding special token (CLS  SEP , PAD) ,
    # First sentence add CLS , SEP totken on the left,right second sentence add SEP on the right
    t1 = [self.tokenizer.vocab["[CLS]"]] + t1_random + [self.tokenizer.vocab["[SEP]"]]
    t2 = t2_random + [self.tokenizer.vocab["[SEP]"]]

    # Label data add PAD token instead of PAD , CLS , SEP token
    t1_label = [self.tokenizer.vocab["[PAD]"]] + t1_label + [self.tokenizer.vocab["[PAD]"]]
    t2_label = t2_label + [self.tokenizer.vocab["[PAD]"]]

    # Step 4 : combine sentence1 and sentence2
    segment_label = ([1 for _ in range(len(t1))]+[2 for _ in range(len(t2))])[:self.seq_len] # identify which is first or second sentence
    bert_input = (t1+t2)[:self.seq_len]
    bert_label = (t1_label + t2_label)[:self.seq_len]
    padding = [self.tokenizer.vocab["[PAD]"] for _ in  range(self.seq_len - len(bert_input))] # when len(seq_len) > len(bert_input) add padding token on last inputdata
    bert_input.extend(padding) , bert_label.extend(padding) , segment_label.extend(padding)
    output = {
        "bert_input" : bert_input,
        "bert_label" : bert_label,
        "segment_label" : segment_label,
        "is_next" : is_next_label
    }
    return {k: torch.tensor(v) for k , v in output.items()}

  def random_word(self , sentence): # random replacement of tokens in each sentence
    tokens = sentence.split()
    output_label = []
    output = []

    # 15% of token replace
    for i, token in enumerate(tokens):
        prob = random.random()

        # remove CLS , SEP token
        token_id = self.tokenizer(token)["input_ids"][1:-1]


        if prob < 0.15: # replace token
            prob /= 0.15
            # 80% token change to mask token (15%로 선택된 token 중 80% 확률로 MASK 토큰으로 변환)
            if prob < 0.8:
                for i in range(len(token_id)): # 해당 단어를 token화 했을때 반환된 토큰의 개수 반큼 MASK 토큰으로 변환
                    output.append(self.tokenizer.vocab["[MASK]"])
                # 10% token change to random token # (15% 선택된 token 중 10% 확률로 random 토큰으로 변환)
            elif prob < 0.9:
                for i in range(len(token_id)):
                    output.append(random.randrange(len(self.tokenizer.vocab)))

            #  10% token change current token # (15% 선택된 token 중 10% 확률로 기존의 토큰으로 유지)
            else:
                output.append(token_id)

            output_label.append(token_id)

        else:  # not replace token
            output.append(token_id)
            for i in range(len(token_id)):
                output_label.append(0)
    # flattening
    output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
    output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
    assert len(output) == len(output_label) , f"{len(output)}  ,  {len(output_label)} , {output} , {output_label}"
    return output, output_label



  def get_sent(self , idx): #
    # return random sentence
    t1 , t2 = self.get_corpus_line(idx)

    # negative of positive pair , for next sentence prediction
    # when two sentences are related, return 1 , else return 0
    if random.random() > 0.5:
      return t1,t2 , 1
    else:
      return t1,self.get_random_line() , 0

  def get_corpus_line(self , idx):
    # return sentence pair
    return self.lines[idx][0] , self.lines[idx][1]

  def get_random_line(self):
    # return random sentence
    random_idx = random.randint(0, self.corpus_lines)
    return self.lines[random_idx][1]
