import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1():
    def __init__(self,tokenizer,text,max_length,stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    #__len__이랑 __getitem__이 있는 객체를 요구하기 때문에 작성
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    
def create_dataloader_v1( text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDatasetV1(tokenizer,text,max_length,stride)

    dataloader = DataLoader(dataset, # type: ignore
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader
