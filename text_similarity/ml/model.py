import torch
import transformers
from tqdm import tqdm
import os
class Dataset(torch.utils.data.Dataset):
    def __init__(self, anchors, positives):
        self.anchors = anchors
        self.positives = positives
    
    def __getitem__(self, idx):
        return {
            'anchor': self.anchors[idx],
            'positive': self.positives[idx]
        }
    
    def __len__(self):
        return len(self.anchors)
    

class SBERT:
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=32, learning_rate=2e-5, epochs=1, scale=20.0):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.cos_sim = torch.nn.CosineSimilarity()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func.to(self.device)
        self.scale = scale
    
    def mean_pool(self,token_embeds, attention_mask):
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    def anchors_and_positives(self, dataset):
        anchors = dataset.map(
        lambda x: self.tokenizer(
                x['premise'], max_length=128, padding='max_length', truncation=True
            ), batched=True
        )
        anchors = anchors.remove_columns(['premise', 'hypothesis', 'token_type_ids'])

        positives = dataset.map(
            lambda x: self.tokenizer(
                x['hypothesis'], max_length=128, padding='max_length', truncation=True
            ), batched=True
        )
        positives = positives.remove_columns(['premise', 'hypothesis', 'token_type_ids'])

        anchors.remove_columns(['label'])
        positives.remove_columns(['label'])

        anchors.set_format(type='torch', output_all_columns=True)
        positives.set_format(type='torch', output_all_columns=True)
        return anchors, positives
    
    def train(self, dataset):
        
        anchors, positives = self.anchors_and_positives(dataset)
        ds = Dataset(anchors, positives)
        
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optim = transformers.AdamW(self.model.parameters(), lr=self.learning_rate, no_deprecation_warning=True)
        total_steps = int(len(anchors) / self.batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps-warmup_steps)
        
        for batch in loader:
            break
        anchor_ids = batch['anchor']['input_ids'].to(self.device)
        anchor_mask = batch['anchor']['attention_mask'].to(self.device)
        pos_ids = batch['positive']['input_ids'].to(self.device)
        pos_mask = batch['positive']['attention_mask'].to(self.device)

        anchor_embeds = self.model( 
            anchor_ids, attention_mask=anchor_mask
        )[0]
        pos_embeds = self.model(pos_ids, attention_mask=pos_mask)[0]

        anchor_embeds = self.mean_pool(anchor_embeds, anchor_mask)
        pos_embeds = self.mean_pool(pos_embeds, pos_mask)
        scores = []
        for anchor in anchor_embeds:
            scores.append(self.cos_sim(anchor.reshape(1, anchor.shape[0]), pos_embeds))

        scores = torch.stack(scores)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=self.device)
        self.loss_func(scores, labels)
        
        for epoch in range(self.epochs):
            self.model.train()
            loop = tqdm(loader, leave=True)
            for batch in loop:
                optim.zero_grad()
                anchor_ids = batch['anchor']['input_ids'].to(self.device)
                anchor_mask = batch['anchor']['attention_mask'].to(self.device)
                pos_ids = batch['positive']['input_ids'].to(self.device)
                pos_mask = batch['positive']['attention_mask'].to(self.device)
                
                anchor_embeds = self.model(
                    anchor_ids, attention_mask=anchor_mask
                )[0]
                
                pos_embeds = self.model(
                    pos_ids, attention_mask=pos_mask
                )[0]
                anchor_embeds = self.mean_pool(anchor_embeds, anchor_mask)
                pos_embeds = self.mean_pool(pos_embeds, pos_mask)
                
                scores = torch.stack([
                    self.cos_sim(
                        anchor.reshape(1, anchor.shape[0]), pos_embeds 
                    ) for anchor in anchor_embeds
                ])
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                loss = self.loss_func(scores*self.scale, labels)
                loss.backward()
                optim.step()
                scheduler.step()
                loop.set_description(f'epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                
    def save(self, path):
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_pretrained(path,safe_serialization=True)