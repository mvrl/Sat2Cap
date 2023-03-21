import pytorch_lightning as pl
import torch.nn as nn
import torch
import webdataset as wds
from transformers import CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from tqdm import tqdm
import numpy as np

#local imports
from ..multidata import MultiData
from ..evaluations.metrics import Retrieval
from .clip import Clip



class GeoMoCo(pl.LightningModule):

    def __init__(self, hparams):
        #save hyperparameters
        super().__init__()
        print('Implementing MoCo GeoClip')
        self.save_hyperparameters(hparams)
        print(f'Args.vit is {self.hparams.vit}')
        #set path attributes
        self.train_path = self.hparams.train_path
        self.vali_path = self.hparams.vali_path
        self.test_path = self.hparams.test_path

        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        
        #this is deprecated in favor of self.save_hyperparameters(hparams)
        #self.hparams = hparams
        #todo:modify the img_processor config file for the large overhead images
        #self.imo_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')

        #frozen image encoder to get ground level image CLIP embeddings
        self.img_encoder = Clip(self.hparams,'ground_level')
    
        if self.hparams.freeze_clip:
            self.img_encoder.eval()
            for params in self.img_encoder.parameters():
                params.requires_grad=False
      
        #trainable image encoder to get overhead image CLIP embeddings
        self.imo_encoder = Clip(self.hparams, 'overhead')
        self.imo_encoder.train()
        
        #instantiate the learnable temperature parameter and queue size
        self.K = self.hparams.queue_size
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/self.hparams.temperature)]))       
        
        #initialize queue
        self.register_buffer('queue', torch.randn(self.hparams.dim_size,self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        #initialize pointer for queue
        self.register_buffer('queue_ptr', torch.zeros(1,dtype=torch.long))


        if self.hparams.prompt_loss:
            #initialize the text model
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.vit_map[self.hparams.vit])
            self.text_model = CLIPTextModelWithProjection.from_pretrained(self.vit_map[self.hparams.vit])
            for params in self.text_model.parameters():
                params.requires_grad=False
        
            self.ground_text_embeddings, self.overhead_text_embeddings = self.get_text_stuff() #Both are 21x512
       
        #check if a valid data path is provided
        if self.train_path:
            self.trainset = MultiData(self.hparams).get_ds(mode='train')
        else:
            raise ValueError('Valid path to webdataset file is required')
        
        #test for validation file
        if self.vali_path:
            self.valiset = MultiData(self.hparams).get_ds(mode='test')
        else:
            self.valiset = None

        #test for test file
        if self.test_path:
            self.testset = MultiData(self.hparams).get_ds(mode='test')
        else:
            self.testset = None
        
        #confirm training state for two encoders
        print(f'Ground Level Image encoder training mode:{self.img_encoder.training}')
        print(f'Overhead Image encoder training model:{self.imo_encoder.training}')


    #fill the queue with initial values    
    def on_fit_start(self):
        self.fill_queue()
        pass

    #clamp the temperature parameter
    def on_train_batch_end(self,outputs,batch, batch_idx):
        if self.logit_scale.data > np.log(100):
            self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, np.log(100))

    #forward function that runs during inference
    def forward(self, batch):
        img, imo, _,keys = batch
        #get the ground img encodings and detach
        with torch.set_grad_enabled(False): #equivalent to torch.no_grad()
            ground_img_embeddings, unnormalized_ground_img_embeddings = self.img_encoder(img).to(self.device)
        #get the overhead image embeddings undetached
        overhead_img_embeddings, unnormalized_overhead_img_embeddings = self.imo_encoder(imo).to(self.device)
        code.interact(local=dict(globals(), **locals()))

        return{'ground_img_embeddings':ground_img_embeddings, 
            'overhead_img_embeddings':overhead_img_embeddings,
            'unnormalized_ground_img_embeddings': unnormalized_ground_img_embeddings,
            'unnormalized_overhead_img_embeddings': unnormalized_overhead_img_embeddings,
            'keys':keys
        }
    
    #shred step for test and validation that returns embeddings and clip loss
    def shared_step(self,batch):
        #get embeddings from forward function
        embeddings = self(batch)
        normalized_ground_img_embeddings = embeddings['ground_img_embeddings']
        normalized_overhead_img_embeddings = embeddings['overhead_img_embeddings']
        
        #positive logits:Nx1
        l_pos = torch.einsum("nc,nc->n", [normalized_overhead_img_embeddings, normalized_ground_img_embeddings]).unsqueeze(-1)

        #negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [normalized_overhead_img_embeddings, self.queue.clone().detach()])

        #logits:Nx(1+K) #N overhead images with K+1 ground level images
        logits = torch.cat([l_pos, l_neg], dim=1)

        #Compute temperature
        logit_scale = self.logit_scale.exp()
        # if logit_scale > 100:
        #     self.logit_scale = nn.Parameter(torch.tensor([np.log(100)]))
        #     logit_scale = self.logit_scale.exp()

        #Apply temperature
        logits_per_overhead_img = logits*logit_scale # Nx(1+K) Overhead images in batch x Ground images in Queue+1

        #labels:positive key indicators:all will be zero as the gt is in the first column
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

        #compute cross_entropy loss between the cross-modal similarities and hard gt
        cross_contrastive_loss = contrastive_loss(logits_per_overhead_img, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(normalized_ground_img_embeddings)

        if self.hparams.prompt_loss:
            #compute the cross entropy loss between for the text prompt to image similarity between the two modalities
            text_to_ground_sim = torch.matmul(normalized_ground_img_embeddings,self.ground_text_embeddings.t().to('cuda')) # Nx21
            text_to_overhead_sim = torch.matmul(normalized_overhead_img_embeddings, self.overhead_text_embeddings.t().to('cuda')) # Nx21

            prompt_similarity_loss = nn.functional.cross_entropy(input=text_to_overhead_sim, target=text_to_ground_sim)

            loss = 0.7*cross_contrastive_loss + 0.3*prompt_similarity_loss

            return {'loss':loss,
                    'contrastive_loss':loss,
                    'prompt_similarity_loss':prompt_similarity_loss, 
                    'logits_per_overhead_img': logits_per_overhead_img,
                    'normalized_ground_img_embeddings':normalized_ground_img_embeddings,
                    'normalized_overhead_img_embeddings':normalized_overhead_img_embeddings
            }
            
        else:
            return {'loss':cross_contrastive_loss,
                    'contrastive_loss':cross_contrastive_loss,
                    'prompt_similarity_loss': torch.tensor(0, dtype=torch.float32),
                    'logits_per_overhead_img': logits_per_overhead_img,
                    'normalized_ground_img_embeddings':normalized_ground_img_embeddings,
                    'normalized_overhead_img_embeddings':normalized_overhead_img_embeddings,
                    'logit_scale':logit_scale
            }
    
    #forward pass for each batch in training 
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['loss']
        train_contrastive_loss = outputs['contrastive_loss']
        prompt_similarity_loss = outputs['prompt_similarity_loss']
        self.log('loss', loss, sync_dist=True, batch_size=self.hparams.train_batch_size)
        self.log('contrastive_loss', train_contrastive_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.train_batch_size)
        self.log('prompt_loss', prompt_similarity_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.train_batch_size)
        self.log('logit_scale', outputs['logit_scale'], prog_bar=True, batch_size=self.hparams.train_batch_size)
        return loss 
    
    #forward pass for each batch in validation
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss']
        val_contrastive_loss = outputs['contrastive_loss']
        val_prompt_similarity_loss = outputs['prompt_similarity_loss']
        self.log('val_loss', val_loss, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        self.log('val_contrastive_loss', val_contrastive_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.train_batch_size)
        self.log('val_prompt_loss', val_prompt_similarity_loss, sync_dist=True, batch_size=self.hparams.train_batch_size)
        return {'val_loss':outputs['loss'],'normalized_ground_img_embeddings':outputs['normalized_ground_img_embeddings'], 'normalized_overhead_img_embeddings':outputs['normalized_overhead_img_embeddings']}

    #compute retrieval metrics for a random batch of validation 
    def validation_epoch_end(self, outputs):
        if len(outputs)==0:
            print('Skipping Validatiion Epoch End')
            pass
        else:
            random_batch = np.random.randint(0,len(outputs))
            validation_embeddings = outputs[random_batch]
            ground_img_embeddings = validation_embeddings['normalized_ground_img_embeddings']
            overhead_img_embeddings = validation_embeddings['normalized_overhead_img_embeddings']
            retrieval = Retrieval(k=self.hparams.top_k)
            retrieval_metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
            self.log(f'top_k_score', retrieval_metric, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
            #print(f'Retrieval Metric on validation set is {retrieval_metric}') 
            return {'top_k_score':retrieval_metric}
          
    def train_dataloader(self):

        trainloader = wds.WebLoader(self.trainset, batch_size=None,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
        trainloader = trainloader.unbatched().shuffle(10000).batched(self.hparams.train_batch_size)
        return trainloader

    def val_dataloader(self):
        if self.valiset:
            valloader = wds.WebLoader(self.valiset, batch_size=None, #batch_size=self.hparams.val_batch_size,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
            valloader = valloader.unbatched().shuffle(10000).batched(self.hparams.val_batch_size)
            return valloader
        pass

    def test_dataloader(self):
        if self.testset:
            testloader = wds.WebLoader(self.testset, batch_size=None, #batch_size=self.hparams.val_batch_size,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
            #testloader = testloader.unbatched().shuffle(10000).batched(self.hparams.val_batch_size)
            return testloader

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        self.optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.imo_encoder.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=0.2,
            betas=(0.9,0.98),
            eps=1e-6
        )
        
        self.warm_up_iterations = self.hparams.warmup_its
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = self.warm_up_iterations
        )
        return {'optimizer': self.optim, 
        'lr_scheduler': {
            'name':'trian/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }
    
    @torch.no_grad()
    def fill_queue(self):
        num_steps_to_fill = self.hparams.queue_size/self.hparams.train_batch_size
        fill_dataset = MultiData(self.hparams).get_ds(mode='queue')
        print('Filling Queue with initial values')
        for i, batch in tqdm(enumerate(fill_dataset)):
            img, _, _,_ = batch
            #get the ground img encodings and detach
            with torch.set_grad_enabled(False): #equivalent to torch.no_grad()
                normalized_ground_img_embeddings,_ = self.img_encoder(img)   #keys: NxC
                self._dequeue_and_enqueue(normalized_ground_img_embeddings)
            if i==num_steps_to_fill:
                break
        
        print('Queue Fill Successful')


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0,f'Size Mismatch: Queue Size is {self.K} and Batch size is {batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    #only used when prompt loss is true
    def get_text_stuff(self):
        labels = [
                    'A busy city street lined with skyscrapers',
                    'A peaceful park with a pond and ducks swimming',
                    'A quaint small town square with a fountain',
                    'A bustling outdoor market with vendors selling fruits and vegetables',
                    'A deserted beach with crashing waves',
                    'A winding mountain path with beautiful views',
                    'An ancient temple with intricate carvings',
                    'A modern art museum with colorful exhibits',
                    'A charming village with cobblestone streets and thatched roofs',
                    'A crowded shopping mall with neon lights',
                    'A serene forest with tall trees and chirping birds',
                    'A quiet suburban neighborhood with manicured lawns',
                    'A vast desert with towering sand dunes',
                    'A lively nightclub with music and dancing',
                    'A sprawling industrial complex with smokestacks and machinery',
                    'A picturesque vineyard with rows of grapes',
                    'A peaceful monastery with chanting monks',
                    'A high-tech research laboratory with advanced equipment',
                    'A cozy coffee shop with comfortable seating',
                    'A historic castle with turrets and a moat',
                    'A colorful carnival with rides and games',
                    'A bustling train station with trains arriving and departing',
                    'A charming countryside with rolling hills and grazing animals',
                    'A bright and modern hospital with state-of-the-art technology',
                    'A quaint seaside town with a lighthouse and boardwalk',
                    'A crowded sports stadium with cheering fans',
                    'A peaceful cemetery with neatly arranged headstones',
                    'A busy airport with planes taking off and landing',
                    'A romantic garden with fragrant flowers',
                    'A quiet bookstore with shelves filled with books',
                    'A charming bed and breakfast with cozy rooms',
                    'A futuristic city with flying cars and towering buildings',
                    'A deserted ghost town with abandoned buildings',
                    'A quaint coastal village with fishing boats',
                    'A busy harbor with cargo ships and cranes',
                    'A scenic overlook with panoramic views',
                    'A historic courthouse with columns and a bell tower',
                    'A bustling convention center with booths and exhibits',
                    'A quiet cabin in the woods with a cozy fireplace',
                    'A colorful street market with handmade crafts',
                    'A peaceful orchard with rows of fruit trees',
                    'A lively dance club with pulsating music',
                    'A tranquil botanical garden with exotic plants',
                    'A busy financial district with tall buildings and banks',
                    'A quaint bed and breakfast with a garden',
                    'A modern sports arena with scoreboard and jumbotron',
                    'A charming ice cream shop with outdoor seating',
                    'A sprawling ranch with horses and cattle',
                    'A lively amusement park with roller coasters and attractions',
                    'A serene lake with swans and paddleboats',
                    'A bustling bus station with buses arriving and departing',
                    'A quaint teahouse with traditional tea ceremonies',
                    'A peaceful mountain cabin with a fireplace and hot tub',
                    'A busy shipping yard with cranes and cargo containers',
                    'A charming candy shop with jars of sweets',
                    'A modern office building with glass walls and elevators',
                    'A quiet yoga studio with soft music',
                    'A picturesque waterfall with cascading water',
                    'A bustling university campus with students and professors',
                    'A quaint wine bar with local wines',
                    'A serene river with kayakers and canoes',
                    'A busy convention center with exhibits and presentations',
                    'A cozy bistro with outdoor seating',
                    'A sprawling ranch with acres of crops',
                    'A charming bakery with delicious pastries',
                    'A modern data center with racks of servers'
        ]

        print(f'Using {len(labels)} ChatGPT prompts')
        #generate prompts from labels
        ground_prompts = ['a photo of a '+label for label in labels]
        overhead_prompts = ['an aerial image of a '+label for label in labels]
        
        #process the prompts
        ground_inputs = self.text_tokenizer(ground_prompts, padding=True, return_tensors='pt')
        overhead_inputs = self.text_tokenizer(overhead_prompts, padding=True, return_tensors='pt')

        #generate the embeddings
        ground_text_embeddings = self.text_model(**ground_inputs).text_embeds
        ground_text_embeddings = ground_text_embeddings/ground_text_embeddings.norm(p=2,dim=-1,keepdim=True)
        
        overhead_text_embeddings = self.text_model(**overhead_inputs).text_embeds
        overhead_text_embeddings = overhead_text_embeddings/overhead_text_embeddings.norm(p=2,dim=-1,keepdim=True)

        return (ground_text_embeddings, overhead_text_embeddings)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

#compute cross_entropy
def contrastive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)

#computer cross entropy for the similarity matrix both rowwise and columnwise
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity)
    ground_img_loss = contrastive_loss(similarity.t())
    return (overhead_img_loss + ground_img_loss) / 2.0