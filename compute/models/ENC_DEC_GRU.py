import torch
import torch.nn as nn
import numpy as np
import random 

from models.common.Embedding import CategoricalEmbedding
from models.common.ContextAdapter import ContextAdapter
from models.common.Dropout import FeatureDropout

class ENC_DEC_GRU(nn.Module):

    def __init__(self,
                 h:int,
                 output_size:int,
                 enc_feat_size:int,
                 enc_hidden_size:int=8,
                 dec_hidden_size:int=8,
                 ncat_embedding:list[int] = None,
                 dim_embedding:list[int] = None,
                 dropout_enc_feat:float=0.0,
                 dropout_dec_feat:float=0.0,
                 tf_config:dict={'tf_type':'linear','tf_max_rate':0.0,'tf_decay':0.0},
                 ) -> None:
          
        super(ENC_DEC_GRU, self).__init__()

        self.h = h
        self.output_size = output_size
        
        self.tf_config = tf_config
        self.tf_rate = 0.0
        # ----------------------------- Encoder ---------------------------------
        self.dropout_enc_feat = FeatureDropout(dropout_enc_feat)

        self.encoder = nn.GRU(input_size=enc_feat_size,
                              hidden_size=enc_hidden_size,
                              batch_first=True)

        # ----------------------------- Context ---------------------------------
        self.cat_features = ncat_embedding is not None
        if self.cat_features:
            self.embeddings = CategoricalEmbedding(ncat_embedding, dim_embedding)

        self.context = ContextAdapter(enc_hidden_size, dec_hidden_size, 1,
                                      dim_embedding, cell_state=False)

        # ----------------------------- Decoder ---------------------------------
        self.dropout_dec_features = FeatureDropout(dropout_dec_feat) 
        self.decoder = nn.GRU(input_size=2,
                              hidden_size=dec_hidden_size,
                              batch_first=True)
        
        # ----------------------------- Output Layer -----------------------------
        self.output = nn.Linear(in_features=dec_hidden_size,
                                out_features=output_size)
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self,
                model_input:dict[str, torch.Tensor]
                ) -> torch.Tensor:
        
        x_dyn = model_input['x_dyn']
        y = model_input['y']

        # embedded_features: (batch_size, sum(embedding_dim))
        cat = self.embeddings(model_input['x_cat']) if self.cat_features else None

        self.dropout_enc_feat.reset_mask()
        enc_input = self.dropout_enc_feat(x_dyn)

        # enc_hidden: (enc_num_layers, batch_size, enc_hidden_size)
        _, enc_hidden = self.encoder(enc_input)

        # dec_hidden: (dec_num_layers, batch_size, dec_hidden_size)
        dec_hidden, _ = self.context(enc_hidden, None, cat)
        
        # dec_input: (batch_size, 1, 1)
        dec_input = x_dyn[:, -1, -1].unsqueeze(1).unsqueeze(2).repeat(1, 1, self.output_size)

        self.dropout_dec_features.reset_mask()

        outputs = []
        for step in range(self.h):

            step_out, dec_input, dec_hidden = self._decoder_step(step,dec_input,dec_hidden,y)                      
            outputs.append(step_out)

        y_hat = torch.stack(outputs, dim=1)

        return y_hat
    
    def _decoder_step(self,
                      step:int,
                      dec_input:torch.Tensor,
                      dec_hidden:torch.Tensor,
                      y:torch.Tensor
                      ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        
        central_idx = self.output_size // 2
        dec_input = dec_input[:,:,central_idx:central_idx+1]
        dec_input = self.dropout_dec_features(dec_input)

        step_feature = torch.full((dec_input.size(0), 1, 1), step/self.h, device=dec_input.device)
        dec_input = torch.cat([dec_input, step_feature], dim=-1)

        dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
        step_out = self.output(dec_output.squeeze(1))

        use_teacher_forcing = random.random() < self.tf_rate
        if use_teacher_forcing:
            y_dec_input = torch.nan_to_num(y[:,step], nan=0.0).unsqueeze(1).repeat(1, self.output_size)         
        else:
            y_dec_input = step_out
                    
        next_input = y_dec_input.unsqueeze(1)

        return step_out, next_input, dec_hidden
    
    def _compute_tf_rate(self,
                         epoch:int,
                         min_rate:float=0.0
                        ) -> float:
        
        type = self.tf_config['tf_type']
        max_rate, rate_decay = self.tf_config['tf_max_rate'], self.tf_config['tf_decay']

        if type == 'linear':
            rate = max(max_rate - epoch*rate_decay, min_rate)
        elif type == 'exponential':
            rate = max(min(np.exp(-rate_decay * epoch),max_rate), min_rate)
    
        return rate