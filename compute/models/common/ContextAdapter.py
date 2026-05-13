import torch
import torch.nn as nn

class ContextAdapter(nn.Module):
    """
    ContextAdapter:
        Adapt the hidden/cell state of the encoder to the dimensions required by the decoder.
        Args:
            enc_hidden_size(int): The hidden size of the encoder.
            dec_hidden_size(int): The hidden size of the decoder.
            dim_embedding(list[int]): List with the dimensions of the embeddings for each categorical feature.
            cell_state(bool): Whether the decoder uses a cell state (Only for LSTM).
    """
    def __init__(self,
                 enc_hidden_size:int,
                 dec_hidden_size:int,
                 dec_num_layers:int,
                 dim_embedding:list[int],
                 cell_state:bool
                 ):
        
        super(ContextAdapter, self).__init__()

        self.cell_state = cell_state
        self.dec_num_layers = dec_num_layers

        # Calculate the input size of the context adapter
        embed_size = sum(dim_embedding) if dim_embedding is not None else 0
        input_size = enc_hidden_size + embed_size

        self.adapt_context = (input_size != dec_hidden_size) or (embed_size > 0) 
        if self.adapt_context:
            # Hidden state context adapter
            self.hidden_context = nn.Sequential(nn.Linear(input_size, dec_hidden_size),
                                  nn.Tanh()) 
        
            # Cell state context adapter if cell state is used
            self.cell_context = nn.Sequential(nn.Linear(input_size, dec_hidden_size),
                                nn.Tanh()) if cell_state else None 
            
    def forward(self,
                enc_hidden:torch.Tensor,
                enc_cell:torch.Tensor,
                cat_embedding:torch.Tensor=None
                ) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the ContextAdapter.
            Args:
                enc_hidden(torch.Tensor): The hidden state of the encoder shape (num_directions, batch_size, hidden_size).
                enc_cell(torch.Tensor): The cell state of the encoder (Only for LSTM) shape (num_directions, batch_size, hidden_size)
            Returns:
                hidden_context(torch.Tensor): The hidden state context adapted for the decoder shape (dec_num_layers, batch_size, hidden_size).
                cell_context(torch.Tensor): The cell state context adapted for the decoder (Only for LSTM) shape (dec_num_layers, batch_size, hidden_size).
        """

        if cat_embedding is not None:
            hidden_context = torch.cat([enc_hidden[-1], cat_embedding], dim=-1)
            if self.cell_state and enc_cell is not None:
                cell_context = torch.cat([enc_cell[-1], cat_embedding], dim=-1)
        else:
            hidden_context = enc_hidden[-1]
            if self.cell_state:
                cell_context = enc_cell[-1]
        
        # Decoder initial hidden/cell state need to be of shape (dec_num_layers, batch_size, hidden_size)
        # Adapt the dimensions of the hidden/cell state to the decoder
        if self.adapt_context:
            hidden_context = self.hidden_context(hidden_context)
            if self.cell_state:
                cell_context = self.cell_context(cell_context)

        # Repeat for the number of decoder layers
        hidden_context = hidden_context.unsqueeze(0).repeat(self.dec_num_layers, 1, 1)
        cell_context = cell_context.unsqueeze(0).repeat(self.dec_num_layers, 1, 1) if self.cell_state else None

        return hidden_context, cell_context



