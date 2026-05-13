import torch
import torch.nn as nn

class CategoricalEmbedding(nn.Module):
    """
    Categorical Embedding Module
        Convert categorical features into embeddings vectors and concatenate them 
        Args:
            ncat_embedding: list of integers, number of categories for each categorical feature
            dim_embedding: list of integers, dimension of the embedding for each categorical feature
    """

    def __init__(self,
                 ncat_embedding:list[int],
                 dim_embedding:list[int]
                 ) -> None:
           
        super(CategoricalEmbedding, self).__init__()

        # Create a list of embedding layers for each categorical feature
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in zip(ncat_embedding, dim_embedding)]
        )

    def forward(self,
                cat:torch.Tensor
               ) -> torch.Tensor:
        """
        Forward pass of the Categorical Embedding Module
            Convert categorical features into a single tensor of embeddings representations
            Args:
                cat(torch.Tensor): tensor with categorical features of shape (batch_size, num_cat_features)
            Returns:
                embedded_features(torch.Tensor): tensor with embeddings representations of shape (batch_size, sum(embedding_dim))
        """
        
        # For each categorical feature, get the corresponding embedding representation and concatenate them into a single tensor
        embedded_features_list = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_tensor = torch.cat(embedded_features_list, dim=-1)

        return emb_tensor