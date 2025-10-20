# Quantization goal is to map continuous vectors into discrete or categorical tokens.
# Basically we have a sequence of multidimensional sequence, and we want to convert it into sequence of discrete tokens that is categorical
#
# The idea is that we train a codebook which is a collection of 1000 random vectors
# each is the same size of the embeddings (take 512 as an example).
# So we will have [1000, 512] dimension for codebook
#
# The way we map it is by take the closest value just like snapping it to the nearest point
# The problem is that there is always an error between the input vector and the nearest point which causes loss of information.
# 
# Residual Vector Quantization
# Key Idea: Uses multiple codebooks.
# Represent input vector with several codebooks.
# Improves vector quantization by using multiple codebooks.
# After using input and match it with code book, we can calculate the error as residual
# Then we use the residual and put it in the next codebook as the input
# Do this as much as we want for n codebook, and each residual will tell use the information we missed from the previous codebook.

import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings    # int (N), number of codes
        self.embedding_dim = embedding_dim      # int (E), code dimension
        
        # Initialize embedding with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)        # [N, E]
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)                  # Initialize embedding weights randomly
        
        # Weight for commitment loss (helps encoder commit to chosen embeddings)
        self.commitment_cost = commitment_cost
        
    def forward(self, x):
        # Shape: [B, T, E]
        batch_size, sequence_length, embedding_dim = x.shape
        
        # Flatten input to [B*T, E] so we can calculate distances for each vector
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)     # [B*T, E]
        
        # Calculate pairwaise euclidean distance between each input vector and each embedding vector
        # [B*T, N]
        distances = torch.cdist(flat_x, self.embedding.weight, p=2)   # p=2 for euclidean distance
        
        # For each input vector, find the index of the nearest embedding
        # Encoding: closest embedding
        encoding_indices = torch.argmin(distances, dim=1)       # [B*T]
        
        # Replace each input vector with its closest embedding vector (quantized representation)
        quantized = self.embedding(encoding_indices)            # [B*T, E]
        
        # Reshape back to original shape
        quantized = quantized.view(batch_size, sequence_length, embedding_dim)     # [B, T, E]
        
        # Compute vector quantization losses with scaling
        # e_talent_loss: encourages encoder output to commit to chosen embeddings
        # q_talent_loss: moves embeddings close to encoder output
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)           # scalar
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)           # scalar
        
        # Total quantization loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss         # scalar
        
        # Straight-through estimator
        # We want gradients to flow through x, not the discrete lookup
        quantized = x + (quantized - x).detach()      # [B, T, E]
        
        return quantized, loss
    
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        
        # Create multiple vector quantizers (stack codebooks)
        # Each will quantize the residual error from the previous one
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim) for _ in range(num_codebooks)
            ]
        )
        
    def forward(self, x):
        residual = x.clone()
        out = torch.zeros_like(x)
        total_loss = 0.0
        
        for codebook in self.codebooks:
            quantized, q_loss = codebook(residual)        # this_output: [B, T, E]
            residual = residual - quantized                         # residual for next level: [B, T, E]
            out = out + quantized                   # accumulate reconstruction: [B, T, E]
            total_loss += q_loss                    # sum scalar losses
            
        return out, total_loss
    
# if __name__ == "__main__":
#     rvq = ResidualVectorQuantizer(num_codebooks=2, codebook_size=16, embedding_dim=128)
#     x = torch.randn(2, 12, 128, requires_grad=True)
    
#     optimizer = torch.optim.Adam(rvq.parameters(), lr=0.005)
    
#     for i in range(4):
#         output, vq_loss = rvq(x)