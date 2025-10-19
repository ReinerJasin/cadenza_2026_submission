import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def calculate_attention(
        values: torch.Tensor,       # V: [Batch, Time steps / sequence length, Embedding dimension]
        keys: torch.Tensor,         # K: [B, T, E]
        query: torch.Tensor,        # Q: [B, T, E]
):
    # Compute raw attention scores: Q x K^T
    # query: [B, T, E], keys.transpose(-2, -1): [B, E, T]
    # Result: attention_scores [B, T, T]
    attention_scores = torch.matmul(query, keys.transpose(-2, -1))

    # Scale the scores by sqrt(E) to prevent large values
    attention_scores = attention_scores / math.sqrt(keys.shape[-1])

    # Apply softmax along the last dimension so each query's attention weights sum to 1
    # attention score: [B, T, T]
    attention_scores = F.softmax(attention_scores, dim=-1)

    # Multiply attention weights by V to get final attention output
    # attention_scores: [B, T, T], values: [B, T, E]
    # Result: attention: [B, T, E]
    attention = torch.matmul(attention_scores, values)
    
    return attention, attention_scores

class FeedForward(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()

        # Linear layer 1: projects [B, T, E] → [B, T, E]
        self.layer1 = nn.Linear(embed_size, embed_size)
        # Linear layer 2: projects [B, T, E] → [B, T, E]
        self.layer2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x: [B, T, E]
        x = self.layer1(x)       # [B, T, E]
        x = F.gelu(x)            # apply GELU activation (non-linear)
        x = self.layer2(x)       # [B, T, E]
        
        return x
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()

        self.embed_size = embed_size

        # Learnable linear projections for Q, K, V
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings: torch.Tensor):
        # embeddings: [B, T, E]
        query = self.query_dense(embeddings)     # [B, T, E]
        key = self.key_dense(embeddings)         # [B, T, E]
        value = self.value_dense(embeddings)     # [B, T, E]

        # Calculate attention output and weights
        attention, _ = calculate_attention(value, key, query)       # [B, T, E]

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        # Ensure embed_size is divisible by num_heads (so each head has equal size)
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads     # dimension per head

        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Final linear to recombine all heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, E = x.shape

        # Linear projections
        Q = self.query(x)       # [B, T, E]
        K = self.key(x)         # [B, T, E]
        V = self.value(x)       # [B, T, E]

        # Split into heads: [B, num_heads, T, head_dim]
        # view: [B, T, num_heads, head_dim] → transpose: [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T, Hd]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T, Hd]
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T, Hd]

        # Compute scaled dot-product attention per head
        # Q: [B, H, T, Hd], K^T: [B, H, Hd, T]
        # scores: [B, H, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Softmax to normalize across sequence
        attn_weights = F.softmax(scores, dim=-1)    # [B, H, T, T]

        # Multiply weights by V to get attention output per head
        # attn_weight: [B, H, T, T], V: [B, H, T, Hd]
        # out: [B, H, T, Hd]
        out = torch.matmul(attn_weights, V)

        # Combine heads back: [B, T, E]
        out = out.transpose(1, 2).contiguous().view(B, T, E)        # [B, T, E]

        # Final linear layer to combine heads
        out = self.fc_out(out)      # [B, T, E]

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads=3, use_multihead=False):
        super().__init__()
        
        # Choose between using self_attention or multihead attention (1 or many heads)
        if use_multihead:
            self.attention_layer = MultiHeadAttention(embed_size, num_heads)
        else:
            self.attention_layer = SelfAttentionLayer(embed_size)

        self.feed_forward = FeedForward(embed_size)         # MLP Part
        self.layer_norm1 = nn.LayerNorm(embed_size)         # Normalize features

        self.use_multihead = use_multihead

    def forward(self, x: torch.Tensor):

        if self.use_multihead:
            # Multi-head attention with residual connection
            attn_out = self.attention_layer(x)      # [B, T, E]
            x = self.layer_norm1(attn_out + x)      # [B, T, E]

            # Feed-forward with residual
            ff_out = self.feed_forward(x)           # [B, T, E]
            return F.gelu(ff_out + x)               # [B, T, E]
        else:
            context = self.attention_layer(x)       # [B, T, E]
            context = self.layer_norm1(context)     # [B, T, E]

            context = self.feed_forward(context)    # [B, T, E]
            context = F.gelu(context)               # [B, T, E]
            output = context + x                    # [B, T, E]
            return output

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size:int, max_seq_length: int):
        super().__init__()

        # position: [max_seq_length, 1]
        position = torch.arange(max_seq_length).unsqueeze(1)
        
        # div_term: frequency scaling for sine/cosine
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )

        # Create positional encoding matrix: [max_seq_length, embed_size]
        pe = torch.zeros(max_seq_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)    # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # Odd indices
        
        # Save as buffer (not a parameter but stored with model)
        self.register_buffer("positional_embedding", pe)

    def forward(self, x: torch.Tensor):
        # Add position info to embeddings
        # x: [B, T, E], positional_embedding: [max_seq_length, E]
        # :x.size(1) meaning that we only take the first T positions and broadcast it to x
        return x + self.positional_embedding[: x.size(1), :]
    
class Transformer(nn.Module):
    def __init__(self, embed_size: int, num_layers: int, max_seq_length: int):
        super().__init__()
        
        # Positional Encoding layer
        self.positional_encoding = SinusoidalPositionEncoding(
            embed_size, max_seq_length
        )

        # Stack N transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        # Add positional encoding
        # x: [B, T, E]
        x = self.positional_encoding(x)     # x: [B, T, E]
        
        # Pass through stacked transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)        # x: [B, T, E]

        return x
    
if __name__ == "__main__":
    # Create a transformer model
    # embed_size = 128
    # 3 stacked transformer blocks
    # Supports up to 15 tokens
    transformer = Transformer(embed_size=128, num_layers=3, max_seq_length=15)
    
    # Create random input: batch=2, seq_len=10, embed_size=128
    x = torch.randn(2, 10, 128)
    
    # Forward pass through transformer
    # Output: [2, 10, 128] (same as shape input)
    print(transformer(x).shape)