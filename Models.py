import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

from utility.load_data import *

#This source file is based on the GRec published by Bo Li et al.
#We would like to thank and offer our appreciation to them.
#Original algorithm can be found in paper: Embedding App-Library Graph for Neural Third Party Library Recommendation. ESEC/FSE ’21




class HCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list):
        super(HCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.n_layers = layer_num
        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        torch.manual_seed(50)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        torch.manual_seed(50)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_u1, adj_u2, adj_i1, adj_i2, adj_cat):

        device = self.user_embedding.weight.device  # Get the device of the model parameters

        # Ensure all tensors are on the same device as model parameters
        adj_u1 = adj_u1.to(device)
        adj_u2 = adj_u2.to(device)
        adj_i1 = adj_i1.to(device)
        adj_i2 = adj_i2.to(device)
        adj_cat = adj_cat.to(device)



        hu = self.user_embedding.weight
        hi = self.item_embedding.weight

        # User embeddings update
        user_embeddings = [hu]
        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_u2, user_embeddings[-1])
            print("adj_cat shape:", adj_cat.shape)
            print("t shape:", t.shape)
            t = torch.sparse.mm(adj_u1, t)

            
            # Adding the new categorical scale
            t_cat = torch.sparse.mm(adj_cat, t)
            t_cat = self.dropout_list[i](t_cat)  # Applying dropout to the updated embeddings
            user_embeddings.append(t_cat)
        u_emb = torch.mean(torch.stack(user_embeddings, dim=1), dim=1)

        # Item embeddings update
        item_embeddings = [hi]
        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_i2, item_embeddings[-1])
            t = torch.sparse.mm(adj_i1, t)
            # Adding the new categorical scale
            t_cat = torch.sparse.mm(adj_cat, t)
            t_cat = self.dropout_list[i](t_cat)  # Applying dropout to the updated embeddings
            item_embeddings.append(t_cat)
        i_emb = torch.mean(torch.stack(item_embeddings, dim=1), dim=1)

        return u_emb, i_emb











    # def forward(self, adj_matrices):
    #     # Ensure adj_matrices is a tuple of 3 adjacency matrices
    #     if not isinstance(adj_matrices, tuple) or len(adj_matrices) != 3:
    #         raise ValueError("adj_matrices must be a tuple of 3 adjacency matrices.")

    #     print(f"Initial shapes: {adj_matrices[0].shape}, {adj_matrices[1].shape}, {adj_matrices[2].shape}")
    #     adj_matrix_user_item, adj_matrix_item_medium, adj_matrix_item_coarse = adj_matrices
    #     print(f"Post-operation shapes: {adj_matrix_user_item.shape}, {adj_matrix_item_medium.shape}, {adj_matrix_item_coarse.shape}")

    #     # Initial user and item embeddings
    #     user_embeddings = self.user_embedding.weight
    #     item_embeddings = self.item_embedding.weight
        
    #      # Print the shapes of the adjacency matrices
        
    #     print(f"Shape of adj_matrix_user_item: {adj_matrix_user_item.shape}")
    #     print(f"Shape of adj_matrix_item_medium: {adj_matrix_item_medium.shape}")
    #     print(f"Shape of adj_matrix_item_coarse: {adj_matrix_item_coarse.shape}")
    #     print(f"Shape of item_embeddings: {item_embeddings.shape}")

       
    #     #Shape of adj_matrix_user_item: torch.Size([31421, 727])
    #     #Shape of adj_matrix_item_medium: torch.Size([727, 727])
    #     #Shape of ad`j_matrix_item_coarse: torch.Size([727, 727])
    #      #Shape of item_embeddings: torch.Size([727, 128])

    #     # User embeddings updated with item information (Fine scale)
    #     user_embeddings_fine = torch.sparse.mm(adj_matrix_user_item, item_embeddings)   # [31421, 727] *  [727, 128]

    #     # Item embeddings updated with item-item relationships (Medium scale)
    #     item_embeddings_medium = torch.sparse.mm(adj_matrix_item_medium, item_embeddings)   # [727, 727] *  [727, 128]

    #     # Item embeddings updated with item-item relationships (Coarse scale)
    #     item_embeddings_coarse = torch.sparse.mm(adj_matrix_item_coarse, item_embeddings)  # [727, 727] *  [727, 128] 

    #     # Combining item embeddings from Medium and Coarse scales
    #     combined_item_embeddings = (item_embeddings_medium + item_embeddings_coarse) / 2

    #     return user_embeddings_fine, combined_item_embeddings


    # def combine_embeddings(self, embeddings_list):
    #     # Example: Concatenate embeddings and pass through a linear layer
    #     combined = torch.cat(embeddings_list, dim=-1)
    #     combined = self.combination_layer(combined)  # Define this layer in __init__
    #     return combined

