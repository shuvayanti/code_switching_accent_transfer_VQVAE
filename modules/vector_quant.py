import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.logger as logger
import numpy as np

class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()
    
    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        
        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item)
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())
        
        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)
        
        
    def forward_gen(self, x0):
        self.eval()
        if self.normalize_scale:
            print("vq normalize scale")
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        
        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item)
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())
        
        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)
        
        
    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))
                
    def generate_code(self, x0):
        self.eval()
        x = x0
        embedding = self.embedding0
        print("x", x.shape)
        print("embedding", embedding.shape)
        
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        print("x1", x1.shape)
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        print("index", index.shape)
        
        index = torch.cat(index_chunks, dim=0)
        print("index", index.shape)
        print("index", index)
        view_item = index.size(0) * index.size(1)
        print("view_item", view_item)
        view_ref = index + self.offset.cuda()
        print("view_ref", view_ref.shape)
        
        index1 = view_ref.view(view_item)
        print("index1", index1.shape)
        
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        print("output_flat", output_flat.shape)
        output = output_flat.view(x.size())
        print("output", output.shape)
        out0 = (output - x).detach() + x
        print("out0", out0.shape)
        
        return out0, index
        
        
    def customize_code(self, x0, custom_idx):
        self.eval()
        x = x0
        embedding = self.embedding0
        print("x", x.shape)
        print("embedding", embedding.shape)
        
        
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        print("x1", x1.shape)
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        print("index", index.shape)
        
        
        index = torch.cat(index_chunks, dim=0)
        print("index", index.shape)
        print("index", index)
        #[6735, 1]
        
        
#        index = something
        view_item = index.size(0) * index.size(1)
        print("view_item", view_item)
        view_ref = index + self.offset.cuda()
        print("view_ref", view_ref.shape)
        
        
        index1 = view_ref.view(view_item)
        print("index1", index1.shape)
        
        
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        print("output_flat", output_flat.shape)
        output = output_flat.view(x.size())
        print("output", output.shape)
        out0 = (output - x).detach() + x
        print("out0", out0.shape)
        
        return out0, index
        
        
    def forward_custom(self, x0):
        self.eval()
        print('got into vq forward custom')
        target_norm = self.normalize_scale * math.sqrt(x0.size(3))
        x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
        embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        idx= torch.cat(index_chunks, dim=0)
        index = idx
        return idx
        
        
    def forward_codebook_normed(self, n, x0):
        self.eval()
        target_norm = self.normalize_scale * math.sqrt(x0.size(3))
        embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        V = []
        for i in range(0, n):
            V.append([i])
        V = torch.Tensor(V)
        index = V.long().cuda()
        hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
        prob = hist.masked_select(hist > 0) / len(index)
        entropy = - (prob * prob.log()).sum().item()
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item)
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        return output_flat
        
        
    def forward_get_all_codes(self, n):
        self.eval()
        embedding = self.embedding0
        V = []
        for i in range(0, n):
            V.append([i])
        V = torch.Tensor(V)
#        print("V+all", V.shape, type(V))
#        index= torch.cat(V, dim=0)
#        print("index", index.shape)
#        index_chunks = []
#        print("x1+all", x.shape, type(x))
#        for x1_chunk in x1.split(n, dim=0)[:1]:
#            print("x1chunk+all", x1_chunk.shape, type(x1_chunk))
#            item = (x1_chunk - embedding).norm(dim=3).argmin(dim=2)
#            print("item", item.shape, type(item))
#            index_chunks.append(item)
#        index = torch.cat(index_chunks, dim=0)
#        print("index+all", index.shape, type(index))
#        index = torch.cat(index_chunks, dim=0)
        index = V.long().cuda()
        hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
        prob = hist.masked_select(hist > 0) / len(index)
        entropy = - (prob * prob.log()).sum().item()
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item)
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
#        print("of 0", output_flat[0])
#        print("out_flat+all", output_flat.shape)
        return output_flat
        
    def vecs_from_codes(self, indexes):
        #        print(type(indexes))
        #        print(type(indexes[0]))
        #        [print(a) for a in indexes]
        indexes = [torch.Tensor([a]).cuda() for a in indexes]
        #        print(type(indexes))
        #        print(type(indexes[0]))
        index = torch.cat(indexes, dim=0).unsqueeze(1)
        print("index", index.size())
        print("index", index.size(0))
        print("index", index.size(1))
        view_item = index.size(0) * index.size(1)
        print("view_item", view_item)
        view_ref = index + self.offset.cuda()
        print("view_ref", view_ref.shape)
        index1 = view_ref.view(view_item).long()
        print("index1", index1.shape)
        print("index1", type(index1))
        embedding = self.embedding0
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        print("output_flat", output_flat.shape)
        #        output = output_flat.view(x.size())
        #        print("output", output.shape)
        #        out0 = (output - x).detach() + x
        #        print("out0", out0.shape)
        return output_flat
        
        
    def vecs_from_codes_normed(self, indexes):
        '''
        target_norm = self.normalize_scale * math.sqrt(128)
        #x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
        embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        indexes = [torch.Tensor([a]).cuda() for a in indexes]
        index = torch.cat(indexes, dim=0).unsqueeze(1)
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item).long()
        embedding = self.embedding0
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        return output_flat
        '''
        target_norm = self.normalize_scale * math.sqrt(128)
        embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        indexes = [torch.Tensor([a]).cuda() for a in indexes]
        index = torch.cat(indexes, dim=0).unsqueeze(1)
        view_item = index.size(0) * index.size(1)
        view_ref = index + self.offset.cuda()
        index1 = view_ref.view(view_item).long()
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        return output_flat
        
