import torch
import torch.nn as nn

class GatingMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, dna_feat, prot_feat):
        combined = torch.cat([dna_feat, prot_feat], dim=-1)
        g = self.gate(combined)
        
        fused = g * dna_feat + (1 - g) * prot_feat
        return fused

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, context):
        # x: [batch, dim] -> [batch, 1, dim] (to fit MultiheadAttention)
        x_q = x.unsqueeze(1)
        c_kv = context.unsqueeze(1)

        attn_out, _ = self.attn(x_q, c_kv, c_kv)
        return self.norm(x + attn_out.squeeze(1))

class ModalityProjector(nn.Module):
    def __init__(self, emb_dim, proj_dim, dropout, feature_mode='all'):
        super().__init__()
        self.feature_mode = feature_mode

        if feature_mode == 'all': # [ref, alt, diff]
            in_dim = emb_dim*3
        elif feature_mode == 'ref_alt': # [ref, alt]
            in_dim = emb_dim*2
        else: # 'diff' hoặc 'ref' hoặc 'alt'
            in_dim = emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, ref, alt):
        if self.feature_mode == 'all':
            diff = alt - ref
            x = torch.cat([ref, alt, diff], dim=-1)
        elif self.feature_mode == 'ref_alt':
            x = torch.cat([ref, alt], dim=-1)
        elif self.feature_mode == 'diff':
            x = alt - ref
        elif self.feature_mode == 'ref':
            x = ref
        else: # 'alt'
            x = alt
            
        return self.net(x)


class FusionClassifier(nn.Module):
    def __init__(self, dna_dim, prot_dim, proj_dim, hidden_dims, dropout, mode='both', fusion_method='concat', feature_mode='all', use_gating=True):
        super().__init__()
        self.mode = mode
        self.fusion_method = fusion_method
        self.feature_mode = feature_mode
        self.use_gating = use_gating

        if mode == 'dna' or mode == 'both':
            self.dna_proj = ModalityProjector(dna_dim, proj_dim, dropout, feature_mode)
            
        if mode == 'prot' or mode == 'both':
            self.prot_proj = ModalityProjector(prot_dim, proj_dim, dropout)

        if mode == 'both':
            if fusion_method == 'cross_attn':
                self.cross_attn_dna = CrossAttentionBlock(proj_dim, dropout=dropout)
                self.cross_attn_prot = CrossAttentionBlock(proj_dim, dropout=dropout)
            
            if self.use_gating:
                self.gater = GatingMechanism(proj_dim)
                in_dim = proj_dim # Gating combined 2 vector into 1 vector with the same size

            else: # concat
                in_dim = proj_dim * 2
                
        else: # dna or prot
            in_dim = proj_dim

        layers = []
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, dna_ref, dna_alt, prot_ref, prot_alt, return_gates=False):
        # project each modality
        dna_z = self.dna_proj(dna_ref, dna_alt) if self.mode in ['dna', 'both'] else None
        prot_z = self.prot_proj(prot_ref, prot_alt) if self.mode in ['prot', 'both'] else None

        # fusion logic
        if self.mode == 'dna':
            fused = dna_z
        elif self.mode == 'prot':
            fused = prot_z
        else:  # mode == 'both'
            if self.fusion_method == 'cross_attn':
                # DNA "look" at PROT and vice versa
                dna_f = self.cross_attn_dna(dna_z, prot_z)
                prot_f = self.cross_attn_prot(prot_z, dna_z)
            else:  
                dna_f, prot_f = dna_z, prot_z
            
            if self.use_gating:
                # Calc gate g
                combined = torch.cat([dna_f, prot_f], dim=-1)
                g = self.gater.gate(combined) # g: [batch, proj_dim]
                fused = g * dna_f + (1 - g) * prot_f
                
                logits = self.classifier(fused).squeeze(-1)
                
                if return_gates:
                    return logits, g.mean(dim=-1)
                return logits
            
            else: # concat
                fused = torch.cat([dna_f, prot_f], dim=-1)
                return self.classifier(fused).squeeze(-1)
            
        return self.classifier(fused).squeeze(-1)

