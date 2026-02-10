import numpy as np

class MaxEntScorer:
    def __init__(self):
        self.donor_pssm = {
            'A': [0.28, 0.59, 0.08, 0.00, 0.00, 0.46, 0.71, 0.07, 0.15],
            'C': [0.38, 0.13, 0.04, 0.00, 0.00, 0.03, 0.08, 0.05, 0.19],
            'G': [0.18, 0.14, 0.81, 1.00, 0.00, 0.45, 0.12, 0.81, 0.20],
            'T': [0.16, 0.14, 0.07, 0.00, 1.00, 0.06, 0.09, 0.07, 0.46]
        }
        self.bg = {'A': 0.27, 'C': 0.23, 'G': 0.23, 'T': 0.27}

    def score5(self, seq):
        if not isinstance(seq, str) or len(seq) != 9 or 'N' in seq.upper(): return -20.0
        seq = seq.upper().replace('U', 'T')
        score = 0.0
        try:
            for i, base in enumerate(seq):
                p_base = self.donor_pssm.get(base, [0]*9)[i]
                if p_base <= 0: return -20.0
                score += np.log2(p_base / self.bg.get(base, 0.25))
            return score
        except: return -20.0

    def score3(self, seq):
        if not isinstance(seq, str) or len(seq) != 23 or 'N' in seq.upper(): return -25.0
        seq = seq.upper().replace('U', 'T')
        if seq[18:20] != "AG": return -30.0 
        
        ppt_region = seq[0:18]
        t_count = ppt_region.count('T')
        c_count = ppt_region.count('C')
        g_count = ppt_region.count('G')
        a_count = ppt_region.count('A')
        
        # Giữ nguyên logic PPT của bạn
        ppt_score = (t_count * 2.5 + c_count * 1.0) - ((g_count + a_count) * 4.0)
        return ppt_score