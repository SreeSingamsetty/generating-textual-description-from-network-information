# -*- coding: utf-8 -*-
#for adjusting tab between the node name and embeddings in the generated embeddings file.
import regex as re
#enter previously generated embedding file name
with open("33_200d_20wl10nm.emb","r") as f:
    # enter desired new embedding file name
    with open("33_200d_20wl10nm_online.emb","w") as w:
        next(f)
        for line in f:
            line = re.sub(r'\t', ' ', line)
            line=re.sub(r'([^0-9_:-]|\')\s{1,3}(-*[0-9][0-9][.])',r'\1\t\2',line,flags=re.UNICODE)
            line = re.sub(r'([^0-9_:-]|\')\s{1,3}(-*[0-9][.])', r'\1\t\2', line, flags=re.UNICODE)
            line=re.sub(r'(^\d{4})\s{1,3}(-*[0-9][.])',r'\1\t\2',line)
            line=re.sub(r'( \'27)\s{1,3}(-*[0-9][.])',r'\1\t\2',line)
            line = re.sub(r'(-| \d{1,4}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line = re.sub(r'(\d{2,4}-\d{2,4}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line=re.sub(r'(_\d{1}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line=re.sub(r'([n_-]\d{4}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line=re.sub(r'([:%]\d{2,4}) {1,3}(-*[0-9][.])',r'\1\t\2',line)
            line = re.sub(r'([a-z]\.\d{4}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line = re.sub(r'([a-z]\d{1}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            line=re.sub(r'([5][0]{4}) {1,3}(-*[0-9][.])', r'\1\t\2', line)
            w.write(line)
