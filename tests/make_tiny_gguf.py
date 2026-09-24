#!/usr/bin/env python3
import argparse, struct
from pathlib import Path

ALIGN=32

def pad(buf, n=ALIGN):
    r=len(buf)%n
    if r: buf.extend(b'\0'*(n-r))

def s(x: str):
    b=x.encode('utf-8'); return struct.pack('<Q',len(b))+b

def kv_string(k,v): return s(k)+struct.pack('<I',8)+s(v)
def kv_u32(k,v): return s(k)+struct.pack('<II',4,v)
def kv_f32(k,v): return s(k)+struct.pack('<I',6)+struct.pack('<f',v)
def kv_bool(k,v): return s(k)+struct.pack('<I',7)+struct.pack('<B',1 if v else 0)
def kv_strings(k,vals):
    out=s(k)+struct.pack('<IIQ',9,8,len(vals))
    for v in vals: out+=s(v)
    return out
def kv_i32s(k,vals):
    out=s(k)+struct.pack('<IIQ',9,5,len(vals))
    for v in vals: out+=struct.pack('<i',v)
    return out

def f32(values): return b''.join(struct.pack('<f',x) for x in values)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output')
    ap.add_argument('--context', type=int, default=64)
    args=ap.parse_args()
    if args.context < 2 or args.context > 1048576:
        ap.error('--context must be in [2,1048576]')
    out_path=Path(args.output)

    vocab=['u','s','e','r','a','i','t','n','Ċ','<|im_start|>','<|im_end|>','b','c','d']
    token_types=[1]*len(vocab)
    token_types[9]=3; token_types[10]=3
    meta=[
        kv_string('general.architecture','qwen2'),
        kv_string('general.name','AIR tiny integration model'),
        kv_u32('general.alignment',ALIGN),
        kv_u32('qwen2.block_count',1),
        kv_u32('qwen2.context_length',args.context),
        kv_u32('qwen2.embedding_length',4),
        kv_u32('qwen2.feed_forward_length',6),
        kv_u32('qwen2.attention.head_count',2),
        kv_u32('qwen2.attention.head_count_kv',1),
        kv_u32('qwen2.rope.dimension_count',2),
        kv_f32('qwen2.rope.freq_base',10000.0),
        kv_f32('qwen2.attention.layer_norm_rms_epsilon',1e-5),
        kv_string('tokenizer.ggml.model','gpt2'),
        kv_string('tokenizer.ggml.pre','gpt2'),
        kv_strings('tokenizer.ggml.tokens',vocab),
        kv_strings('tokenizer.ggml.merges',[]),
        kv_i32s('tokenizer.ggml.token_type',token_types),
        kv_u32('tokenizer.ggml.eos_token_id',10),
        kv_bool('tokenizer.ggml.add_bos_token',False),
        kv_bool('tokenizer.ggml.add_eos_token',False),
        kv_string('tokenizer.chat_template','qwen2-chatml'),
    ]

    emb=4; vf=len(vocab); ffn=6
    ident=[]
    for row in range(vf):
        for col in range(emb):
            ident.append(1.0 if row==col and row<emb else 0.0)
    ones=[1.0]*emb
    tensors=[
        ('token_embd.weight',[emb,vf],f32(ident)),
        ('output_norm.weight',[emb],f32(ones)),
        ('blk.0.attn_norm.weight',[emb],f32(ones)),
        ('blk.0.attn_q.weight',[emb,emb],f32([0.0]*(emb*emb))),
        ('blk.0.attn_k.weight',[emb,2],f32([0.0]*(emb*2))),
        ('blk.0.attn_v.weight',[emb,2],f32([0.0]*(emb*2))),
        ('blk.0.attn_output.weight',[emb,emb],f32([0.0]*(emb*emb))),
        ('blk.0.ffn_norm.weight',[emb],f32(ones)),
        ('blk.0.ffn_gate.weight',[emb,ffn],f32([0.0]*(emb*ffn))),
        ('blk.0.ffn_up.weight',[emb,ffn],f32([0.0]*(emb*ffn))),
        ('blk.0.ffn_down.weight',[ffn,emb],f32([0.0]*(ffn*emb))),
    ]

    data=bytearray(); descriptors=[]
    for name,dims,raw in tensors:
        pad(data)
        offset=len(data)
        data.extend(raw)
        desc=s(name)+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',0,offset)
        descriptors.append(desc)

    header=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta)))
    for item in meta: header.extend(item)
    for desc in descriptors: header.extend(desc)
    pad(header)
    header.extend(data)
    out_path.write_bytes(header)
    print(out_path)

if __name__=='__main__': main()
