#!/usr/bin/env python3
import argparse
import random
import struct
from pathlib import Path

ALIGN=32

def pad(buf, n=ALIGN):
    r=len(buf)%n
    if r: buf.extend(b'\0'*(n-r))

def s(x):
    b=x.encode(); return struct.pack('<Q',len(b))+b

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

def f32(vals): return b''.join(struct.pack('<f',x) for x in vals)
def h(x): return struct.pack('<e',x)

def q4_0_blocks(count,rng):
    out=bytearray()
    for _ in range(count):
        out += h(0.01)
        out += bytes(rng.randrange(256) for _ in range(16))
    return bytes(out)

def q5_0_blocks(count,rng):
    out=bytearray()
    for _ in range(count):
        out += h(0.008)
        out += bytes(rng.randrange(256) for _ in range(4))
        out += bytes(rng.randrange(256) for _ in range(16))
    return bytes(out)

def q8_0_blocks(count,rng):
    out=bytearray()
    for _ in range(count):
        out += h(0.002)
        out += bytes(rng.randrange(256) for _ in range(32))
    return bytes(out)

def q4_k_blocks(count,rng):
    out=bytearray()
    for _ in range(count):
        out += h(0.002) + h(0.0)
        out += bytes(rng.randrange(256) for _ in range(12))
        out += bytes(rng.randrange(256) for _ in range(128))
    return bytes(out)

def q6_k_blocks(count,rng):
    out=bytearray()
    for _ in range(count):
        out += bytes(rng.randrange(256) for _ in range(128))
        out += bytes(rng.randrange(256) for _ in range(64))
        out += bytes((rng.randrange(-2,3) & 0xff) for _ in range(16))
        out += h(0.002)
    return bytes(out)

TYPE={"f32":0,"q4_0":2,"q5_0":6,"q8_0":8,"q4_k":12,"q6_k":14}
BLOCK={"q4_0":32,"q5_0":32,"q8_0":32,"q4_k":256,"q6_k":256}
ENC={"q4_0":q4_0_blocks,"q5_0":q5_0_blocks,"q8_0":q8_0_blocks,"q4_k":q4_k_blocks,"q6_k":q6_k_blocks}

def quant(name,dims,qtype,rng):
    elements=1
    for d in dims: elements*=d
    if len(dims)!=2 or dims[0] % BLOCK[qtype] != 0:
        raise ValueError((name,dims,qtype))
    blocks=elements//BLOCK[qtype]
    return (name,dims,TYPE[qtype],ENC[qtype](blocks,rng))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output')
    args=ap.parse_args()
    rng=random.Random(0xA174D1FF)
    emb=256; ffn=256; vocab=256; heads=4; kv_heads=1; head_dim=64
    vocab_tokens=[f't{i}' for i in range(vocab)]
    meta=[
        kv_string('general.architecture','qwen2'),
        kv_string('general.name','AIR quantized differential fixture'),
        kv_u32('general.alignment',ALIGN),
        kv_u32('qwen2.block_count',1), kv_u32('qwen2.context_length',64),
        kv_u32('qwen2.embedding_length',emb), kv_u32('qwen2.feed_forward_length',ffn),
        kv_u32('qwen2.attention.head_count',heads), kv_u32('qwen2.attention.head_count_kv',kv_heads),
        kv_u32('qwen2.rope.dimension_count',head_dim), kv_f32('qwen2.rope.freq_base',10000.0),
        kv_f32('qwen2.attention.layer_norm_rms_epsilon',1e-5),
        kv_string('tokenizer.ggml.model','gpt2'), kv_string('tokenizer.ggml.pre','gpt2'),
        kv_strings('tokenizer.ggml.tokens',vocab_tokens), kv_strings('tokenizer.ggml.merges',[]),
        kv_i32s('tokenizer.ggml.token_type',[1]*vocab), kv_u32('tokenizer.ggml.eos_token_id',255),
        kv_bool('tokenizer.ggml.add_bos_token',False), kv_bool('tokenizer.ggml.add_eos_token',False),
    ]
    tensors=[]
    tensors.append(quant('token_embd.weight',[emb,vocab],'q4_0',rng))
    tensors.append(quant('output.weight',[emb,vocab],'q8_0',rng))
    tensors.append(('output_norm.weight',[emb],TYPE['f32'],f32([1.0]*emb)))
    tensors.append(('blk.0.attn_norm.weight',[emb],TYPE['f32'],f32([1.0]*emb)))
    tensors.append(quant('blk.0.attn_q.weight',[emb,emb],'q5_0',rng))
    tensors.append(quant('blk.0.attn_k.weight',[emb,kv_heads*head_dim],'q4_k',rng))
    tensors.append(quant('blk.0.attn_v.weight',[emb,kv_heads*head_dim],'q6_k',rng))
    tensors.append(('blk.0.attn_q.bias',[emb],TYPE['f32'],f32([0.0]*emb)))
    tensors.append(('blk.0.attn_k.bias',[kv_heads*head_dim],TYPE['f32'],f32([0.0]*(kv_heads*head_dim))))
    tensors.append(('blk.0.attn_v.bias',[kv_heads*head_dim],TYPE['f32'],f32([0.0]*(kv_heads*head_dim))))
    tensors.append(quant('blk.0.attn_output.weight',[emb,emb],'q4_k',rng))
    tensors.append(('blk.0.ffn_norm.weight',[emb],TYPE['f32'],f32([1.0]*emb)))
    tensors.append(quant('blk.0.ffn_gate.weight',[emb,ffn],'q5_0',rng))
    tensors.append(quant('blk.0.ffn_up.weight',[emb,ffn],'q6_k',rng))
    tensors.append(quant('blk.0.ffn_down.weight',[ffn,emb],'q4_0',rng))

    data=bytearray(); desc=[]
    for name,dims,t,raw in tensors:
        pad(data); off=len(data); data.extend(raw)
        desc.append(s(name)+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',t,off))
    header=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta)))
    for m in meta: header.extend(m)
    for d in desc: header.extend(d)
    pad(header); header.extend(data)
    Path(args.output).write_bytes(header)
    print(args.output)

if __name__=='__main__': main()
