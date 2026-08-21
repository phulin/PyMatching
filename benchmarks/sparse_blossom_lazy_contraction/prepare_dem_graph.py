#!/usr/bin/env python3
import math
import re
import struct
import sys
from collections import OrderedDict
from pathlib import Path

REPEAT_RE = re.compile(r'^repeat\s+(\d+)\s*\{$')
ERROR_RE = re.compile(r'^error\(([^)]+)\)\s*(.*)$')
D_RE = re.compile(r'^D(\d+)$')
L_RE = re.compile(r'^L(\d+)$')

class Error:
    def __init__(self,p,targets): self.p=p; self.targets=targets
class Shift:
    def __init__(self,amount): self.amount=amount
class Detector:
    def __init__(self,targets): self.targets=targets
class Repeat:
    def __init__(self,n,body): self.n=n; self.body=body

def parse_block(lines, i=0, nested=False):
    out=[]
    while i < len(lines):
        line=lines[i].strip()
        i+=1
        if not line or line.startswith('#'): continue
        if line=='}':
            if not nested: raise ValueError('unmatched }')
            return out,i
        m=REPEAT_RE.match(line)
        if m:
            body,i=parse_block(lines,i,True)
            out.append(Repeat(int(m.group(1)),body));continue
        m=ERROR_RE.match(line)
        if m:
            out.append(Error(float(m.group(1)),m.group(2).split()));continue
        if line.startswith('shift_detectors'):
            suffix=line.split(')',1)[1].strip() if ')' in line else line[len('shift_detectors'):].strip()
            amount=int(suffix.split()[0])
            out.append(Shift(amount));continue
        if line.startswith('detector'):
            suffix=line.split(')',1)[1].strip() if ')' in line else line[len('detector'):].strip()
            out.append(Detector(suffix.split()));continue
        if line.startswith('logical_observable'):
            continue
        raise ValueError(f'unknown line: {line}')
    if nested: raise ValueError('unterminated repeat')
    return out,i

def execute(nodes, state, edges):
    for node in nodes:
        if isinstance(node, Repeat):
            for _ in range(node.n): execute(node.body,state,edges)
        elif isinstance(node,Shift):
            state['offset'] += node.amount
        elif isinstance(node,Detector):
            for t in node.targets:
                m=D_RE.match(t)
                if m: state['max_det']=max(state['max_det'],state['offset']+int(m.group(1)))
        elif isinstance(node,Error):
            dets=[];obs=0
            def flush():
                nonlocal dets,obs
                if len(dets)==1:
                    key=(dets[0],-1)
                elif len(dets)==2:
                    key=tuple(sorted(dets))
                else:
                    dets=[];obs=0;return
                state['max_det']=max(state['max_det'],*dets)
                if key in edges:
                    oldp,oldobs=edges[key]
                    edges[key]=(node.p*(1-oldp)+oldp*(1-node.p),oldobs)
                else:
                    edges[key]=(node.p,obs)
                dets=[];obs=0
            for t in node.targets:
                if t=='^': flush();continue
                m=D_RE.match(t)
                if m: dets.append(state['offset']+int(m.group(1)));continue
                m=L_RE.match(t)
                if m: obs ^= 1<<int(m.group(1));continue
                raise ValueError(f'bad target {t}')
            flush()
        else: raise TypeError(node)

def main():
    if len(sys.argv)!=3: raise SystemExit('usage: prepare_dem_graph.py DEM OUT')
    dem=Path(sys.argv[1]);out=Path(sys.argv[2])
    ast,_=parse_block(dem.read_text().splitlines())
    edges=OrderedDict();state={'offset':0,'max_det':-1}
    execute(ast,state,edges)
    weights=[]
    for (u,v),(p,obs) in edges.items():
        if not 0 < p < 1: raise ValueError((u,v,p))
        weights.append(math.log((1-p)/p))
    max_w=max(abs(w) for w in weights)
    normalizer=((1<<24)-1)/max_w
    rows=[]
    for ((u,v),(p,obs)),w in zip(edges.items(),weights):
        q=math.floor(w*normalizer + 0.5)*2
        if not -(1<<31) <= q < (1<<31): raise OverflowError(q)
        rows.append((u,v,q,obs))
    with out.open('wb') as f:
        f.write(struct.pack('<8sQQQd',b'PMDEM1\0\0',state['max_det']+1,1,len(rows),normalizer*2))
        for u,v,q,obs in rows:
            f.write(struct.pack('<QqiQ',u,v,q,obs))
    print(f'detectors={state["max_det"]+1} edges={len(rows)} max_weight={max_w:.17g} graph_normalizer={normalizer*2:.17g} bytes={out.stat().st_size}')

if __name__=='__main__': main()
