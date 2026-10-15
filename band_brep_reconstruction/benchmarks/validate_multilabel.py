"""Persistent full-STEP smoke validation of the multi-label pipeline (analytic fields)."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bandbrep.demo_fields import box,cylinder,sphere_with_bore,split_sphere,hollow_sphere
from bandbrep.config import Config
from bandbrep.reconstruct import reconstruct
from bandbrep.schemas import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result={'field_source':'analytic demo scalar fields; no truth instance IDs supplied','results':{}}
    jobs=[('box',box,(6,12,8)),('cylinder',cylinder,(3,2,0)),
          ('sphere_bore',sphere_with_bore,(2,2,0)),('split_sphere',split_sphere,(2,1,0)),('hollow_sphere',hollow_sphere,(2,0,0))]
    for name,make,expected in jobs:
        start=time.monotonic();target=out/name
        try:
            data=reconstruct(make(),Config(resolution=65,face_samples=400,save_queries=False),target)
            graph=data['graph'];actual=(graph.face_count,len(graph.edges),len(graph.vertices))
            result['results'][name]={'status':data['report']['status'],'expected_counts':expected,'actual_counts':actual,
                'counts_match':actual==expected,'step':str(target/'reconstruction.step')}
        except Exception as exc:
            result['results'][name]={'status':'failed','code':getattr(exc,'code',type(exc).__name__),'message':str(exc)}
        result['results'][name]['seconds']=time.monotonic()-start
        write_json(out/'report.json',result)
    rows=['# 多标签方案完整重建验证','',
          '解析三场输入，未向重建传入真实拓扑 ID。计数仅为额外诊断，solid_validated 另包含 CAD/STEP 读回检查。','',
          '| 模型 | 状态 | 面/边/顶点 | 耗时（秒） |','|---|---|---|---:|']
    for name,r in result['results'].items():
        rows.append(f"| {name} | {r.get('code',r['status'])} | {r.get('actual_counts','')} | {r['seconds']:.2f} |")
    rows+=['','STEP 文件位于各模型子目录中的 `reconstruction.step`。这组理想场验证不等同于 ABC 批量或扩散生成场的成功率评估。','']
    (out/'REPORT.md').write_text('\n'.join(rows))
    return 0 if all(r['status']=='solid_validated' and r['counts_match'] for r in result['results'].values()) else 1

if __name__=='__main__':sys.exit(main())
