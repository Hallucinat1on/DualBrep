"""Export audited, offline Plotly viewers for the eight 11-09 reconstruction failures.

Run in the OCC/Plotly environment. Each case uses its historical reconstruction
modules in a separate process; this exporter never updates experiment records.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

CASES = {
    '00000688': ('面碎片与两条边粘连', '9个预测邻面中多个属于同一真实面，边支撑同时包含真实边58和62。'),
    '00004610': ('闭环与开边粘连', '真实闭合环被另一条开边的支撑和端点污染；数量正确不代表实例关联正确。'),
    '00005094': ('闭合边被当作开曲线', '真实边71没有端点，预测却关联两个顶点。诊断重放显示开路径遗漏闭环支撑。'),
    '00008133': ('缺面缺边与实例粘连', '预测少4个面、16条边、8个顶点；失败实例又混入两条真实边。'),
    '00008804': ('共同顶点附近漏分一条边', '真实边62和47共用顶点42，预测把两条边的支撑及3个端点归入同一个实例。'),
    '00004627': ('曲面UV与硬边界约束冲突', '曲线和图已完成；PCA初始化UV下的张量样条无法满足固定边界，诊断曲面不是成功重建产物。'),
    '00000602': ('球面接缝切开共享圆边', '接缝修复前14条边、后16条边；完整共享圆被切开并替换。两侧显示原诊断保存的BRep。'),
    '00008222': ('环面边参数一致性失败', '拒绝面28有两条边SameParameter为假。采样偏差小于容差不等于连续参数一致性已建立。'),
}
PALETTE = ['#2563eb', '#ea580c', '#059669', '#9333ea', '#db2777', '#0891b2', '#ca8a04', '#475569', '#65a30d', '#7c3aed', '#dc2626', '#0d9488']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def color(index):
    return PALETTE[int(index) % len(PALETTE)]


def stratified_indices(labels, budget, seed=0):
    """Keep every represented label, including rare contamination; stable output."""
    labels = np.asarray(labels)
    if len(labels) <= budget:
        return np.arange(len(labels))
    groups, counts = np.unique(labels, return_counts=True)
    if budget < len(groups):
        raise ValueError('Display budget is smaller than the number of labels')
    rng = np.random.default_rng(seed)
    allocations = np.ones(len(groups), dtype=int)
    remaining = budget - len(groups)
    weights = counts - 1
    shares = remaining * weights / max(weights.sum(), 1)
    allocations += np.floor(shares).astype(int)
    for j in np.argsort(-(shares - np.floor(shares)), kind='stable')[:budget - allocations.sum()]:
        allocations[j] += 1
    selected = [rng.choice(np.flatnonzero(labels == g), int(n), replace=False) for g, n in zip(groups, allocations)]
    return np.sort(np.concatenate(selected))


def extent(points, margin=.12):
    p = np.asarray(points, float)
    lo, hi = p.min(0), p.max(0)
    size = max(float(np.max(hi - lo)), .015)
    center = (lo + hi) * .5
    return np.vstack((center - size * (.5 + margin), center + size * (.5 + margin)))


def scatter(points, name, shade, layer, ids=None, size=3, text=None, values=None):
    p = np.asarray(points, float).reshape(-1, 3)
    trace = dict(type='scatter3d', mode='markers', x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                 name=name, marker=dict(size=size, color=shade, opacity=.9), meta=dict(layer=layer),
                 hovertemplate=name + '<br>%{text}<br>(%{x:.5f}, %{y:.5f}, %{z:.5f})<extra></extra>')
    trace['text'] = text if text is not None else ([f'node={int(i)}' for i in ids] if ids is not None else [''] * len(p))
    if values is not None:
        trace['customdata'] = np.asarray(values).tolist()
        trace['hovertemplate'] = trace['hovertemplate'].replace('<extra></extra>', '<br>dE=%{customdata[0]:.6g}, dV=%{customdata[1]:.6g}<extra></extra>')
    return trace


def line(points, name, shade, layer='prediction', width=5, dash=None):
    p = np.asarray(points, float).reshape(-1, 3)
    style = dict(color=shade, width=width)
    if dash:
        style['dash'] = dash
    return dict(type='scatter3d', mode='lines', x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                name=name, line=style, meta=dict(layer=layer), hovertemplate=name + '<extra></extra>')


def connectors(a, b, name, shade='#dc2626', scale=1.):
    a, b = np.asarray(a), np.asarray(b)
    q = a + scale * (b - a)
    points = np.full((len(a), 3, 3), np.nan)
    points[:, 0] = a
    points[:, 1] = q
    # JSON has no NaN: Plotly uses null as a line break.
    result = line(np.zeros((0, 3)), name, shade, 'displacement', 2)
    for k, axis in enumerate(('x', 'y', 'z')):
        result[axis] = [None if not np.isfinite(v) else float(v) for v in points[:, :, k].ravel()]
    result['meta'].update(scale_vectors=True, starts=a.tolist(), ends=b.tolist())
    return result


def mesh_trace(vertices, triangles, name, shade, layer='faces', opacity=.6):
    v, f = np.asarray(vertices), np.asarray(triangles, int)
    if not len(f):
        return None
    ids, inverse = np.unique(f, return_inverse=True)
    p = v[ids]
    i = inverse.reshape(-1, 3)
    return dict(type='mesh3d', x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                i=i[:, 0].tolist(), j=i[:, 1].tolist(), k=i[:, 2].tolist(), name=name,
                color=shade, opacity=opacity, flatshading=True, meta=dict(layer=layer),
                hovertemplate=name + '<extra></extra>', showscale=False)


def triangle_subset(vertices, triangles, labels, bounds, budget):
    centers = np.asarray(vertices)[np.asarray(triangles)].mean(1)
    active = np.all((centers >= bounds[0]) & (centers <= bounds[1]), axis=1)
    ids = np.flatnonzero(active)
    if len(ids) > budget:
        ids = ids[stratified_indices(np.asarray(labels)[ids], budget)]
    return ids


def occ_mesh(shape, deflection=.001):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    BRepMesh_IncrementalMesh(shape, deflection, False, .18, False).Perform()
    vertices, triangles, offset = [], [], 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = topods.Face(ex.Current()); location = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, location)
        if tri is not None:
            v = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(j).Transformed(location.Transformation()) for j in range(1, tri.NbNodes() + 1))])
            f = np.array([[tri.Triangle(j).Value(k) - 1 for k in (1, 2, 3)] for j in range(1, tri.NbTriangles() + 1)])
            vertices.append(v); triangles.append(f + offset); offset += len(v)
        ex.Next()
    if not vertices:
        return np.empty((0, 3)), np.empty((0, 3), int)
    return np.vstack(vertices), np.vstack(triangles)


def sample_edge(edge, count=257):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    ad = BRepAdaptor_Curve(edge)
    return np.array([[p.X(), p.Y(), p.Z()] for p in (ad.Value(float(t)) for t in np.linspace(ad.FirstParameter(), ad.LastParameter(), count))])


def association_plot(edge_id, faces, vertices, score):
    nodes = [('E' + str(edge_id), 0., 0., '#dc2626')]
    for j, f in enumerate(faces):
        nodes.append((f'F{f} → GT F{score["face_map"][f]}', -1., j - (len(faces) - 1) / 2, color(f)))
    for j, v in enumerate(vertices):
        nodes.append((f'V{v} → GT V{score["vertex_map"][v]}', 1., j - (len(vertices) - 1) / 2, '#111827'))
    xx, yy = [], []
    for _, x, y, _ in nodes[1:]:
        xx += [0, x, None]; yy += [0, y, None]
    return dict(title='失败实例关联：点击节点查看；普通边通常应有两个邻面', data=[
        dict(type='scatter', mode='lines', x=xx, y=yy, line=dict(color='#cbd5e1'), hoverinfo='skip', showlegend=False),
        dict(type='scatter', mode='markers+text', x=[n[1] for n in nodes], y=[n[2] for n in nodes],
             text=[n[0] for n in nodes], textposition='top center', marker=dict(size=15, color=[n[3] for n in nodes]), showlegend=False)],
        layout=dict(xaxis=dict(visible=False, range=[-1.8, 1.8]), yaxis=dict(visible=False), height=max(330, 42 * len(faces))))


def order_replay(points, ends, provider, cfg):
    """Capture the actual frozen order_general locals on its exception, no rewrite."""
    from bandbrep.curve_distance_fitting import representative_indices
    from bandbrep.edge_geometry import project_support, order_general
    selected = representative_indices(points, cfg.curve_init_samples, cfg.seed + 2)
    projected = project_support(points[selected], provider, max(cfg.tau_edge * 3, cfg.surface_tolerance * 4), cfg, max_iterations=8)
    captured = {}
    def tracer(frame, event, arg):
        if frame.f_code is order_general.__code__ and event in ('exception', 'return'):
            for k in ('p', 'q', 'worst'):
                if k in frame.f_locals:
                    value = frame.f_locals[k]
                    captured[k] = value.copy() if isinstance(value, np.ndarray) else value
        return tracer
    previous = sys.gettrace()
    try:
        sys.settrace(tracer)
        order_general(projected, ends, cfg)
    except Exception as exc:
        captured['error'] = dict(code=getattr(exc, 'code', type(exc).__name__), details=getattr(exc, 'details', {}))
    finally:
        sys.settrace(previous)
    if 'q' not in captured or 'p' not in captured:
        raise RuntimeError('Frozen ordering replay did not reach coverage check')
    # Exact distances to every segment, rather than the candidate upper bound used by the guard.
    q, p = captured['q'], captured['p']; delta = q[1:] - q[:-1]
    rel = p[:, None] - q[None, :-1]
    t = np.clip(np.einsum('nsi,si->ns', rel, delta) / np.maximum((delta * delta).sum(1), 1e-20), 0, 1)
    closest = q[None, :-1] + t[:, :, None] * delta
    distance = np.linalg.norm(p[:, None] - closest, axis=2)
    best = distance.argmin(1)
    captured.update(coverage=distance[np.arange(len(p)), best], closest=closest[np.arange(len(p)), best])
    return captured


def surface_probe(folder, mesh, labels, targets, cfg, distance_cfg):
    from bandbrep.cache import load_graph, load_records
    from bandbrep.surface_fitting import sample_face_regions, tensor_basis
    from veudf.preprocess.distance import EdgeOracle
    graph = load_graph(folder / 'cad_graph.json'); curves = load_records(folder / 'curves.json', 'curve')
    p, _ = sample_face_regions(mesh, labels, cfg)[0]
    safe = EdgeOracle(targets.edges, distance_cfg).query(p)['distance'] > cfg.tau_edge * .5
    if safe.sum() >= 20:
        p = p[safe]
    related = [e.id for e in graph.edges if 0 in e.faces]
    boundary = np.vstack([curves[e].evaluate(np.linspace(0, 1, 65)) for e in related])
    mean = p.mean(0); _, _, vt = np.linalg.svd(p - mean, full_matrices=False); x, y = vt[:2]
    uv = np.column_stack(((p - mean) @ x, (p - mean) @ y)); bu = np.column_stack(((boundary - mean) @ x, (boundary - mean) @ y))
    lo, hi = np.minimum(uv.min(0), bu.min(0)), np.maximum(uv.max(0), bu.max(0)); span = np.maximum(hi - lo, 1e-8)
    uv = (uv - lo) / span; bu = (bu - lo) / span
    counts, residuals, fit = [], [], None
    for n in range(cfg.spline_controls, cfg.max_spline_controls + 1, 2):
        knots = np.r_[np.zeros(4), np.linspace(0, 1, n - 2)[1:-1], np.ones(4)]
        B = tensor_basis(bu, knots); poles = np.linalg.lstsq(B, boundary, rcond=1e-11)[0]
        fit = B @ poles
        counts.append(n); residuals.append(float(np.abs(fit - boundary).max()))
    u, v = np.meshgrid(np.linspace(0, 1, 35), np.linspace(0, 1, 35)); grid = tensor_basis(np.c_[u.ravel(), v.ravel()], knots) @ poles
    faces = []
    for a in range(34):
        for b in range(34):
            j = a * 35 + b; faces.extend([[j, j + 1, j + 35], [j + 1, j + 36, j + 35]])
    return dict(points=p, boundary=boundary, uv=uv, boundary_uv=bu, fit=fit, controls=counts, residuals=residuals, grid=grid, faces=np.array(faces))


def pcurve_probe(path):
    from OCC.Core.BRep import BRep_Tool, BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Shape, topods
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface, BRepAdaptor_Curve2d
    shape = TopoDS_Shape()
    if not breptools.Read(shape, str(path), BRep_Builder()):
        raise RuntimeError('Cannot read rejected face')
    face = topods.Face(TopExp_Explorer(shape, TopAbs_FACE).Current()); surface = BRepAdaptor_Surface(face)
    ex = TopExp_Explorer(face, TopAbs_EDGE); rows = []
    while ex.More():
        edge = topods.Edge(ex.Current()); curve = BRepAdaptor_Curve(edge); pc = BRepAdaptor_Curve2d(edge, face)
        t = np.linspace(pc.FirstParameter(), pc.LastParameter(), 257)
        a, b, uv = [], [], []
        for ti in t:
            p = curve.Value(float(ti)); q = pc.Value(float(ti)); r = surface.Value(q.X(), q.Y())
            a.append([p.X(), p.Y(), p.Z()]); b.append([r.X(), r.Y(), r.Z()]); uv.append([q.X(), q.Y()])
        a, b = np.array(a), np.array(b)
        rows.append(dict(index=len(rows), a=a, b=b, uv=np.array(uv), t=t,
                         error=np.linalg.norm(a - b, axis=1), same_parameter=bool(BRep_Tool.SameParameter(edge)),
                         tolerance=float(BRep_Tool.Tolerance(edge))))
        ex.Next()
    return shape, rows


def worker(args):
    started = time.monotonic(); batch = args.batch.resolve(); out = args.output.resolve(); sid = args.case
    folder = batch / 'runs' / sid; identity = read(folder / 'identity.json'); snapshot = folder / 'source_snapshot'
    for name, expected in identity['inputs'].items():
        if digest(name) != expected:
            raise RuntimeError('Historical input changed: ' + name)
    for name, expected in identity['code'].items():
        if digest(snapshot / name) != expected:
            raise RuntimeError('Historical code changed: ' + name)
    sys.path[:0] = [str(snapshot / 'band_brep_reconstruction'), str(snapshot / 'veudf/src'), str(snapshot / 'band_brep_reconstruction/benchmarks')]
    from bandbrep.cache import load_records
    from bandbrep.config import Config
    from bandbrep.schemas import SurfaceMesh
    from veudf.preprocess.config import Config as DistanceConfig
    from veudf.preprocess.cad import read_normalized, collect_targets
    from veudf.preprocess.distance import EdgeOracle
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Shape
    policy = read(folder / 'policy.json'); cfg = Config(**policy['config']).validate(); dcfg = DistanceConfig().validate()
    original = Path(policy['input_group']) / 'runs' / sid / 'reconstruction'
    result = read(batch / 'records' / (sid + '.json')); old = read(Path(policy['input_group']) / 'records' / (sid + '.json')); score = old['score']
    incidence = read(original / 'incidence.json'); vertices = read(original / 'vertices.json')
    shape, normalization = read_normalized(policy['row']['step'], dcfg); targets = collect_targets(shape, dcfg)
    audit = read(Path(policy['input_group']) / 'runs' / sid / 'source_audit.json')
    np.testing.assert_allclose(normalization['world_to_normalized'], audit['normalization']['world_to_normalized'], atol=1e-10, rtol=0)
    if dict(faces=len(targets.faces), edges=len(targets.edges), vertices=len(targets.vertices)) != audit['target_counts']:
        raise RuntimeError('CAD target counts differ from frozen run')
    with np.load(original / 'input_mesh.npz') as d:
        bgv, bgf = d['vertices'], d['triangles']
    with np.load(original / 'working_mesh.npz') as d:
        mv, mf = d['vertices'], d['triangles']
    with np.load(original / 'topology_labels.npz') as d:
        face_labels, vertex_labels = d['face_labels'], d['vertex_labels']
    with np.load(original / 'bands.npz') as d:
        points, values = d['points'], d['values']
    progress = read(folder / 'curve_recovery_progress.json')
    edge_id = progress.get('active_edge') if result['failed_stage'] == 'curves' else None
    face_id = result['error']['details'].get('face')
    selected_edges = [edge_id] if edge_id is not None else [e['id'] for e in incidence['edges'] if face_id in e['faces']]
    selected_faces = incidence['edges'][edge_id]['faces'] if edge_id is not None else [face_id]
    vertex_ids = sorted({v for e in selected_edges for v in incidence['edges'][e]['vertices']})
    node_ids = np.unique(np.concatenate([np.asarray(incidence['edges'][e]['support_nodes'], int) for e in selected_edges]))
    focus_points = points[node_ids]
    if vertex_ids:
        focus_points = np.vstack((focus_points, [vertices[v]['position'] for v in vertex_ids]))
    focus = extent(focus_points); global_bounds = extent(bgv)
    bg_ids = stratified_indices(np.zeros(len(bgf), int), min(args.max_triangles, 30000))
    background = mesh_trace(bgv, bgf[bg_ids], '全局背景（原始网格抽样）', '#94a3b8', 'background', .13)
    predicted, truth, diag, notes = [], [], [], []
    if background:
        predicted.append(background); truth.append(background)
    roi_ids = triangle_subset(mv, mf, face_labels, focus, args.max_triangles)
    for fid in selected_faces:
        ids = roi_ids[face_labels[roi_ids] == fid]
        if len(ids):
            predicted.append(mesh_trace(mv, mf[ids], f'预测面 F{fid} → GT F{score["face_map"][fid]}', color(fid), opacity=.42))
            center = points[ids].mean(0)[None]
            predicted.append(scatter(center, f'F{fid}', color(fid), 'faces', size=5, text=[f'F{fid} → GT F{score["face_map"][fid]}']))
    truth_faces = sorted({score['face_map'][f] for f in selected_faces if score['face_map'][f] >= 0})
    for fid in truth_faces:
        v, f = occ_mesh(targets.faces[fid]); trace = mesh_trace(v, f, f'真实面 F{fid}', color(fid), opacity=.38)
        if trace:
            truth.append(trace)
    if vertex_ids:
        p = np.array([vertices[v]['position'] for v in vertex_ids])
        predicted.append(scatter(p, '预测端点', '#111827', 'vertices', size=7, text=[f'V{v} → GT V{score["vertex_map"][v]}' for v in vertex_ids]))
        vids = np.flatnonzero(np.isin(vertex_labels, vertex_ids) & np.all((points >= focus[0]) & (points <= focus[1]), axis=1))
        if len(vids):
            keep = stratified_indices(vertex_labels[vids], args.max_points)
            vids = vids[keep]
            for vid in vertex_ids:
                ids = vids[vertex_labels[vids] == vid]
                predicted.append(scatter(points[ids], f'V域 {vid}', '#a855f7', 'vertex_domain', ids=ids, size=2))
    curves = load_records(folder / 'curves.json', 'curve') if (folder / 'curves.json').exists() else {}
    for eid in selected_edges:
        if eid in curves:
            predicted.append(line(curves[eid].evaluate(np.linspace(0, 1, 257)), f'预测曲线 E{eid}', '#e11d48'))
    diagnostics = {}; counts = {}; true_edge_ids = set()
    if edge_id is not None:
        owners = EdgeOracle(targets.edges, dcfg).query(points[node_ids])['nearest_id']
        counts = dict(Counter(map(int, owners)))
        selected = stratified_indices(owners, args.max_points); shown = node_ids[selected]; displayed_owners = owners[selected]
        support_trace = scatter(points[shown], f'预测实例 E{edge_id}（按预测归属）', '#e11d48', 'support', ids=shown, values=values[shown, 1:], size=3)
        support_trace['meta']['owners'] = displayed_owners.tolist()
        predicted.append(support_trace)
        for tid in sorted(counts):
            pick = shown[displayed_owners == tid]; true_edge_ids.add(tid)
            text = [f'node={i} / 预测E{edge_id} / GT E{tid}' for i in pick]
            support_trace = scatter(points[pick], f'支撑 → GT E{tid} ({counts[tid]})', color(tid), 'support', ids=pick, text=text, values=values[pick, 1:], size=4 if counts[tid] == max(counts.values()) else 6)
            support_trace['meta']['truth_owner'] = tid
            truth.append(support_trace)
        e = incidence['edges'][edge_id]
        diag.append(association_plot(edge_id, e['faces'], e['vertices'], score))
        diagnostics['support_ownership'] = dict(total=len(node_ids), displayed=len(shown), counts={str(k):v for k,v in counts.items()}, display_counts={str(k):int(np.sum(displayed_owners == k)) for k in counts})
        notes.append('右侧支撑归属对选定实例全部点执行精确CAD距离查询，再分层抽样显示；少数归属保留，悬停显示原始node ID及E/V值。')
    else:
        true_edge_ids.update(score['edge_map'][eid] for eid in selected_edges if score['edge_map'][eid] >= 0)
    for tid in sorted(true_edge_ids):
        truth.append(line(sample_edge(targets.edges[tid]), f'真实边 E{tid}', color(tid), 'truth', 6))
    true_vids = sorted({v for tid in true_edge_ids for v in targets.topology['edges'][tid]['vertex_ids']})
    if true_vids:
        truth.append(scatter(np.asarray(targets.vertices)[true_vids], '真实端点', '#111827', 'vertices', size=7, text=[f'GT V{v}' for v in true_vids]))
    if sid == '00008133':
        missing_faces = sorted(set(range(len(targets.faces))) - set(score['face_map']))
        missing_edges = sorted(set(range(len(targets.edges))) - set(score['edge_map']))
        for fid in missing_faces:
            v, f = occ_mesh(targets.faces[fid]); trace = mesh_trace(v, f, f'缺失真实面 F{fid}', '#dc2626', 'missing', .75)
            if trace:
                truth.append(trace)
        for tid in missing_edges:
            truth.append(line(sample_edge(targets.edges[tid]), f'缺失真实边 E{tid}', '#dc2626', 'missing', 5))
        diagnostics.update(missing_faces=missing_faces, missing_edges=missing_edges)
        notes.append('缺失实体显示在真值侧；点“全局”可查看所有缺失面与缺失边。缺失按冻结dominant映射诊断。')
    views = {'local': dict(title_left='预测分割、支撑和端点', title_right='真实CAD与支撑真实归属', left=predicted, right=truth, bounds=focus.tolist())}
    if sid == '00005094':
        from bandbrep.field_provider import CallableUDFProvider, MeshFieldProvider
        from run_mesh_input_experiment import ExactUDF
        oracle = ExactUDF(targets, dcfg)
        provider = MeshFieldProvider(SurfaceMesh(bgv, bgf), CallableUDFProvider(oracle.values, fingerprint='offline-diagnostic', presence=('present', 'present')), cfg)
        ends = np.array([vertices[v]['position'] for v in incidence['edges'][edge_id]['vertices']])
        replay = order_replay(points[node_ids], ends, provider, cfg)
        missed = replay['coverage'] > 3 * cfg.curve_tolerance
        replay_left = list(predicted) + [line(replay['q'], '诊断重放：实际MST开路径', '#2563eb', 'diagnostic', 7), scatter(replay['p'][missed], '遗漏的投影支撑', '#dc2626', 'diagnostic', size=6)]
        replay_left.append(connectors(replay['p'][missed], replay['closest'][missed], '遗漏支撑到所选路径'))
        views['ordering'] = dict(title_left='诊断重放：开路径与遗漏支撑', title_right='真实闭环（无端点）', left=replay_left, right=truth, bounds=focus.tolist())
        diag.append(dict(title='诊断重放：投影支撑到所选路径距离', data=[dict(type='scatter', mode='lines+markers', y=replay['coverage'].tolist(), name='精确点到折线距离'), dict(type='scatter', mode='lines', x=[0, len(replay['p'])-1], y=[.003, .003], name='覆盖阈值0.003')], layout=dict(xaxis=dict(title='代表点索引'), yaxis=dict(title='归一化距离'))))
        diagnostics['ordering_replay'] = dict(error=replay.get('error'), guard_upper_bound=float(replay.get('worst', 0)), exact_distance_max=float(replay['coverage'].max()), omitted=int(missed.sum()), projected_points=replay['p'].tolist(), selected_path=replay['q'].tolist())
        notes.append('开路径来自冻结order_general函数异常现场；投影按冻结256代表点、8轮差分重放。连线默认真实比例，可放大。')
    if sid == '00004627':
        probe = surface_probe(folder, SurfaceMesh(mv, mf), face_labels, targets, cfg, dcfg)
        b, fitted = probe['boundary'], probe['fit']; residual = np.linalg.norm(b-fitted, axis=1)
        trace = scatter(b, '边界等式残差', residual.tolist(), 'diagnostic', size=4)
        trace['marker'].update(colorscale='Turbo', showscale=True, colorbar=dict(title='边界距离'))
        failure_left = list(predicted) + [trace, connectors(b[::4], fitted[::4], '边界到诊断最小二乘解')]
        surface = mesh_trace(probe['grid'], probe['faces'], '诊断最小二乘曲面（未通过硬边界）', '#a855f7', 'diagnostic', .35)
        failure_left.append(surface)
        views['surface'] = dict(title_left='诊断曲面与无法满足的边界', title_right='真实B样条面与边界', left=failure_left, right=truth, bounds=focus.tolist())
        diag.append(dict(title='PCA初始化UV：边界与内部点', data=[dict(type='scatter', mode='markers', x=probe['uv'][:, 0].tolist(), y=probe['uv'][:, 1].tolist(), marker=dict(size=3, color='#94a3b8'), name='内部点'), dict(type='scatter', mode='markers', x=probe['boundary_uv'][:, 0].tolist(), y=probe['boundary_uv'][:, 1].tolist(), marker=dict(size=5, color=residual.tolist(), colorscale='Turbo'), name='边界点，按诊断残差着色')], layout=dict(xaxis=dict(title='u'), yaxis=dict(title='v', scaleanchor='x'))))
        diag.append(dict(title='增加控制点后的硬约束残差', data=[dict(type='scatter', x=probe['controls'], y=probe['residuals'], mode='lines+markers', name='最大坐标残差'), dict(type='scatter', x=probe['controls'], y=[1e-7]*len(probe['controls']), mode='lines', name='硬约束阈值1e-7')], layout=dict(xaxis=dict(title='每轴控制点数'), yaxis=dict(type='log', title='最大坐标残差'))))
        diagnostics['surface_probe'] = dict(controls=probe['controls'], residuals=probe['residuals'], uv=probe['boundary_uv'].tolist(), boundary=b.tolist(), diagnostic_fit=fitted.tolist())
        notes.append('紫色曲面仅为首次UV迭代的无约束边界最小二乘诊断，不是本轮成功拟合曲面；硬约束在该处已失败。')
    if sid == '00000602':
        origin = batch / 'failure_analysis_20261114'; seam = []
        for stage in ('before', 'after'):
            path = origin / f'seam_{stage}_0.brep'; s = TopoDS_Shape()
            if not breptools.Read(s, str(path), BRep_Builder()):
                raise RuntimeError('Missing saved seam diagnostic: ' + str(path))
            v, f = occ_mesh(s); traces = []
            surface_trace = mesh_trace(v, f, '球面', '#cbd5e1', 'faces', .28)
            if surface_trace:
                traces.append(surface_trace)
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopoDS import topods
            ex = TopExp_Explorer(s, TopAbs_EDGE); edges = []
            while ex.More():
                edge = topods.Edge(ex.Current())
                if not any(edge.IsSame(e) for e in edges):
                    edges.append(edge)
                ex.Next()
            for j, e in enumerate(edges):
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                ad = BRepAdaptor_Curve(e)
                split_circle = int(ad.GetType()) == 1 and ad.LastParameter()-ad.FirstParameter() > .7
                span = ad.LastParameter() - ad.FirstParameter()
                is_seam = stage == 'after' and int(ad.GetType()) == 6 and (abs(ad.FirstParameter()) > 1e-6 or abs(ad.LastParameter()-1) > 1e-6)
                shade = ('#ea580c' if stage == 'after' and span < np.pi else '#dc2626') if split_circle else ('#a855f7' if is_seam else '#64748b')
                suffix = '（完整圆）' if split_circle and stage == 'before' else '（切分圆段）' if split_circle else '（新增接缝）' if is_seam else ''
                samples = sample_edge(e)
                traces.append(line(samples, f'{stage} OCC边 {j}' + suffix, shade, 'diagnostic', 8 if split_circle or is_seam else 3))
                if stage == 'after' and split_circle:
                    traces.append(scatter(samples[[0,-1]], f'圆段{j}接缝端点', '#ea580c', 'vertices', size=8, text=[f'圆段{j}端点 t={ad.FirstParameter():.6f}', f'圆段{j}端点 t={ad.LastParameter():.6f}']))
            seam.append(traces)
        views['seam'] = dict(title_left='接缝修复前：14条边', title_right='接缝修复后：16条边', left=seam[0], right=seam[1], bounds=focus.tolist())
        diagnostics['seam_probe'] = read(origin / 'assembly_probes.json')['sphere_seam_repairs']
        circle_parts = sorted((e for e in diagnostics['seam_probe'][0]['added'] if e['type'] == 1 and e['last']-e['first'] > .7), key=lambda e:e['first'])
        if len(circle_parts) != 2:
            raise RuntimeError('Saved seam diagnostic does not contain the expected two circle pieces')
        diag.append(dict(title='共享圆边被切分：参数区间和对象身份', data=[dict(type='bar', orientation='h', y=['修复前：完整共享边', '修复后：圆段A', '修复后：圆段B'], x=[2*np.pi]+[e['last']-e['first'] for e in circle_parts], base=[0]+[e['first'] for e in circle_parts], marker=dict(color=['#dc2626','#dc2626','#ea580c']), name='圆边参数范围', hovertemplate='%{y}<br>起点%{base:.6f}，跨度%{x:.6f}<extra></extra>')], layout=dict(xaxis=dict(title='CAD圆参数（弧度）', range=[0,2*np.pi]), yaxis=dict(autorange='reversed'))))
        notes.append('接缝前后形状直接读取11-14诊断保存BRep。颜色按各自遍历序号，不代表边对象身份；身份变化见下方证据。')
    if sid == '00008222':
        rejected, rows = pcurve_probe(folder / 'rejected_face_28.brep'); v, f = occ_mesh(rejected)
        left = [mesh_trace(v, f, '拒绝的环面', '#cbd5e1', 'faces', .35)]; right = list(truth)
        table = []
        for r in rows:
            name = f'边遍历#{r["index"]} SameParameter={r["same_parameter"]}'
            shade = '#059669' if r['same_parameter'] else '#dc2626'
            left.extend([line(r['a'], name+' 三维曲线', shade, 'diagnostic'), line(r['b'], name+' pcurve反投影', '#2563eb', 'diagnostic', 3), connectors(r['a'][::8], r['b'][::8], name+' 偏差')])
            table.append(dict(index=r['index'], same_parameter=r['same_parameter'], max_sampled_error=float(r['error'].max()), tolerance=r['tolerance']))
            diag.append(dict(title=name, data=[dict(type='scatter', mode='lines', x=r['t'].tolist(), y=r['error'].tolist(), name='|C(t)-S(pcurve(t))|'), dict(type='scatter', mode='lines', x=[0,1], y=[r['tolerance']]*2, name='边容差')], layout=dict(xaxis=dict(title='同一参数t'), yaxis=dict(title='归一化距离'))))
        views['parameters'] = dict(title_left='三维边与pcurve映射位置', title_right='真实CAD面与真实边', left=left, right=right, bounds=focus.tolist())
        diagnostics['pcurve_probe'] = table
        notes.append('红色边的SameParameter为假；连线默认真实比例，可调倍率。采样误差在容差内不能替代连续一致性检查。未修改任何标志。')
    # Export no unknown or failed curves as if they had been reconstructed.
    notes.append('真实CAD信息仅用于离线诊断。原始记录、拟合结果和冻结源码不被改写；所有长度为归一化坐标。')
    meta = dict(id=sid, title=CASES[sid][0], explanation=CASES[sid][1], stage=result['failed_stage'], error=result['error'],
                counts=dict(truth=score['truth'], predicted=score['predicted']), fixed_face_good=score['face_good'], fixed_graph_good=score['graph_good'],
                active_edge=edge_id, active_face=face_id, notes=notes, source_group=str(original), global_bounds=global_bounds.tolist(),
                displayed_triangles=len(roi_ids), original_triangles=len(mf), seconds=time.monotonic()-started)
    payload = dict(metadata=meta, views=views, plots=diag, diagnostics=diagnostics)
    target = out / 'cases' / sid; target.mkdir(parents=True)
    write_json(target / 'data.json', payload)
    template = Path(__file__).with_name('failure_viewer.html').read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    (target / 'index.html').write_text(template.replace('__PAYLOAD__', encoded).replace('__CASE_TITLE__', html.escape(sid+' '+CASES[sid][0])))
    # In addition to existing identity checks, hash newly read downstream and diagnostic files.
    inputs = dict(identity['inputs']); inputs.update({str(snapshot/name):h for name,h in identity['code'].items()})
    for path in [batch/'records'/(sid+'.json'), Path(policy['input_group'])/'records'/(sid+'.json'), folder/'curve_recovery_progress.json', folder/'cad_graph.json', folder/'curves.json', folder/'face_patches.json', folder/'rejected_face_28.brep']:
        if path.exists():
            inputs[str(path)] = digest(path)
    if sid == '00000602':
        for name in ['seam_before_0.brep','seam_after_0.brep','assembly_probes.json']:
            path=batch/'failure_analysis_20261114'/name;inputs[str(path)]=digest(path)
    write_json(target / 'identity.json', dict(inputs=inputs, artifacts={p.name:digest(p) for p in [target/'data.json', target/'index.html']},
                                             exporter=digest(Path(__file__)), template=digest(Path(__file__).with_name('failure_viewer.html'))))
    print(json.dumps(meta, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/curve_distance_coarse003_20261109_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--max-points', type=int, default=18000)
    parser.add_argument('--max-triangles', type=int, default=45000)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--case', choices=list(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args(); args.batch=args.batch.resolve();args.output=args.output.resolve()
    if args.max_points < 100 or args.max_triangles < 100:
        parser.error('Display budgets must be at least 100')
    if args.worker:
        worker(args);return
    if args.output.exists():
        parser.error('Use a new output directory; existing exports are immutable')
    if len(args.cases)!=len(set(args.cases)):
        parser.error('Duplicate cases')
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        parser.error('Install visualization dependencies: pip install -e ./band_brep_reconstruction[visualization]')
    args.output.mkdir(parents=True);(args.output/'assets').mkdir();(args.output/'source_snapshot').mkdir()
    (args.output/'assets/plotly.min.js').write_text(get_plotlyjs())
    for name in [Path(__file__).name, 'failure_viewer.html']:
        shutil.copy2(Path(__file__).with_name(name),args.output/'source_snapshot'/name)
    write_json(args.output/'policy.json',dict(batch=str(args.batch),cases=args.cases,max_points=args.max_points,max_triangles=args.max_triangles,scope='Offline diagnostics only; no reconstruction results changed'))
    entries=[]
    for sid in args.cases:
        print('Exporting',sid,flush=True)
        log=args.output/(sid+'.log')
        env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        with log.open('w') as stream:
            proc=subprocess.run([sys.executable,str(args.output/'source_snapshot'/Path(__file__).name),'--worker','--case',sid,'--batch',str(args.batch),'--output',str(args.output),'--max-points',str(args.max_points),'--max-triangles',str(args.max_triangles)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:
            entries.append(dict(id=sid,status='export_failed',log=log.name));write_json(args.output/'state.json',dict(status='running',cases=entries));continue
        entries.append(dict(id=sid,status='ready',title=CASES[sid][0],url=f'cases/{sid}/index.html'))
        write_json(args.output/'state.json',dict(status='running',cases=entries))
    cards=''.join(f'<li><a href="{html.escape(e.get("url",e.get("log","")))}">{e["id"]} — {html.escape(e.get("title","导出失败，查看日志"))}</a></li>' for e in entries)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>重建失败可视化</title><style>body{font:17px system-ui;max-width:950px;margin:55px auto;background:#f3f6fb;color:#172033}li{margin:20px}a{color:#1d4ed8}</style><h1>重建失败可视化</h1><p>11-09固定粗0.03批次 · 同步三维对照 · 本地离线打开</p><p>左侧预测，右侧真实CAD；真实信息仅用于诊断。每例页面可切换专门视图、图层、全局/局部范围和偏差放大倍率。</p><ul>'+cards+'</ul>')
    complete=all(e['status']=='ready' for e in entries)
    write_json(args.output/'state.json',dict(status='completed' if complete else 'completed_with_errors',cases=entries))
    write_json(args.output/'identity.json',dict(files={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file() and p != args.output/'identity.json'}))
    print(args.output/'index.html',flush=True)
    if not complete:raise SystemExit(1)


if __name__=='__main__':
    main()
