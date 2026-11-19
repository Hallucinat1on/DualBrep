import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from bandbrep.config import Config
from bandbrep.schemas import SurfaceMesh,ReconstructionError
from bandbrep.region_repair import repair_regions,RegionState,validate_repair_coverage
from bandbrep.bands import bands_from_labels,build_bands
from bandbrep.surface_mesh import evaluate_batched
from bandbrep.vendor.dualbrep_segmentation import hierarchical_segmentation
from helpers import AnalyticFields,box


def grid():
    nx,ny=13,9
    v=np.array([[x,y,0.] for y in np.linspace(-.008,.008,ny) for x in np.linspace(-.012,.012,nx)])
    f=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;f.extend([[a,a+1,a+nx],[a+1,a+nx+1,a+nx]])
    mesh=SurfaceMesh(v,np.array(f));q=v[mesh.triangles].mean(1)
    return mesh,q


def field(edge=.025,vertex=False):
    return AnalyticFields(lambda q:np.column_stack((q[:,2],abs(q[:,0]-edge),
        np.linalg.norm(q,axis=1) if vertex else np.ones(len(q)))),
        presence=('present','present' if vertex else 'absent'))


def fixture(edge=.025,vertex=False):
    mesh,q=grid();p=field(edge,vertex);v=evaluate_batched(p,q)
    labels=np.zeros(len(q),np.int32);stripe=abs(q[:,0])<.002
    labels[stripe]=-1;reasons=np.where(stripe,3,0).astype(np.int8)
    cfg=Config(face_second_pass=False,region_repair='single_face',face_mean_filter=False,
        face_threshold1=.003,region_repair_samples=16)
    return mesh,q,p,v,labels,reasons,cfg


def test_mean_filter_can_be_disabled_without_disabling_size_filter():
    mesh=SimpleNamespace(face_adjacency=np.array([[0,1],[2,3]]))
    args=(mesh,np.array([.006,.006,.02,.02]),.005,.01,2,.007)
    assert hierarchical_segmentation(*args,second_pass=False)[0]==1
    assert hierarchical_segmentation(*args,second_pass=False,mean_filter=False)[0]==2


def test_absorb_interior_strip_preserves_mesh_values_and_masks():
    mesh,q,p,v,labels,reasons,cfg=fixture();oldv=v.copy();oldf=mesh.triangles.copy();oldreason=reasons.copy()
    state=repair_regions(mesh,q,v,labels,reasons,p,cfg)
    assert state.report['accepted']>0
    assert state.absorbed.any() and np.all(state.labels>=0)
    b=bands_from_labels(mesh,q,v,state.labels,np.zeros_like(reasons),p,cfg,{},state)
    assert not b.edge_band.any() and not b.edge_support_mask.any()
    assert not state.trusted[state.absorbed & (v[:,1]<cfg.face_threshold1)].any()
    np.testing.assert_array_equal(v,oldv);np.testing.assert_array_equal(mesh.triangles,oldf)
    np.testing.assert_array_equal(reasons,oldreason);np.testing.assert_array_equal(state.raw_reasons,oldreason)


def test_true_interface_between_coplanar_faces_is_not_absorbed():
    mesh,q,p,v,labels,reasons,cfg=fixture(edge=0.)
    labels[q[:,0]>.002]=1
    state=repair_regions(mesh,q,v,labels,reasons,p,replace(cfg,region_repair='joint'))
    assert not state.absorbed.any()
    assert set(state.labels[state.labels>=0])=={0,1}
    assert any(a.get('reason') in ('observed_E_valley','independent_E_branch_in_patch') for a in state.report['actions'])


def test_single_face_does_not_hide_a_missing_other_face():
    mesh,q,p,v,labels,reasons,cfg=fixture(edge=0.)
    state=repair_regions(mesh,q,v,labels,reasons,p,cfg)
    assert not state.absorbed.any()
    np.testing.assert_array_equal(state.labels,labels)


def test_vertex_neighborhood_is_protected():
    mesh,q,p,v,labels,reasons,cfg=fixture(vertex=True)
    state=repair_regions(mesh,q,v,labels,reasons,p,cfg)
    assert not state.absorbed.any()
    assert any(a.get('reason')=='necessary_V_near_patch' for a in state.report['actions'])


def test_local_coverage_rejects_absorption_without_retained_edges():
    mesh,q,p,v,labels,reasons,cfg=fixture()
    state=repair_regions(mesh,q,v,labels,reasons,p,cfg)
    assert len(state.protected_e)>0
    b=bands_from_labels(mesh,q,v,state.labels,np.zeros_like(reasons),p,cfg,{},state)
    with pytest.raises(ReconstructionError,match='No retained curves'):
        validate_repair_coverage(b,{},[],p,cfg)


def test_repair_requires_single_pass_and_valid_budgets():
    with pytest.raises(ValueError,match='first-pass-only'):Config(region_repair='joint').validate()
    for key in ('region_repair_samples','region_repair_max_branches','region_repair_max_rounds'):
        with pytest.raises(ValueError):replace(Config(),**{key:0}).validate()


def test_failed_candidate_rolls_back_to_same_prepared_mesh(tmp_path):
    from bandbrep.reconstruct import reconstruct_from_prepared
    from bandbrep.field_provider import MeshFieldProvider,CenterFields
    from bandbrep.topology import recover_topology
    mesh,udf=box();cfg=Config(face_filter_size=1,face_second_pass=False,region_repair='single_face',
        face_mean_filter=False,save_queries=False,stop_after='graph')
    provider=MeshFieldProvider(mesh,udf,cfg);q=mesh.vertices[mesh.triangles].mean(1)
    centers=CenterFields(q,evaluate_batched(provider,q),provider.fingerprint)
    def build(mesh,provider,cfg,center_fields):
        _,b=build_bands(mesh,provider,replace(cfg,region_repair='off'),center_fields)
        if cfg.region_repair!='off':
            b.report['region_repair']=dict(accepted=1)
            b.raw_face_labels=b.face_labels.copy();b.raw_exclusion_reasons=b.exclusion_reasons.copy()
            b.trusted_core_mask=b.core_mask.copy();b.absorbed_mask=np.zeros(len(q),bool)
            b.edge_support_mask=b.edge_band.copy()
            b.repair_state=RegionState(b.raw_face_labels,b.raw_exclusion_reasons,b.face_labels,
                b.trusted_core_mask,b.absorbed_mask,np.empty((0,3)),np.empty((0,3)),b.report['region_repair'])
        return mesh,b
    def topology(mesh,bands,seeds,cfg,provider):
        if cfg.region_repair!='off':
            (tmp_path/'reconstruction.unverified.step').write_text('failed candidate STEP')
            raise ReconstructionError('candidate_rejected','test downstream rejection')
        return recover_topology(mesh,bands,seeds,cfg,provider)
    with patch('bandbrep.reconstruct.build_bands',side_effect=build),patch('bandbrep.reconstruct.recover_topology',side_effect=topology):
        result=reconstruct_from_prepared(mesh,mesh,centers,udf,cfg,tmp_path,provenance={'test':'verified_fixture'})
    assert result['status']=='logical_graph_ready'
    assert result['effective_config']['region_repair']=='off'
    assert [r['status'] for r in result['repair_attempts']]==['failed','logical_graph_ready']
    assert (tmp_path/'attempts/00_single_face/bands.npz').exists()
    assert (tmp_path/'attempts/00_single_face/reconstruction.unverified.step').read_text()=='failed candidate STEP'
    assert not (tmp_path/'reconstruction.unverified.step').exists()
    with np.load(tmp_path/'working_mesh.npz') as d:np.testing.assert_array_equal(d['triangles'],mesh.triangles)
    saved=json.loads((tmp_path/'run.json').read_text());assert saved['config']['region_repair']=='single_face'


def test_incompatible_small_patch_is_restored_as_an_independent_face():
    # A raised, nearly planar patch cannot be explained by either surrounding
    # plane. Its own independent samples certify a surviving high-E interior.
    mesh,q=grid();p=field();values=evaluate_batched(p,q)
    center=(abs(q[:,0])<.002)&(abs(q[:,1])<.002)
    labels=np.where(q[:,0]<0,0,1).astype(np.int32);labels[center]=-1
    reasons=np.where(center,3,0).astype(np.int8)
    # Use a separate planar island attached in topology but displaced in 3D.
    chosen=np.flatnonzero(center)
    patch_vertices=np.unique(mesh.triangles[chosen])
    mesh.vertices[patch_vertices,2]=.01
    q=mesh.vertices[mesh.triangles].mean(1);values=evaluate_batched(p,q)
    cfg=Config(face_second_pass=False,region_repair='small',face_mean_filter=False,
        face_threshold1=.003,region_repair_samples=16)
    state=repair_regions(mesh,q,values,labels,reasons,p,cfg)
    assert np.all(state.labels[chosen]>=2)
    assert any(a['kind']=='restore_small_face' and a['accepted'] for a in state.report['actions'])
    # Never force a geometrically incompatible island into a surrounding face.
    for action in state.report['actions']:
        if action.get('accepted') and action.get('kind')=='small_component':
            assert not np.intersect1d(chosen,action['nodes']).size


def test_invalid_field_queries_veto_absorption():
    mesh,q,p,v,labels,reasons,cfg=fixture()
    bad=AnalyticFields(lambda q:np.full((len(q),3),np.nan),presence=('present','absent'))
    state=repair_regions(mesh,q,v,labels,reasons,bad,cfg)
    assert not state.absorbed.any()


def test_hole_ring_with_real_valley_is_preserved():
    mesh,q=grid();radius=.005
    p=AnalyticFields(lambda x:np.column_stack((x[:,2],abs(np.linalg.norm(x[:,:2],axis=1)-radius),np.ones(len(x)))),presence=('present','absent'))
    values=evaluate_batched(p,q);ring=abs(np.linalg.norm(q[:,:2],axis=1)-radius)<.002
    labels=np.where(ring,-1,0).astype(np.int32);reasons=np.where(ring,1,0).astype(np.int8)
    state=repair_regions(mesh,q,values,labels,reasons,p,Config(face_second_pass=False,
        region_repair='single_face',face_threshold1=.003,region_repair_samples=32))
    assert not state.absorbed.any()


def test_prepared_resume_restores_raw_evidence_without_repairing_again(tmp_path):
    from bandbrep.reconstruct import reconstruct_from_prepared
    from bandbrep.field_provider import MeshFieldProvider,CenterFields
    mesh,udf=box();cfg=Config(face_filter_size=1,face_second_pass=False,region_repair='joint',
        save_queries=False,stop_after='seeds')
    provider=MeshFieldProvider(mesh,udf,cfg);points=mesh.vertices[mesh.triangles].mean(1)
    centers=CenterFields(points,evaluate_batched(provider,points),provider.fingerprint)
    args=(mesh,mesh,centers,udf)
    first=reconstruct_from_prepared(*args,cfg,tmp_path,provenance={'test':'prepared'})
    assert first['status']=='face_seeds_ready'
    with np.load(tmp_path/'bands.npz') as a:
        trusted=a['trusted_core_mask'].copy();raw=a['raw_face_labels'].copy()
    with patch('bandbrep.region_repair.repair_regions',side_effect=AssertionError('repair should be cached')):
        final=reconstruct_from_prepared(*args,replace(cfg,stop_after='graph'),tmp_path,
            provenance={'test':'prepared'},resume=True)
    assert final['status']=='logical_graph_ready'
    with np.load(tmp_path/'bands.npz') as a:
        np.testing.assert_array_equal(a['raw_face_labels'],raw)
        np.testing.assert_array_equal(a['trusted_core_mask'],trusted)


def test_prepared_resume_rejects_changed_center_observations(tmp_path):
    from bandbrep.reconstruct import reconstruct_from_prepared
    from bandbrep.field_provider import MeshFieldProvider,CenterFields
    mesh,udf=box();cfg=Config(face_filter_size=1,face_second_pass=False,save_queries=False,stop_after='seeds')
    provider=MeshFieldProvider(mesh,udf,cfg);points=mesh.vertices[mesh.triangles].mean(1)
    centers=CenterFields(points,evaluate_batched(provider,points),provider.fingerprint)
    reconstruct_from_prepared(mesh,mesh,centers,udf,cfg,tmp_path,provenance={'test':'same source'})
    changed=centers.values.copy();changed[:,1]*=1.01
    with pytest.raises(ReconstructionError) as exc:
        reconstruct_from_prepared(mesh,mesh,CenterFields(points,changed,provider.fingerprint),udf,cfg,tmp_path,
            provenance={'test':'same source'},resume=True)
    assert exc.value.code=='cache_mismatch'
