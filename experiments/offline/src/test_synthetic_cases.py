"""Hand-checkable, pre-result tests for the counterfactual audit."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from audit import (Checks, geometry, normalize, penalties, partition, associate,
                   strict_join, censored_groups, write)


class Cases(unittest.TestCase):
    def setUp(self): self.ck=Checks(); self.r=np.array([1.,1.,0.,0.])
    def close(self,a,b): self.ck.close("fixture",a,b)
    def test_equal_penalty(self):
        y=self.r-np.array([0,0,2,2])
        self.close(y,[1,1,-2,-2]); self.close(normalize(y,"std",0),[1,1,-1,-1])
        self.close(geometry(normalize(self.r,"mean"),normalize(y,"mean"))["norm_ratio"],3)
    def test_unequal_penalty(self):
        y=self.r-np.array([0,0,1,3]); self.close(normalize(y,"mean"),[1.5,1.5,-.5,-2.5]); self.close(y.var(),2.75)
    def test_one_incorrect(self):
        r=np.array([1.,1.,1.,0.])
        for penalty in (0.,.1,2.,100.):
            y=r-np.array([0,0,0,penalty])
            for norm in ("std","mean"):
                m=geometry(normalize(r,norm),normalize(y,norm),self.ck)
                self.close(m["cosine"],1); self.close(m["energy"],0)
    def test_equal_distances_and_zero(self):
        l=np.array([5,7,9,11]); d=np.array([0,0,2,2])
        self.close(penalties(self.r,d,l,"learned",.2),penalties(self.r,d,l,"constant",.2))
        self.close(penalties(self.r,np.zeros(4),l,"learned",.2),np.zeros(4))
    def test_homogeneous(self):
        for r in (np.zeros(4),np.ones(4)):
            for rule in ("learned","constant","length_global","length_group"):
                q=penalties(r,np.ones(4),np.arange(1,5),rule,.2)
                self.close(q,np.zeros(4))
                for norm in ("std","mean"):
                    self.assertEqual(geometry(normalize(r,norm),normalize(r-q,norm))["undefined_reason"],"both_zero")
    def test_prompt_weighting(self):
        x=normalize(self.r,"mean"); z=partition([x,x],[x,3*x],self.ck)
        self.close(z["within_energy"],0); self.assertGreater(z["weight_energy"],0)
        self.close(z["energy"],z["weight_energy"])
        self.close(geometry(x,3*x)["cosine"],1)
    def test_normalization_angle(self):
        y=np.array([1,1,-1,-3])
        self.close(geometry(normalize(self.r,"std"),normalize(y,"std"))["cosine"],geometry(normalize(self.r,"mean"),normalize(y,"mean"))["cosine"])
    def test_stabilizer(self):
        y=np.array([1,1,-2,-2]); s=3.; sd=self.r.std(); e=1e-4
        self.close(geometry(normalize(self.r,"std",e),normalize(y,"std",e))["norm_ratio"],s*(sd+e)/(s*sd+e))
        self.assertGreater(geometry(normalize(self.r,"std",e),normalize(y,"std",e))["relative_change"],0)
    def test_calibration(self):
        d=np.array([0,0,4,12]); l=np.array([5,6,10,30]); coefficient=.1*d.sum()/l[2:].sum()
        q=penalties(self.r,d,l,"length_global",coefficient)
        self.close(q.sum(),.1*d.sum())
        for rule in ("constant","length_group"):
            self.close(penalties(self.r,d,l,rule,coefficient)[2:].mean(),(.1*d[2:]).mean())
            self.close(penalties(self.r,np.zeros(4),l,rule,coefficient),np.zeros(4))
    def test_geometry_identities(self):
        rng=np.random.default_rng(17)
        for _ in range(30):
            r=np.array([1.,0.,0.,0.]); q=np.r_[0.,rng.uniform(0,10,3)]; u=np.r_[0.,q[1:]-q[1:].mean()]; x=r-r.mean(); s=1+q[1:].mean()
            self.close((r-q)-(r-q).mean(),s*x-u); self.close(x@u,0)
            for norm in ("std","mean"):
                m=geometry(normalize(r,norm),normalize(r-q,norm),self.ck)
                self.close(m["cosine"]**2,s*s*(x@x)/(s*s*(x@x)+u@u))
    def test_rank_ties(self):
        self.close(associate([1,1,3],[1,2,3])[0],np.sqrt(3)/2)
        self.assertEqual(associate([1],[3])[1],"fewer_than_two")
        self.assertEqual(associate([1,1],[3,4])[1],"distance_constant")
        self.assertEqual(associate([1,2],[3,3])[1],"length_constant")
        self.assertEqual(associate([1,1],[3,3])[1],"both_constant")
        self.close(associate([1,2],[3,4])[0],1); self.close(associate([1,2],[4,3])[0],-1)
    def fixture(self):
        raw={"per_problem":[]}; geo={"records":[]}
        for gid in ("b","a"):
            samples=[{"correct":True,"prediction":"x","token_ids":[1]}, {"correct":False,"prediction":"y","token_ids":[2,3]}]
            raw["per_problem"].append({"unique_id":gid,"samples":samples,"n_correct":1,"any_correct":True})
            geo["records"].extend({"group_id":gid,"sample_idx":i,"correct":s["correct"],"n_correct":1,"group_type":"mixed","deployed_distance":float(i)} for i,s in enumerate(samples))
        return raw,geo,copy.deepcopy(raw),{"group_size":2,"expected_groups":2,"beta":.1}
    def test_join_order(self):
        a,g,p,c=self.fixture(); first=strict_join(a,g,p,c)
        a["per_problem"].reverse(); g["records"].reverse()
        self.assertEqual(first,strict_join(a,g,p,c))
    def test_invalid_join(self):
        for mode in ("duplicate","missing","distance","string_boolean","bool_token"):
            a,g,p,c=self.fixture()
            if mode=="duplicate": g["records"].append(g["records"][0])
            elif mode=="missing": g["records"].pop()
            elif mode=="distance": g["records"][1]["deployed_distance"]=None
            elif mode=="string_boolean": a["per_problem"][0]["samples"][0]["correct"]="False"
            else: a["per_problem"][0]["samples"][0]["token_ids"]=[True]
            with self.assertRaises(ValueError): strict_join(a,g,p,c)
    def test_cap_exclusion(self):
        def row(gid,hit): return {"group_id":gid,"termination_status":"unknown","nominal_cap_reached":hit,"effective_cap_reached":None}
        groups=[[row("a",False),row("a",True)],[row("b",False),row("b",False)]]
        self.assertEqual(censored_groups(groups),{"a"})
        coefficient=.123
        remaining=[g for g in groups if g[0]["group_id"] not in censored_groups(groups)]
        self.assertEqual(len(remaining),1); self.assertEqual(len(remaining[0]),2)
        self.close(penalties(self.r,[0,0,2,3],[10,20,30,40],"length_global",coefficient)[2:],[30*coefficient,40*coefficient])
    def test_signed_penalty(self):
        # Primary rules are nonnegative; these check the qualified algebra only.
        self.close(geometry(normalize(self.r,"mean"),normalize([1,1,2,2],"mean"))["cosine"],-1)
        self.assertEqual(geometry(normalize(self.r,"mean"),normalize([1,1,1,1],"mean"))["undefined_reason"],"zero_shaped")
        y=np.array([1,1,0,2]); self.close(geometry(normalize(self.r,"mean"),normalize(y,"mean"))["cosine"],0)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(args.output,{"schema_version":1,"passed":result.wasSuccessful(),"tests_run":result.testsRun,"failures":[str(x) for x in result.failures],"errors":[str(x) for x in result.errors],"scope":"Synthetic fixtures only; no new empirical outcomes read"})
    raise SystemExit(0 if result.wasSuccessful() else 1)
