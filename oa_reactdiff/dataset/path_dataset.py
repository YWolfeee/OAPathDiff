"""
This module is an extension of `transition1x.py` and loads the trajectory data of chemical reactions (instead of only transition states). 

"""
import copy
import numpy as np
import torch

from oa_reactdiff.dataset.base_dataset import BaseDataset, ATOM_MAPPING

n_element = len(list(ATOM_MAPPING.keys()))
FRAG_MAPPING = {
    "reactant": "product",
    "transition_state": "transition_state",
    "product": "reactant",
}


def reflect_z(x):
    x = np.array(x)
    x[:, -1] = -x[:, -1]
    return x


class ProcessedPath(BaseDataset):
    def __init__(
        self,
        datadir,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        single_frag_only=True,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=False,
        only_ts=False,
        confidence_model=False,
        position_key="positions",
        ediff=None,
        **kwargs,
    ):
        super().__init__(
            npz_path=datadir,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )

        if confidence_model:
            use_by_ind = False
        if remove_h:
            print("remove_h is ignored because it is not reasonble for TS.")
        assert not single_frag_only, "single_frag_only is not supported for path dataset"
        assert not use_by_ind, "use_by_ind is not supported for path dataset"
        assert confidence_model is False, "confidence_model is not supported for path dataset"
        assert ediff is None, "ediff is not supported for path dataset"

        # each element in raw is a trajectory
        if swapping_react_prod:
            cpy = copy.deepcopy({'inv' + k[3:] : v[::-1] for k, v in self.raw_dataset.items()}) 
            self.raw_dataset = {**self.raw_dataset, **cpy}
        
        if reflection:
            raise NotImplementedError("reflection is not implemented for path dataset")
            for k, mapped_k in FRAG_MAPPING.items():
                for v, val in self.raw_dataset[k].items():
                    if v in ["wB97x_6-31G(d).forces", position_key]:
                        self.raw_dataset[k][v] += [reflect_z(_val) for _val in val]
                    else:
                        self.raw_dataset[k][v] += val

        # TODO: modify this part
        self.n_fragments = 3
        self.device = torch.device(device)
        self.append_t = kwargs.get('append_t', False)
        self.only_ts = only_ts

        # self.data = {
        #     f"{u}_{v}": [] for u in ["positions", "charges", "fragments"] for v in range(self.n_fragments)
        # }

        ename = "wB97x_6-31G(d).atomization_energy"
        for k, traj in self.raw_dataset.items():
            # add time step into the trajectory
            traj[0]['time'] = 0.0
            traj[-1]['time'] = 1.0
            # find the index of frame with maximum energy
            max_idx = np.argmax([frame[ename] for frame in traj])
            traj[max_idx]['time'] = 0.5
            if self.only_ts:
                self.raw_dataset[k] = [traj[0], traj[max_idx], traj[-1]]
            else:
                # compute time using the linear interpolation of energy
                for i in range(1, max_idx):
                    traj[i]['time'] = (traj[i][ename] - traj[0][ename]) / (traj[max_idx][ename] - traj[0][ename]) / 2
                for i in range(max_idx+1, len(traj)-1):
                    traj[i]['time'] = 0.5 + (traj[i][ename] - traj[max_idx][ename]) / (traj[-1][ename] - traj[max_idx][ename]) / 2
        
        # repeat len(traj) - 2 times for all intermediate frames
        self.reactant = [w[0] for w in self.raw_dataset.values() for _ in range(len(w) - 2)] 
        self.product = [w[-1] for w in self.raw_dataset.values() for _ in range(len(w) - 2)]
        self.transition_state = [w[i] for w in self.raw_dataset.values() for i in range(1, len(w) - 1)]

        self.n_samples = len(self.reactant)

        # for each of reactant, transition_state, product, switch the list of dicts into a dict of list
        for k, v in zip(["reactant", "transition_state", "product"], [self.reactant, self.transition_state, self.product]):
            setattr(self, k, {key: [frame[key] for frame in v] for key in v[0].keys()})
            getattr(self, k)["num_atoms"] = [len(w) for w in getattr(self, k)['atomic_numbers']]
            getattr(self, k)["charges"] = getattr(self, k)['atomic_numbers']

        if not append_frag:
            self.process_molecules(
                "reactant", self.n_samples, idx=0, position_key=position_key, 
            )
            self.process_molecules("transition_state", self.n_samples, idx=1,position_key=position_key,
            )
            self.process_molecules(
                "product", self.n_samples, idx=2, position_key=position_key,
                
            )
        else:
            self.process_molecules(
                "reactant",
                self.n_samples,
                idx=0,
                append_charge=0,
                position_key=position_key,
            )
            self.process_molecules(
                "transition_state", self.n_samples, idx=1, append_charge=1,
            )
            self.process_molecules(
                "product",
                self.n_samples,
                idx=2,
                append_charge=0,
                position_key=position_key,
            )

        for idx in range(pad_fragments):
            self.patch_dummy_molecules(idx + 3)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]

        self.data = np.asarray(self.data)
        print(f"Number of samples: {self.n_samples}")
