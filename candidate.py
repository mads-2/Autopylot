import itertools as it
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import io_utils as io


@dataclass
class CandidateListGenerator:
    calc_type: str
    nelec: int
    charge: int
    n_singlets: int
    method: Optional[list[str]] = field(default_factory=lambda: [])
    fon_temperature: Optional[list[float]] = field(default_factory=lambda: [])
    rc_w: Optional[list[float]] = field(default_factory=lambda: [])
    active_space: Optional[list[list[int]]] = field(default_factory=lambda: [])

    def create_candidate_list(self):
        attribute_list = [self.method, self.fon_temperature, self.rc_w, self.active_space]
        arg_prod = [x if x else [[]] for x in attribute_list]
        candidate_list = []
        for met, fon, rcw, acs in it.product(*arg_prod):
            d = {'calc_type': self.calc_type, 'nelec': self.nelec, 'charge': self.charge, 'n_singlets': self.n_singlets, 'method': met, 'active_space': acs, 'rc_w': rcw, 'fon_temperature': fon}
            cand = Candidate(**d)
            candidate_list.append(cand)
        return candidate_list


@dataclass
class Candidate:
    calc_type: str
    nelec: int
    charge: int
    n_singlets: int
    method: Optional[str] = None
    fon_temperature: Optional[float] = None
    rc_w: Optional[float] = None
    active_space: Optional[list[int]] = None

    # Tracks number of unique fon_temperatures per method set
    t0_dict = {}
    all_cand = []

    def __init__(self, calc_type, nelec, charge, n_singlets, method=None, fon_temperature=None, rc_w=None, active_space=None):
        """Initialize candidate and track its FON temperature."""
        self.calc_type = calc_type
        self.nelec = nelec
        self.charge = charge
        self.n_singlets = n_singlets
        self.method = method
        self.fon_temperature = fon_temperature
        self.rc_w = rc_w
        self.active_space = active_space

        Candidate.all_cand.append(self)

        if fon_temperature is not None:  
            if calc_type not in Candidate.t0_dict:
                Candidate.t0_dict[calc_type] = set()

            if isinstance(fon_temperature, list):
                for t0 in fon_temperature:
                    Candidate.t0_dict[calc_type].add(float(t0))
            else:
                Candidate.t0_dict[calc_type].add(float(fon_temperature))

    @property
    def orbitals(self):
        """Extracts the number of orbitals from active_space if available."""
        return self.active_space[1] if self.active_space else None

    @property
    def electrons(self):
        """Extracts the number of electrons from active_space if available."""
        return self.active_space[0] if self.active_space else None

    @property
    def folder_name(self):
        """Generates a folder name while omitting _T{self.fon_temperature} if only one unique value exists."""
        name = f'_{self.method}' if self.method else ""

        if self.fon_temperature and len(Candidate.t0_dict[self.calc_type]) > 1:
            name += f'_T{self.fon_temperature}'

        rc_w_check = {m.rc_w for m in Candidate.all_cand if m.rc_w}
        if self.method and self.method.lower() in ['wpbe', 'wpbeh', 'wb97'] and len(rc_w_check) > 1:
            name += f'_w{self.rc_w}' if self.rc_w else ""

        if self.active_space:
            name += f'_AS{self.electrons}{self.orbitals}'

        return name

    @property
    def full_method(self):
        """Generates a full method name while omitting _T{self.fon_temperature} if only one unique value exists."""
        full_method = self.calc_type

        if self.method:
            full_method += f'_{self.method}'

        if self.fon_temperature and len(Candidate.t0_dict[self.calc_type]) > 1:
            full_method += f'_T{self.fon_temperature}'

        rc_w_check = {m.rc_w for m in Candidate.all_cand if m.rc_w}
        if self.method and self.method.lower() in ['wpbe', 'wpbeh', 'wb97'] and len(rc_w_check) > 1:
            full_method += f'_w{self.rc_w}' if self.rc_w else ""

        if self.active_space:
            full_method += f'_AS{self.electrons}{self.orbitals}'

        return full_method

    def validate_as(self):
        if self.active_space:
            acs1 = self.electrons
            acs2 = self.orbitals
            assert ((acs1 % 2) == 0), f"You want {acs1} electrons in your active space? I don't think so."
            assert ((acs1 >= 2) and ((acs1/2) < acs2)), f"How can I put {acs1} electrons in {acs2} orbitals? This is stupid."

    @property
    def calc_settings(self):
        new_settings = {}

        if self.calc_type == 'casci_fomo':
            new_settings = {
                'run': 'energy',
                'method': 'rhf',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'casci': 'yes',
                'fon': 'yes',
                'fon_temperature': self.fon_temperature,
                'closed': int((self.nelec/2)-(self.electrons/2)),
                'active': self.orbitals,
                'cassinglets': self.n_singlets,
                #'castriplets': self.n_triplets,
                'gpus': '1',
                'maxit': '10000',
                'cphfiter': '1000'
                }
        if self.calc_type == 'casscf':
            new_settings = {
                'run': 'energy',
                'guess': 'generate',
                'method': 'rhf',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'fon': 'yes',
                'fon_temperature': '0.2',
                'fon_target': '0.2',
                'fon_anneal': 'no',
                'casscf': 'yes',
                'closed': int((self.nelec/2)-(self.electrons/2)),
                'active': self.orbitals,
                'cassinglets': self.n_singlets,
                #'castriplets': self.n_triplets,
                'gpus': '1',
                'maxit': '10000',
                'cphfiter': '1000',
                'casscfmacromaxiter': '300',
                'casscfmicromaxiter': '100',
                'casscfmaxiter': '400',
                'casscfmacroconvthre': '1.0e-3',
                'casscfconvthre': '1.0e-4',
                'casscfenergyconvthre': '1.0e-6',
                'cpsacasscfmaxiter': '300',
                'cpsacasscfconvthre': '1e-6',
                'casscftrustmaxiter': '100',
                'casscftrustconvthre': '1e-5',
                'dcimaxiter': '50',
                'casscfnriter': '50'
                }
        #if self.calc_type == 'casdft':
            #new_settings = {
                #'run': 'energy',
                #'threall': '1.1e-14',
                #'convthre': '1.0e-6',
                #'precision': 'mixed',
                #'method': self.method,
                #'rc_w': self.rc_w,
                #'fon': 'yes',
                #'fon_temperature': self.fon_temperature,
                #'casci': 'yes',
                #'cphftol': '1.0e-6',
                #'cphfiter': '1000',
                #'cphfalgorithm': 'inc_diis',
                #'closed': int((self.nelec/2)-(self.electrons/2)),
                #'active': self.orbitals,
                #'cassinglets': self.n_singlets,
                #'castriplets': self.n_triplets,
                #'gpus': '2'
                #}
        #if self.calc_type == 'hhtda':
            #new_settings = {
                #'run': 'energy',
                #'charge': int(self.charge-2),
                #'threall': '1.1e-14',
                #'convthre': '1.0e-6',
                #'precision': 'mixed',
                #'hhtda': 'yes',
                #'method': self.method,
                #'rc_w': self.rc_w,
                #'scf': 'diis+a',
                #'cphftol': '1.0e-6',
                #'cphfiter': '1000',
                #'cphfalgorithm': 'inc_diis',
                #'cismax': '300',
                #'cismaxiter': '500',
                #'cisconvtol': '1.0e-6',
                #'cisnumstates': int(self.n_singlets + self.n_triplets),
                #'hhtdasinglets': self.n_singlets,
                #'hhtdatriplets': self.n_triplets,
                #'gpus': '2'
                #}
        if self.calc_type == 'hhtda':
            new_settings = {
                'run': 'energy',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'hhtda': 'yes',
                'method': self.method,
                'fon': 'yes',
                'fon_temperature': self.fon_temperature,
                'cphftol': '1.0e-6',
                'cphfiter': '1000',
                'cphfalgorithm': 'inc_diis',
                'scf': 'diis+a',
                'cismax': '300',
                'cismaxiter': '500',
                'cisconvtol': '1.0e-6',
                'cisnumstates': int(self.n_singlets),
                'closed': 0,
                'active': int((self.nelec/2)+1),
                'hhtdasinglets': self.n_singlets,
                'maxit': '10000',
                #'hhtdatriplets': self.n_triplets,
                'gpus': '1'
                }
            if self.method.lower() in ['wpbe','wpbeh','wb97']:
                 new_settings['rc_w'] = self.rc_w
        return new_settings

def main():
    fn = Path('test_input.yaml')
    settings = io.yload(fn)
    a = CandidateListGenerator(**settings['candidates']['fomo'])
    print(f'{a=}')
    b = a.create_candidate_list()
    print(f'{b=}')
    c = b[0]
    c.validate_as()
    print(c.folder_name)

if __name__ == "__main__":
    main()
