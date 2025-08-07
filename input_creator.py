from dataclasses import dataclass

@dataclass
class TerachemInput:
    data: dict

    @classmethod
    def from_default_TC(cls, new_keywords):
        return cls(new_keywords)

    def to_file(self, fn='tc.in'):
        with open(fn, 'w') as f:
            f.write(self.to_string())

    def to_string(self):
        return '\n'.join(f'{k} {v}' for k, v in self.data.items())


@dataclass
class TurbomoleInput:
    data: dict
    
    @staticmethod
    def from_default_eom(n_singlets, charge):
        commands = f'''

a coord
*
no
b all def2-TZVP 
*
eht

{charge}

scf
iter
300

cc
freeze
*
cbas
b all def2-SV(P)
*
memory 5000
ricc2
model cc2
maxiter 300
*
exci
irrep=a multiplicity=1 nexc={n_singlets+1}
spectrum states=all
exprop states=all
*
*
*
'''
        #else:
            #commands = f'''

#a coord
#*
#no
#b all 6-31G**
#*
#eht



#scf
#iter
#300

#cc
#freeze
#*
#cbas




#b all def2-SV(P)
#*
#memory 5000
#ricc2
#model cc2
#maxiter 300
#*
#exci
#irrep=a multiplicity=1 nexc={n_singlets+1}
#irrep=a multiplicity=3 nexc={n_triplets}
#spectrum states=all
#exprop states=all
#*
#*
#*
#'''
        return TurbomoleInput({"input": commands})

    def to_file(self, fn='define-inputs.txt'):
        with open(fn, 'w') as f:
            f.write(self.data["input"])

def main():
    new_keys = {'coordinates': 'geom.xyz', 'basis': '6-31G*', 'method': 'pbe0'}
    input = TerachemInput.from_default_TC(new_keys)
    input.to_file()


if __name__ == "__main__":
    main()
