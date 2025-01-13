from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Excitation:
    label: str
    energy: float
    osc: float

    @classmethod
    def from_string(cls, excitation: str):
        fields = excitation.split()
        if len(fields) < 7:
            raise ValueError(f"Expected at least 7 fields, got {len(fields)}. Fields: {fields}")
        state = fields[1]
        vee = float(fields[4])
        osc = float(fields[6])
        if state == 'triplet':
            osc = None
        return cls(state, vee, osc)


@dataclass
class FileParser:
    n_singlets: int
    hhtda_based: bool
    fn: Path
    excitation_lines: list[str] = field(default_factory=lambda: [])

    def parse_exc_lines_from_TC(self):
        number_of_exc_states = self.n_singlets - 1
        text = 'Root   Mult.   Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)'
        juicy_lines = []
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if text in line:
                    index += 3 
                    for i in range(index, index + number_of_exc_states):
                        juicy_lines.append(lines[i])
        self.excitation_lines = juicy_lines
        return

    def parse_nm_lines_from_TC(self):
        number_of_exc_states = self.n_singlets - 1
        nm_string = '|  Root   Mult.      Ex. Wavelength (nm)'
        nm_lines = []
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if nm_string in line:
                    index += 2
                    for i in range(index, index + number_of_exc_states):
                        nm_lines.append(lines[i])
        return nm_lines

    def add_nm_strengths_into_lines(self, nm_lines):
        new_exc_lines = []
        iterator = iter(nm_lines)
        for line in self.excitation_lines:
            if line.split()[1] == 'singlet':
                try:
                    split_line = next(iterator).split()
                    if len(split_line) > 3:  # Ensure there are enough elements
                        vnm = split_line[3]
                    else:
                        vnm = "0.0000"  # Assign a default value or handle appropriately
                except StopIteration:
                    vnm = "0.0000"  # Assign a default value or handle appropriately if nm_lines is exhausted
                new_line = f'{line} {vnm}'
            else:
                new_line = f'{line} 0.0000'
            new_exc_lines.append(new_line)
        self.excitation_lines = new_exc_lines
        return

    def parse_osc_lines_from_TC(self):
        number_of_exc_states = self.n_singlets - 1
        osc_string = 'Singlet state electronic transitions:'
        osc_lines = []
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if osc_string in line:
                    index += 4
                    for i in range(index, index + self.n_singlets - 1):
                        osc_lines.append(lines[i])
        return osc_lines    

    def add_osc_strengths_into_lines(self, osc_lines):
        new_exc_lines = []
        iterator = iter(osc_lines)
    
        for line in self.excitation_lines:
            try:
                if line.split()[1] == 'singlet':
                   # Try to get the oscillator strength
                   vosc = next(iterator).split()[7]
                   new_line = f'{line} {vosc}'
                else:
                    new_line = f'{line} 0.0000'
            except StopIteration:
                print("Error: Oscillator strengths data is incomplete or malformed.")
                new_line = f'{line} NaN'
            except IndexError:
                print(f"Error: Expected oscillator strength data not found in line: {line}")
                new_line = f'{line} NaN'
            except Exception as e:
                print(f"Unexpected error: {e}")
                new_line = f'{line} NaN'
        
            new_exc_lines.append(new_line)
    
        self.excitation_lines = new_exc_lines
        return

    def parse_TC(self):
        self.parse_exc_lines_from_TC()
        #if not self.hhtda_based:
        if self.hhtda_based == False:
            nm_lines = self.parse_nm_lines_from_TC()
            self.add_nm_strengths_into_lines(nm_lines)
            osc_lines = self.parse_osc_lines_from_TC()
            self.add_osc_strengths_into_lines(osc_lines)
        return

    def create_dict_entry(self):
        dict_entry = {}
        sc = 1
        tc = 1
        for excitation in self.excitation_lines:
            exc = Excitation.from_string(excitation)
            if exc.label == 'singlet':
                dict_entry[f'S{sc}'] = exc
                sc += 1
            else:
                dict_entry[f'T{tc}'] = exc
                tc += 1
        return dict_entry

    def parse_TM(self):
        osc_string = "oscillator strength (length gauge)"
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if osc_string in line:
                    counter = index - 12
                    osc = float(line.split()[5])
                    mult = lines[counter].split()[6]
                    state = ('singlet' if mult == '1' else 'triplet')
                    energy = lines[counter + 1].split()[5]
                    excitation_line = f"X {state} X X {energy} X {osc}"
                    self.excitation_lines.append(excitation_line)
        print(self.excitation_lines)
        return


def main():
    #fn = 'eom/ricc2.out'
    #fn = 'casscf_AS44/tc.out'
    fn = 'hhtda_fomo_wB97x_T0.15_w0.2/tc.out'
    n_singlets = 5
    #hhtda_based = False
    hhtda_based = True
    f = FileParser(n_singlets, hhtda_based, fn)
    f.parse_TM()
    di = f.create_dict_entry()
    print(di)


if __name__ == "__main__":
    main()
