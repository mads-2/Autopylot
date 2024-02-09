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
        state = fields[1] # The 2-nd index of each dictionary, is the state S_n
        vee = float(fields[4]) # The 5-th index of each dictionary, is the excitation energy in eV
        osc = float(fields[5]) # The 6-th index of each dictionary, is the oscillation strength
        if state == 'triplet':
            osc = None
        return cls(state, vee, osc)


@dataclass
class FileParser:
    n_singlets: int # Number of singlets
    n_triplets: int # Number of triplets
    hhtda_based: bool # hh-TDA or not? True & False
    fn: Path # Path of the output file
    excitation_lines: list[str] = field(default_factory=lambda: [])

    def parse_exc_lines_from_TC(self): # This function reads excitation energies, and oscillation strengths for hh-TDA
        number_of_exc_states = self.n_singlets + self.n_triplets - 1 # Cope with python's starting from 0
        text = 'Root   Mult.   Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)' # This line locates the excitation energy output from .out file 
        # For hh-TDA computation, the oscillation strength will be printed at the same line as text
        juicy_lines = [] # This list stores the excited state data, containing also oscillation strength for hh-TDA based computations
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if text in line: # Search for the corresponding output in .out file
                    index += 3 # Why += 3? Start reading from the line with root 2 (S_1)
                    for i in range(index, index + number_of_exc_states):
                        juicy_lines.append(lines[i]) # Append excitation energy values into the list
        self.excitation_lines = juicy_lines
        # For hh-TDA based comoputations, it will read oscillation strength automatically from this function
        # Otherwise, it only reads excitation energies
        return

    def parse_osc_lines_from_TC(self): # This function reads the oscillation strengths for non-hh-TDA outputs, which is formatted differently as hh-TDA
        osc_string = 'Singlet state electronic transitions:' # This line appears only in .out files which are not hh-TDA based calculations
        osc_lines = []
        with open(self.fn, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if osc_string in line:
                    index += 4 # Data appears in the fourth line below the osc_string
                    for i in range(index, index + self.n_singlets - 1):
                        osc_lines.append(lines[i]) # Append 1->n excitation oscillation strength; not 2->n, 3->n .etc
        return osc_lines

    def add_osc_strengths_into_lines(self, osc_lines): # For non-hh-TDA, append oscillation strengths at the end of the excitation energy file
        new_exc_lines = []
        iterator = iter(osc_lines) # Extract all oscillation strength values
        for line in self.excitation_lines:
            if line.split()[1] == 'singlet': # Singlets
                value = next(iterator).split()[7]
                new_line = f'{line} {value}' # Input oscillation strengths for singlets
            else: # Triplets
                new_line = f'{line} 0.0000' # Do not consider oscillation strengths for triplets
            new_exc_lines.append(new_line)
        self.excitation_lines = new_exc_lines
        return

    def parse_TC(self):
        self.parse_exc_lines_from_TC() # hh-TDA based computations
        if self.hhtda_based == False: # Non-hh-TDA computations
            osc_lines = self.parse_osc_lines_from_TC()
            self.add_osc_strengths_into_lines(osc_lines)
        return

    def create_dict_entry(self): # Create dictionary 
        dict_entry = {}
        sc = 1 # Start from state S_1, the energy of state S_0 has been set to 0
        tc = 1
        for excitation in self.excitation_lines:
            exc = Excitation.from_string(excitation)
            if exc.label == 'singlet': # Create entries for singlets
                dict_entry[f'S{sc}'] = exc
                sc += 1
            else: # Create entries for triplets
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
        return


def main():
    fn = 'eom/ricc2.out' # TC or TM output file
    n_singlets = 5 # Number of singlets being calculated 
    n_triplets = 0 # Number of triplets being calculated 
    hhtda_based = False # To determine if we are running an hh-TDA based method
    f = FileParser(n_singlets, n_triplets, hhtda_based, fn)
    f.parse_TM() 
    di = f.create_dict_entry()
    print(di)


if __name__ == "__main__":
    main()
