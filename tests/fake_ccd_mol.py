"""Stand-in for the RDKit molecules inside ccd.pkl, importable from any process (pickles need that)."""


class Mol:
    def __init__(self, conformers: int = 1):
        self.conformers = conformers

    def GetNumConformers(self) -> int:  # noqa: N802 - RDKit's name
        return self.conformers
