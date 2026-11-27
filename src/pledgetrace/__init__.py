"""PledgeTrace: traceable extraction of policy proposals from electoral manifestos.

Python reconstructs numbered text units from the PDF; a language model only
selects unit identifiers; Python rebuilds each proposal verbatim with its page,
units, character offsets and bounding box.
"""

__version__ = "1.0.0"
