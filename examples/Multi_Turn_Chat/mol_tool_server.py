################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from charge.servers.SMILES import SMILES_mcp

if __name__ == "__main__":
    SMILES_mcp.run(transport="sse")
