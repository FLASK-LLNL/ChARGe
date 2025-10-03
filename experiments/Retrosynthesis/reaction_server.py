################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from charge.servers.SMARTS_reactions import SMARTS_mcp

if __name__ == "__main__":
    SMARTS_mcp.run(
        transport="stdio",
    )
