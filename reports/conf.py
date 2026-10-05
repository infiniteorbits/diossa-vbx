import os
import sys

from multiproject.utils import get_project
from sphinx_lmodoc import conf_helper

sys.path.insert(0, os.path.abspath('.'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DIOSSA Phase 2 GSTP Documents"
copyright = "2025, LMO Sarl"
author = "Maciej Zurad"
release = "1.0"
project_id = "PL24-0001"
document_id = "TRACE"
project_label = "DIOSSA"

multiproject_projects = {
    "traceability": {
        "path": '.',
        "config": {
            "project": project,
            "copyright": copyright,
            "author": author,
            "release": release,
            "project_id": project_id,
            "document_id": document_id,
            "project_label": project_label
        },
        "use_config_file": False
    },
    "RS_0001_URS": {"use_config_file": False},
    "TN_0001_MDD": {"use_config_file": False},
    "ICD_0001_SYS": {"use_config_file": False},
    "TN_0002_ConOps": {"use_config_file": False},
    "RS_0002_SYS": {"use_config_file": False},
    "RS_0003_VIS": {"use_config_file": False},
    "RS_0004_TIR": {"use_config_file": False},
    "RS_0005_CMP": {"use_config_file": False},
    "RS_0006_SW": {"use_config_file": False},
    "RS_0009_FPGA": {"use_config_file": False},
    "ICD_0002_SW": {"use_config_file": False},
    "DJF_0001_SW": {"use_config_file": False},
    "PL_0001_VP": {"use_config_file": False},
    "PL_0004_SW": {"use_config_file": False},
    "DD_0002_OHU": {"use_config_file": False},
    "UM_0001_UserManual": {"use_config_file": False},
    "PL_0002_ENV": {"use_config_file": False},
    "SW_0002_CNN": {"use_config_file": False},
    "SW_0004_ASW": {"use_config_file": False},
    "AD_0001_SW": {"use_config_file": False},
    "RP_0001_SCAR": {"use_config_file": False},
    "DJF_0002_ALG": {"use_config_file": False},
    "PolarFire_Xilinx_Tradeoff": {"use_config_file": False},
}

current_project = get_project(multiproject_projects)
if current_project == "RS_0001_URS":
    from RS_0001_URS.conf import *
elif current_project == "TN_0001_MDD":
    from TN_0001_MDD.conf import *
elif current_project == "ICD_0001_SYS":
    from ICD_0001_SYS.conf import *
elif current_project == "TN_0002_ConOps":
    from TN_0002_ConOps.conf import *
elif current_project == "RS_0002_SYS":
    from RS_0002_SYS.conf import *
elif current_project == "RS_0003_VIS":
    from RS_0003_VIS.conf import *
elif current_project == "RS_0004_TIR":
    from RS_0004_TIR.conf import *
elif current_project == "RS_0005_CMP":
    from RS_0005_CMP.conf import *
elif current_project == "RS_0006_SW":
    from RS_0006_SW.conf import *
elif current_project == "RS_0009_FPGA":
    from RS_0009_FPGA.conf import *
elif current_project == "ICD_0002_SW":
    from ICD_0002_SW.conf import *
elif current_project == "DJF_0001_SW":
    from DJF_0001_SW.conf import *
elif current_project == "PL_0001_VP":
    from PL_0001_VP.conf import *
elif current_project == "PL_0004_SW":
    from PL_0004_SW.conf import *
elif current_project == "DD_0002_OHU":
    from DD_0002_OHU.conf import *
elif current_project == "UM_0001_UserManual":
    from UM_0001_UserManual.conf import *
elif current_project == "PL_0002_ENV":
    from PL_0002_ENV.conf import *
elif current_project == "SW_0002_CNN":
    from SW_0002_CNN.conf import *
elif current_project == "SW_0004_ASW":
    from SW_0004_ASW.conf import *
elif current_project == "AD_0001_SW":
    from AD_0001_SW.conf import *
elif current_project == "RP_0001_SCAR":
    from RP_0001_SCAR.conf import *
elif current_project == "DJF_0002_ALG":
    from DJF_0002_ALG.conf import *
elif current_project == "PolarFire_Xilinx_Tradeoff":
    from PolarFire_Xilinx_Tradeoff.conf import *


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["multiproject", "sphinx_lmodoc.ext", "sphinx_needs",
              "sphinxcontrib.plantuml", "sphinx_simplepdf",
              "sphinx.ext.imgmath"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = 'index'

today_fmt = '%d/%m/%y'

numfig = True
numfig_format = {
    'code-block': 'Listing %s',
    'figure': 'Fig. %s',
    'section': 'Section %s',
    'table': 'Table %s',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for simplepdf

simplepdf_theme = "lmodoc_theme"
simplepdf_vars = {
    "top-center-content": f'"{project_label}"',
    "top-right-content": f'"{project_id}-{document_id} i{release}"',
}
simplepdf_file_name = f"{project_id}-{document_id} i{release} - {project}.pdf"

# -- sphinx-needs configuration
needs_types = [dict(directive="req", title="Requirement", prefix="", color="#BFD8D2", style="node"),
               dict(directive="spec", title="Specification", prefix="", color="#FEDCD2", style="node"),
               dict(directive="impl", title="Implementation", prefix="", color="#DF744A", style="node"),
               dict(directive="test", title="Test Case", prefix="", color="#DCB239", style="node"),
               # Kept for backwards compatibility
               dict(directive="need", title="Need", prefix="", color="#9856a5", style="node"),
               # Added for SCAR
               dict(directive="fail", title="Failure mode", prefix="", color="#9856a5", style="node")
           ]
needs_extra_options = ['verif','data_type','default','range','function',
                       'effect','category']
needs_layouts = conf_helper.get_needs_layouts()
needs_default_layout = 'my_layout'
needs_build_json = True
needs_title_optional = True
