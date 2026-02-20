# Source Documents Directory

Place your data files here following the **domain folder structure** and **file naming convention**.

## Directory Structure

```
source_documents/
├── semiconductor/                # Domain subdirectory
│   ├── Semi_Spec_v4.2_Public.txt
│   ├── Semi_Email_v1_Internal.txt
│   └── ...
├── medical/                      # Domain subdirectory (placeholder)
│   └── README.md
├── _template_data.json           # JSON template (backward compatible)
└── README.md                     # This file
```

## File Naming Convention

**Format**: `Domain_Type_Version_Access.txt`

| Part | Allowed Values | Description |
|------|---------------|-------------|
| `Domain` | `Semi`, `Med`, or custom | Short domain code |
| `Type` | `Spec`, `Email`, `SOP`, `DB`, `DM`, `Doc` | Document type |
| `Version` | `v1`, `v2`, `v1.0`, etc. | Version identifier |
| `Access` | `Public`, `Internal`, `Confidential` | RBAC access level |

## RBAC Auto-Tagging

The ingestion pipeline automatically tags documents based on type:

| Type | Default Access | Mapped Source |
|------|---------------|---------------|
| `Spec` | `public` | `datasheet` |
| `Email` | `internal_only` | `email` |
| `SOP` | `public` | `sop` |
| `DB` | `confidential` | `db_info` |
| `DM` | `internal_only` | `email` |
| `Doc` | `public` | `document` |

The `Access` part of the filename **overrides** the default. For example:
- `Semi_Email_v1_Confidential.txt` → `confidential` (override)
- `Semi_Email_v1_Internal.txt` → `internal_only` (default)

## Adding Data

```bash
# 1. Create domain folder (if new)
mkdir source_documents/my_domain/

# 2. Add your files following the naming convention
# Example: Semi_SOP_v1_Public.txt

# 3. Re-run ingestion
python ingestion.py
```

## Content Guidelines

- **Email/DM files**: Include date and decision-maker information for Case Agent conflict detection
- **Spec files**: Include version numbers and technical parameters
- **DB files**: Use table format for structured data
- **SOP files**: Include procedure numbers and classification

## JSON Support (Backward Compatible)

You can also place `.json` files in the root `source_documents/` directory.
See `_template_data.json` for the format. Files starting with `_` are skipped.
