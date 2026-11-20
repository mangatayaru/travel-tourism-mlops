# Phase 1 Audit Package

This is an **audit-only** package. It does not contain replacement application code and does not retrain or overwrite your models.

## Where to extract it

Extract the ZIP anywhere convenient. Then copy the `phase1_audit` folder into your existing `travel-tourism-mlops` project root, or run the script by providing the project root path.

## Run in PowerShell

From the existing project root:

```powershell
py -3.11 .\phase1_audit\scripts\phase1_audit.py --project-root .
```

If your Python launcher does not have Python 3.11, use:

```powershell
python .\phase1_audit\scripts\phase1_audit.py --project-root .
```

The script writes two new files under `phase1_audit\reports\`:
- `phase1_audit.json`
- `phase1_audit.txt`

It only reads project files and writes its reports inside the Phase 1 package folder.

## Expected result

The terminal should end with an `Audit completed` message and the report paths. Review the warnings; do not treat a warning as proof of a defect.

## Important

The archive and separately uploaded files showed the baseline project structure and model metrics. This script rechecks files on your own machine. It does not test Docker, Jenkins, Airflow, Kubernetes, or API runtime behavior.
