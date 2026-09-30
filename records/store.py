"""The only module that writes study files under the records root.

Layout: <root>/files/<patient_id>/<study_id>/<kind><ext>. Files are staged in
<root>/files/.staging/ first and moved into place only as part of saving the
study, so a failed save never leaves a half-populated study folder.
"""
import os
import shutil
import uuid

STAGING = ".staging"


def files_root(root):
    return os.path.join(root, "files")


def stage(root, files):
    """Copy {kind: source_path} into a fresh staging folder.

    Returns (staging_dir, {kind: file_name}). Raises FileNotFoundError, after
    removing the staging folder, if any source is missing.
    """
    staging = os.path.join(files_root(root), STAGING, uuid.uuid4().hex)
    os.makedirs(staging)
    names = {}
    try:
        for kind, src in files.items():
            if not src or not os.path.isfile(src):
                raise FileNotFoundError(f"study file missing: {kind} ({src})")
            name = kind + os.path.splitext(src)[1].lower()
            shutil.copyfile(src, os.path.join(staging, name))
            names[kind] = name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging, names


def rel_path(patient_id, study_id, name):
    return os.path.join("files", patient_id, study_id, name)


def commit(root, staging, patient_id, study_id):
    final = os.path.join(files_root(root), patient_id, study_id)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(staging, final)
    return final


def discard(path):
    if path:
        shutil.rmtree(path, ignore_errors=True)


def add_file(root, patient_id, study_id, kind, data, ext):
    """Write one more file (e.g. the generated PDF) into a saved study."""
    folder = os.path.join(files_root(root), patient_id, study_id)
    os.makedirs(folder, exist_ok=True)
    name = kind + ext
    with open(os.path.join(folder, name), "wb") as fh:
        fh.write(data)
    return rel_path(patient_id, study_id, name)


def absolute(root, rel):
    """Resolve a stored relative path, refusing anything outside the store."""
    base = os.path.realpath(files_root(root))
    path = os.path.realpath(os.path.join(root, rel))
    if not path.startswith(base + os.sep):
        raise ValueError("path outside the record store")
    return path
