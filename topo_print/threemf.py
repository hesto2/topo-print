import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import trimesh

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
COLOR_GROUP_ID = 1

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _color_group(parts):
    colors = "".join(f"<m:color color={quoteattr(color.upper() + 'FF')}/>" for _, color, _ in parts)
    return f'<m:colorgroup id="{COLOR_GROUP_ID}">{colors}</m:colorgroup>'


def _mesh_object(object_id, name, color_index, mesh: trimesh.Trimesh):
    vertices = "".join('<vertex x="%.4f" y="%.4f" z="%.4f"/>' % tuple(v) for v in mesh.vertices)
    triangles = "".join('<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(f) for f in mesh.faces)
    return (
        f'<object id="{object_id}" name={quoteattr(name)} type="model" '
        f'pid="{COLOR_GROUP_ID}" pindex="{color_index}"><mesh>'
        f"<vertices>{vertices}</vertices><triangles>{triangles}</triangles>"
        "</mesh></object>"
    )


def write_colored_3mf(path: Path, name, parts, plate_center_mm, size_mm):
    """Standard 3MF with one top-level object per color, all placed together.

    Colors use the Materials extension's m:colorgroup, the form Bambu Studio 2.8 reads: it maps each
    object's color to its own filament and offers to merge the objects into one multi-part object.
    parts: list of (part name, hex color, mesh).
    """
    object_ids = range(COLOR_GROUP_ID + 1, COLOR_GROUP_ID + 1 + len(parts))
    meshes = "".join(
        _mesh_object(object_id, f"{name}-{part_name}", k, mesh)
        for k, (object_id, (part_name, _, mesh)) in enumerate(zip(object_ids, parts))
    )
    offset = plate_center_mm - size_mm / 2
    items = "".join(
        f'<item objectid="{object_id}" transform="1 0 0 0 1 0 0 0 1 {offset} {offset} 0"/>' for object_id in object_ids
    )
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}" xmlns:m="{MATERIAL_NS}">'
        f'<metadata name="Title">{escape(name)}</metadata>'
        f"<resources>{_color_group(parts)}{meshes}</resources>"
        f"<build>{items}</build>"
        "</model>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("3D/3dmodel.model", model)
