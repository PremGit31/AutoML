"""Semi-structured XML inspection, cleaning, validation, and export dashboard."""

import copy
import hashlib
import io
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Semi-Structured Data (.XML)",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_name(name):
    """Return a readable XML name without its namespace URI."""
    return name.rsplit("}", 1)[-1] if "}" in name else name


def element_text(element):
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def element_path(parent_path, tag, sibling_index):
    segment = f"{local_name(tag)}[{sibling_index}]"
    return f"{parent_path}/{segment}"


def flatten_elements(root, source=""):
    """Create an element-level table while retaining hierarchy and attributes."""
    rows = []

    def visit(element, parent_path="", sibling_index=1, depth=0):
        tag = local_name(element.tag)
        path = element_path(parent_path, element.tag, sibling_index)
        attrs = {local_name(key): value for key, value in element.attrib.items()}
        namespace = (
            element.tag[1:].split("}", 1)[0]
            if isinstance(element.tag, str) and element.tag.startswith("{")
            else ""
        )
        rows.append({
            "source": source,
            "path": path,
            "parent_path": parent_path or "/",
            "tag": tag,
            "namespace": namespace,
            "depth": depth,
            "attributes": json.dumps(attrs, ensure_ascii=False),
            "direct_text": (element.text or "").strip(),
            "content_text": element_text(element),
        })
        tag_counts = {}
        for child in element:
            child_tag = local_name(child.tag)
            tag_counts[child_tag] = tag_counts.get(child_tag, 0) + 1
            visit(child, path, tag_counts[child_tag], depth + 1)

    visit(root)
    return rows


def serialize_xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def normalized_text(text):
    return " ".join((text or "").split())


def subtree_signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        normalized_text(element.text),
        tuple(
            (subtree_signature(child), normalized_text(child.tail))
            for child in element
        ),
    )


def find_direct_text(element, tag):
    for child in element:
        if local_name(child.tag) == tag:
            return element_text(child)
    return ""


def clean_xml(root, options):
    """Apply selected cleaning operations to a fresh tree copy."""
    cleaned = copy.deepcopy(root)
    counts = {
        "duplicates": 0,
        "removed_empty": 0,
        "removed_tag": 0,
        "removed_incomplete": 0,
        "removed_attribute": 0,
        "filled": 0,
        "text_changes": 0,
        "invalid_dates": [],
        "invalid_numbers": [],
    }

    if options["normalize_whitespace"] or options["case"] != "Preserve":
        case_func = {
            "Preserve": lambda value: value,
            "lowercase": str.lower,
            "UPPERCASE": str.upper,
            "Title Case": str.title,
        }[options["case"]]

        def normalize_tree(node):
            for field in ("text", "tail"):
                value = getattr(node, field)
                if value is None:
                    continue
                new_value = " ".join(value.split()) if options["normalize_whitespace"] else value
                new_value = case_func(new_value)
                if value != new_value:
                    counts["text_changes"] += 1
                    setattr(node, field, new_value)
            for key, value in list(node.attrib.items()):
                new_value = case_func(value)
                if value != new_value:
                    counts["text_changes"] += 1
                    node.set(key, new_value)
            for child in node:
                normalize_tree(child)

        normalize_tree(cleaned)

    def remove_tags(node):
        for child in list(node):
            if local_name(child.tag) in options["remove_tags"]:
                node.remove(child)
                counts["removed_tag"] += 1
            else:
                remove_tags(child)

    if options["remove_tags"]:
        if local_name(cleaned.tag) in options["remove_tags"]:
            raise ValueError("The root element cannot be removed.")
        remove_tags(cleaned)

    if options["remove_attributes"]:
        for node in cleaned.iter():
            for attr in list(node.attrib):
                if local_name(attr) in options["remove_attributes"]:
                    del node.attrib[attr]
                    counts["removed_attribute"] += 1

    if options["fill_empty_tags"]:
        for node in cleaned.iter():
            if local_name(node.tag) in options["fill_empty_tags"] and not element_text(node):
                node.text = options["fill_value"]
                counts["filled"] += 1

    if options["date_tags"]:
        for node in cleaned.iter():
            if local_name(node.tag) not in options["date_tags"] or not normalized_text(node.text):
                continue
            try:
                value = datetime.strptime(node.text.strip(), options["date_input_format"])
                converted = value.strftime(options["date_output_format"])
                if converted != node.text:
                    node.text = converted
                    counts["text_changes"] += 1
            except ValueError:
                counts["invalid_dates"].append(
                    f"{local_name(node.tag)}: {node.text.strip()}"
                )

    if options["number_tags"]:
        quantizer = Decimal("1").scaleb(-options["number_decimals"])
        for node in cleaned.iter():
            if local_name(node.tag) not in options["number_tags"] or not normalized_text(node.text):
                continue
            try:
                value = Decimal(node.text.strip()).quantize(quantizer)
                converted = format(value, f".{options['number_decimals']}f")
                if converted != node.text:
                    node.text = converted
                    counts["text_changes"] += 1
            except InvalidOperation:
                counts["invalid_numbers"].append(
                    f"{local_name(node.tag)}: {node.text.strip()}"
                )

    if options["incomplete_tag"] and options["incomplete_field"]:
        if local_name(cleaned.tag) == options["incomplete_tag"]:
            raise ValueError("The root element cannot be removed as an incomplete record.")

        def remove_incomplete(node):
            for child in list(node):
                if local_name(child.tag) == options["incomplete_tag"]:
                    field = options["incomplete_field"]
                    if field.startswith("@"):
                        value = next(
                            (
                                attr_value for attr, attr_value in child.attrib.items()
                                if local_name(attr) == field[1:]
                            ),
                            "",
                        )
                    else:
                        value = find_direct_text(child, field)
                    if not normalized_text(value):
                        node.remove(child)
                        counts["removed_incomplete"] += 1
                        continue
                remove_incomplete(child)

        remove_incomplete(cleaned)

    if options["remove_empty"]:
        def remove_empty_children(node):
            for child in list(node):
                remove_empty_children(child)
                if not child.attrib and not list(child) and not normalized_text(child.text):
                    node.remove(child)
                    counts["removed_empty"] += 1

        remove_empty_children(cleaned)

    if options["dedupe_exact"] or options["dedupe_key_tag"]:
        def deduplicate(node):
            seen_subtrees = set()
            seen_keys = set()
            for child in list(node):
                deduplicate(child)
                remove_child = False
                if options["dedupe_exact"]:
                    signature = subtree_signature(child)
                    if signature in seen_subtrees:
                        remove_child = True
                    else:
                        seen_subtrees.add(signature)
                key_tag = options["dedupe_key_tag"]
                if not remove_child and key_tag and local_name(child.tag) == key_tag:
                    key_name = options["dedupe_key"]
                    if key_name.startswith("@"):
                        key_value = next(
                            (
                                value for attr, value in child.attrib.items()
                                if local_name(attr) == key_name[1:]
                            ),
                            "",
                        )
                    else:
                        key_value = find_direct_text(child, key_name)
                    key_value = normalized_text(key_value)
                    if key_value:
                        identity = (key_tag, key_value)
                        if identity in seen_keys:
                            remove_child = True
                        else:
                            seen_keys.add(identity)
                if remove_child:
                    node.remove(child)
                    counts["duplicates"] += 1

        deduplicate(cleaned)

    if options["rename_tag_from"] and options["rename_tag_to"]:
        for node in cleaned.iter():
            if local_name(node.tag) == options["rename_tag_from"]:
                if isinstance(node.tag, str) and node.tag.startswith("{"):
                    namespace = node.tag[1:].split("}", 1)[0]
                    node.tag = f"{{{namespace}}}{options['rename_tag_to']}"
                else:
                    node.tag = options["rename_tag_to"]

    if options["rename_attribute_from"] and options["rename_attribute_to"]:
        for node in cleaned.iter():
            source_attr = next(
                (
                    attr for attr in node.attrib
                    if local_name(attr) == options["rename_attribute_from"]
                ),
                None,
            )
            if source_attr is None:
                continue
            if local_name(source_attr) == options["rename_attribute_to"]:
                continue
            if any(
                local_name(attr) == options["rename_attribute_to"]
                for attr in node.attrib
            ):
                raise ValueError(
                    f"Cannot rename attribute `{options['rename_attribute_from']}` "
                    f"to `{options['rename_attribute_to']}` because the target already exists."
                )
            value = node.attrib.pop(source_attr)
            if source_attr.startswith("{"):
                namespace = source_attr[1:].split("}", 1)[0]
                target_attr = f"{{{namespace}}}{options['rename_attribute_to']}"
            else:
                target_attr = options["rename_attribute_to"]
            node.set(target_attr, value)

    return cleaned, counts


def relative_field_values(record, selected_field):
    """Extract values matching a relative tag path or @attribute path."""
    parts = selected_field.split("/")
    values = []

    def walk(node, depth):
        if depth == len(parts):
            value = node.attrib.get(parts[-1][1:], "") if parts[-1].startswith("@") else element_text(node)
            if value:
                values.append(value)
            return
        part = parts[depth]
        if part.startswith("@"):
            value = node.attrib.get(part[1:], "")
            if value:
                values.append(value)
        else:
            for child in node:
                if local_name(child.tag) == part:
                    walk(child, depth + 1)

    if parts[0].startswith("@"):
        value = record.attrib.get(parts[0][1:], "")
        if value:
            values.append(value)
    else:
        for child in record:
            if local_name(child.tag) == parts[0]:
                walk(child, 1)
    return " | ".join(values)


def record_field_options(root, record_tag):
    fields = set()

    def walk(node, prefix="", is_record=False):
        for attr in node.attrib:
            fields.add(f"{prefix}@{local_name(attr)}")
        for child in node:
            tag = local_name(child.tag)
            path = f"{prefix}{tag}"
            fields.add(path)
            walk(child, f"{path}/")

    for record in root.iter():
        if local_name(record.tag) == record_tag:
            for attr in record.attrib:
                fields.add(f"@{local_name(attr)}")
            walk(record)
    return sorted(fields)


def flatten_records(root, record_tag, fields):
    records = [node for node in root.iter() if local_name(node.tag) == record_tag]
    rows = []
    for index, record in enumerate(records, 1):
        row = {"record_number": index}
        for field in fields:
            row[field] = relative_field_values(record, field)
        rows.append(row)
    return pd.DataFrame(rows), records


def compare_xml_trees(left_rows, right_rows):
    left = {
        row["path"]: (row["tag"], row["attributes"], row["direct_text"])
        for row in left_rows
    }
    right = {
        row["path"]: (row["tag"], row["attributes"], row["direct_text"])
        for row in right_rows
    }
    paths = sorted(set(left) | set(right))
    differences = []
    for path in paths:
        if path not in left:
            differences.append({"path": path, "change": "Added", "left": "", "right": str(right[path])})
        elif path not in right:
            differences.append({"path": path, "change": "Removed", "left": str(left[path]), "right": ""})
        elif left[path] != right[path]:
            differences.append({"path": path, "change": "Changed", "left": str(left[path]), "right": str(right[path])})
    return pd.DataFrame(differences, columns=["path", "change", "left", "right"])


def validate_schema(xml_bytes, schema_bytes, schema_type):
    """Validate against XSD or DTD with external network/entity access disabled."""
    try:
        from lxml import etree
    except ImportError as exc:
        raise RuntimeError(
            "Schema validation requires lxml. Install it with `pip install lxml`."
        ) from exc

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        remove_comments=False,
    )
    document = etree.fromstring(xml_bytes, parser)
    if schema_type == "XSD":
        schema_document = etree.fromstring(schema_bytes, parser)
        schema = etree.XMLSchema(schema_document)
        valid = schema.validate(document)
        messages = [entry.message for entry in schema.error_log]
    else:
        dtd = etree.DTD(io.BytesIO(schema_bytes))
        valid = dtd.validate(document)
        messages = [entry.message for entry in dtd.error_log]
    return valid, messages


def make_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def dataframe_to_xlsx(dataframe):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="XML data")
    return buffer.getvalue()


def update_processed(filename, digest, root):
    st.session_state.setdefault("xml_processed", {})[filename] = {
        "digest": digest,
        "content": serialize_xml(root),
    }


st.title("🧩 Semi-Structured Data")
st.caption(
    "Upload XML (.xml) files to inspect, clean, validate, transform, compare, analyze, and export them."
)

uploaded_files = st.file_uploader(
    "Upload XML file(s)",
    type=["xml"],
    accept_multiple_files=True,
    help="Select one or more .xml documents. Upload an XSD or DTD separately in the Validation tab.",
    key="xml_documents",
)

if not uploaded_files:
    st.info("Upload one or more .xml files to get started.")
    st.stop()

documents = {}
parse_errors = {}
for uploaded_file in uploaded_files:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        parse_errors[uploaded_file.name] = str(exc)
        continue
    documents[uploaded_file.name] = {
        "raw": content,
        "digest": digest,
        "root": root,
    }
    processed = st.session_state.get("xml_processed", {}).get(uploaded_file.name)
    if processed and processed["digest"] != digest:
        del st.session_state["xml_processed"][uploaded_file.name]

for filename, error in parse_errors.items():
    st.error(f"**{filename}** is not well-formed XML: {error}")

if not documents:
    st.stop()

names = list(documents)
tabs = st.tabs([
    "🔎 Structure",
    "🧹 Cleaning",
    "✅ Validation",
    "🔄 Transform",
    "📊 Analyze & Compare",
    "📤 Export",
])

with tabs[0]:
    st.subheader("XML structure")
    selected_structure = st.selectbox("Choose a document", names, key="xml_structure_doc")
    structure_doc = documents[selected_structure]
    structure_rows = flatten_elements(
        structure_doc["root"], source=selected_structure
    )
    structure_df = pd.DataFrame(structure_rows)
    root_tag = local_name(structure_doc["root"].tag)
    elements, attributes = st.columns(2)
    elements.metric("Elements", len(structure_rows))
    attributes.metric(
        "Attributes",
        sum(len(node.attrib) for node in structure_doc["root"].iter()),
    )
    st.caption(f"Root element: `{root_tag}`. Well-formedness is checked when the file is uploaded.")
    st.dataframe(structure_df, use_container_width=True, hide_index=True)

    with st.expander("Find elements by tag, attribute, or text"):
        tag_options = sorted({row["tag"] for row in structure_rows})
        search_tag = st.selectbox("Tag (optional)", ["All"] + tag_options, key="xml_find_tag")
        search_attribute = st.text_input("Attribute name (optional)", key="xml_find_attr")
        search_attribute_value = st.text_input("Attribute value contains (optional)", key="xml_find_attr_value")
        search_text = st.text_input("Text contains (optional)", key="xml_find_text")
        matches = []
        for node_row in structure_rows:
            if search_tag != "All" and node_row["tag"] != search_tag:
                continue
            node_attrs = json.loads(node_row["attributes"])
            if search_attribute and search_attribute not in node_attrs:
                continue
            if search_attribute_value and search_attribute_value.casefold() not in str(
                node_attrs.get(search_attribute, "")
            ).casefold():
                continue
            if search_text and search_text.casefold() not in node_row["content_text"].casefold():
                continue
            matches.append(node_row)
        st.caption(f"{len(matches):,} matching element(s)")
        st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)

    with st.expander("Run an XPath query"):
        xpath_query = st.text_input(
            "XPath expression",
            value=".//*",
            help="Uses ElementTree's supported XPath subset.",
            key="xml_xpath_query",
        )
        if st.button("Run XPath", key="xml_run_xpath"):
            try:
                xpath_matches = structure_doc["root"].findall(xpath_query)
                if xpath_matches:
                    xpath_rows = []
                    for match in xpath_matches:
                        xpath_rows.extend(
                            flatten_elements(match, source=selected_structure)
                        )
                    st.caption(f"{len(xpath_matches):,} matching element(s)")
                    st.dataframe(
                        pd.DataFrame(xpath_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("The XPath query matched no elements.")
            except (SyntaxError, KeyError, TypeError) as exc:
                st.error(f"Invalid or unsupported XPath expression: {exc}")

with tabs[1]:
    st.subheader("Clean and prepare an XML document")
    clean_name = st.selectbox("Document to clean", names, key="xml_clean_doc")
    clean_doc = documents[clean_name]
    original_root = clean_doc["root"]
    tag_options = sorted({local_name(node.tag) for node in original_root.iter()})
    attribute_options = sorted({
        local_name(attr) for node in original_root.iter() for attr in node.attrib
    })
    key_candidates = sorted({
        f"@{local_name(attr)}" for node in original_root.iter() for attr in node.attrib
    } | {local_name(node.tag) for node in original_root.iter()})
    incomplete_tag_options = ["(disabled)"] + tag_options

    st.markdown("**Text and missing-value handling**")
    clean_col1, clean_col2 = st.columns(2)
    normalize_whitespace = clean_col1.checkbox(
        "Trim text and collapse repeated whitespace", value=True, key="xml_clean_ws"
    )
    case_choice = clean_col2.selectbox(
        "Text case", ["Preserve", "lowercase", "UPPERCASE", "Title Case"],
        key="xml_clean_case",
    )
    fill_tags = st.multiselect(
        "Fill selected empty elements",
        tag_options,
        key="xml_fill_tags",
    )
    fill_value = st.text_input(
        "Replacement text for empty elements", value="N/A", key="xml_fill_value"
    )
    with st.expander("Standardize dates and numbers"):
        date_tags = st.multiselect(
            "Date element tags", tag_options, key="xml_date_tags"
        )
        date_format_col1, date_format_col2 = st.columns(2)
        date_input_format = date_format_col1.text_input(
            "Current date format (Python strptime format)",
            value="%d/%m/%Y",
            key="xml_date_input_format",
        )
        date_output_format = date_format_col2.selectbox(
            "Output date format",
            ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"],
            key="xml_date_output_format",
        )
        number_tags = st.multiselect(
            "Number element tags", tag_options, key="xml_number_tags"
        )
        number_decimals = st.number_input(
            "Decimal places", min_value=0, max_value=12, value=2,
            key="xml_number_decimals",
        )
    remove_empty = st.checkbox(
        "Remove empty leaf elements (those with no attributes, children, or text)",
        key="xml_remove_empty",
    )
    incomplete_record_tag = st.selectbox(
        "Remove records missing a required field",
        incomplete_tag_options,
        key="xml_incomplete_tag",
    )
    incomplete_fields = (
        sorted({
            f"@{local_name(attr)}"
            for node in original_root.iter()
            if local_name(node.tag) == incomplete_record_tag
            for attr in node.attrib
        } | {
            local_name(child.tag)
            for node in original_root.iter()
            if local_name(node.tag) == incomplete_record_tag
            for child in node
        })
        if incomplete_record_tag != "(disabled)"
        else []
    )
    incomplete_field = (
        st.selectbox(
            "Required attribute or direct child field",
            incomplete_fields,
            key="xml_incomplete_field",
        )
        if incomplete_fields
        else ""
    )

    st.markdown("**Remove duplicates**")
    exact_col, key_col = st.columns(2)
    dedupe_exact = exact_col.checkbox(
        "Remove exact duplicate sibling subtrees", key="xml_dedupe_exact"
    )
    dedupe_key_tag = key_col.selectbox(
        "Or remove duplicate sibling records by tag/key",
        ["(disabled)"] + tag_options,
        key="xml_dedupe_tag",
    )
    dedupe_key = st.selectbox(
        "Key attribute or child tag",
        key_candidates or ["(none)"],
        key="xml_dedupe_key",
    )
    st.caption(
        "Exact duplicates are compared among siblings. Key-based deduplication keeps the first "
        "sibling record with a non-empty key."
    )

    remove_tags = st.multiselect(
        "Remove all elements with these tags", tag_options, key="xml_remove_tags"
    )
    remove_attributes = st.multiselect(
        "Remove these attributes from all elements",
        attribute_options,
        key="xml_remove_attributes",
    )
    rename_col1, rename_col2 = st.columns(2)
    rename_tag_from = rename_col1.selectbox(
        "Rename element tag (optional)",
        ["(disabled)"] + tag_options,
        key="xml_rename_tag_from",
    )
    rename_tag_to = rename_col2.text_input(
        "New element name",
        key="xml_rename_tag_to",
    )
    rename_attr_col1, rename_attr_col2 = st.columns(2)
    rename_attribute_from = rename_attr_col1.selectbox(
        "Rename attribute (optional)",
        ["(disabled)"] + attribute_options,
        key="xml_rename_attr_from",
    )
    rename_attribute_to = rename_attr_col2.text_input(
        "New attribute name",
        key="xml_rename_attr_to",
    )
    clean_options = {
        "normalize_whitespace": normalize_whitespace,
        "case": case_choice,
        "fill_empty_tags": set(fill_tags),
        "fill_value": fill_value,
        "date_tags": set(date_tags),
        "date_input_format": date_input_format,
        "date_output_format": date_output_format,
        "number_tags": set(number_tags),
        "number_decimals": int(number_decimals),
        "remove_empty": remove_empty,
        "incomplete_tag": "" if incomplete_record_tag == "(disabled)" else incomplete_record_tag,
        "incomplete_field": incomplete_field,
        "dedupe_exact": dedupe_exact,
        "dedupe_key_tag": "" if dedupe_key_tag == "(disabled)" else dedupe_key_tag,
        "dedupe_key": dedupe_key,
        "remove_tags": set(remove_tags),
        "remove_attributes": set(remove_attributes),
        "rename_tag_from": "" if rename_tag_from == "(disabled)" else rename_tag_from,
        "rename_tag_to": rename_tag_to.strip(),
        "rename_attribute_from": "" if rename_attribute_from == "(disabled)" else rename_attribute_from,
        "rename_attribute_to": rename_attribute_to.strip(),
    }
    try:
        for new_name in (clean_options["rename_tag_to"], clean_options["rename_attribute_to"]):
            if new_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", new_name):
                raise ValueError(
                    f"`{new_name}` is not a valid unprefixed XML name. "
                    "Use letters, digits, underscores, periods, or hyphens; start with a letter or underscore."
                )
        preview_root, preview_counts = clean_xml(original_root, clean_options)
        preview_rows = flatten_elements(preview_root, source=clean_name)
        st.markdown("**Preview**")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Elements before", len(list(original_root.iter())))
        metric_columns[1].metric("Elements after", len(preview_rows))
        metric_columns[2].metric("Duplicates removed", preview_counts["duplicates"])
        metric_columns[3].metric(
            "Other removals",
            preview_counts["removed_empty"] + preview_counts["removed_tag"]
            + preview_counts["removed_incomplete"],
        )
        if preview_counts["filled"]:
            st.caption(f"Filled {preview_counts['filled']} empty element(s).")
        if preview_counts["removed_attribute"]:
            st.caption(f"Removed {preview_counts['removed_attribute']} attribute(s).")
        if preview_counts["text_changes"]:
            st.caption(f"Standardized {preview_counts['text_changes']} text or attribute value(s).")
        if preview_counts["invalid_dates"]:
            st.warning(
                f"Could not parse {len(preview_counts['invalid_dates'])} selected date value(s): "
                + ", ".join(preview_counts["invalid_dates"][:10])
            )
        if preview_counts["invalid_numbers"]:
            st.warning(
                f"Could not parse {len(preview_counts['invalid_numbers'])} selected number value(s): "
                + ", ".join(preview_counts["invalid_numbers"][:10])
            )
        with st.expander("Preview cleaned element table"):
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        if st.button("Apply cleaning to this document", key="xml_apply_cleaning"):
            update_processed(clean_name, clean_doc["digest"], preview_root)
            st.success(f"Cleaning applied to `{clean_name}`.")
            st.rerun()
    except ValueError as exc:
        st.error(str(exc))

    processed = st.session_state.get("xml_processed", {}).get(clean_name)
    if processed and processed["digest"] == clean_doc["digest"]:
        st.success(f"A cleaned version of `{clean_name}` is available for validation and export.")

with tabs[2]:
    st.subheader("Well-formedness and schema validation")
    validation_name = st.selectbox("XML document", names, key="xml_validation_doc")
    validation_doc = documents[validation_name]
    use_cleaned = st.checkbox(
        "Validate the cleaned version when available",
        value=False,
        key="xml_validate_cleaned",
    )
    stored_processed = st.session_state.get("xml_processed", {}).get(validation_name)
    validation_bytes = (
        stored_processed["content"]
        if use_cleaned and stored_processed
        and stored_processed["digest"] == validation_doc["digest"]
        else validation_doc["raw"]
    )
    try:
        ET.fromstring(validation_bytes)
        st.success("XML is well-formed.")
    except ET.ParseError as exc:
        st.error(f"XML is not well-formed: {exc}")

    schema_kind = st.selectbox("Schema type", ["XSD", "DTD"], key="xml_schema_kind")
    schema_upload = st.file_uploader(
        f"Upload {schema_kind} schema",
        type=["xsd"] if schema_kind == "XSD" else ["dtd"],
        key="xml_schema_file",
    )
    if schema_upload:
        if st.button("Validate against schema", key="xml_validate_schema"):
            try:
                valid, validation_messages = validate_schema(
                    validation_bytes, schema_upload.getvalue(), schema_kind
                )
                if valid:
                    st.success(f"Document is valid against the supplied {schema_kind} schema.")
                else:
                    st.error(f"Document does not satisfy the supplied {schema_kind} schema.")
                    for message in validation_messages:
                        st.write(f"- {message}")
            except Exception as exc:
                st.error(f"{schema_kind} validation failed: {exc}")
    else:
        st.info("Upload an XSD or DTD file to run schema validation.")

with tabs[3]:
    st.subheader("Flatten repeated XML records into a table")
    transform_name = st.selectbox("Source document", names, key="xml_transform_doc")
    transform_doc = documents[transform_name]
    transform_root = transform_doc["root"]
    record_tags = sorted({local_name(node.tag) for node in transform_root.iter()})
    tag_counts = {}
    tag_depths = {}
    for row in flatten_elements(transform_root):
        tag_counts[row["tag"]] = tag_counts.get(row["tag"], 0) + 1
        tag_depths[row["tag"]] = min(
            tag_depths.get(row["tag"], row["depth"]),
            row["depth"],
        )
    default_record_tag = min(
        record_tags,
        key=lambda tag: (-tag_counts[tag], tag_depths[tag], tag),
    )
    record_tag = st.selectbox(
        "Element representing one record",
        record_tags,
        index=record_tags.index(default_record_tag),
        key="xml_record_tag_v2",
    )
    available_fields = record_field_options(transform_root, record_tag)
    selected_fields = st.multiselect(
        "Fields to extract (tags and @attributes)",
        available_fields,
        default=available_fields,
        key="xml_record_fields",
    )
    flattened_df, selected_records = flatten_records(
        transform_root, record_tag, selected_fields
    )
    st.caption(f"{len(selected_records):,} record(s) found.")
    filter_col1, filter_col2 = st.columns(2)
    filter_field = filter_col1.selectbox(
        "Filter field (optional)",
        ["(none)"] + selected_fields,
        key="xml_filter_field",
    )
    filter_value = filter_col2.text_input(
        "Keep records where the field contains",
        key="xml_filter_value",
    )
    filtered_df = flattened_df
    if filter_field != "(none)" and filter_value:
        filtered_df = flattened_df[
            flattened_df[filter_field].astype(str).str.contains(
                filter_value, case=False, regex=False, na=False
            )
        ]
    st.caption(f"{len(filtered_df):,} record(s) after filtering.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    if selected_fields:
        group_field = st.selectbox(
            "Summarize record counts by field (optional)",
            ["(none)"] + selected_fields,
            key="xml_group_field",
        )
        if group_field != "(none)" and not filtered_df.empty:
            group_summary = (
                filtered_df.groupby(group_field, dropna=False)
                .size()
                .reset_index(name="records")
                .sort_values("records", ascending=False)
            )
            st.dataframe(group_summary, use_container_width=True, hide_index=True)
    if not flattened_df.empty:
        record_downloads = st.columns(3)
        record_downloads[0].download_button(
            "⬇ Filtered records as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{transform_name.rsplit('.', 1)[0]}_records.csv",
            mime="text/csv",
            key="xml_flat_csv",
        )
        record_downloads[1].download_button(
            "⬇ Filtered records as JSON",
            data=filtered_df.to_json(
                orient="records", force_ascii=False, indent=2
            ).encode("utf-8"),
            file_name=f"{transform_name.rsplit('.', 1)[0]}_records.json",
            mime="application/json",
            key="xml_flat_json",
        )
        record_downloads[2].download_button(
            "⬇ Filtered records as Excel",
            data=dataframe_to_xlsx(filtered_df),
            file_name=f"{transform_name.rsplit('.', 1)[0]}_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="xml_flat_xlsx",
        )

    st.divider()
    st.subheader("Combine compatible XML files")
    merge_sources = st.multiselect(
        "XML files to combine", names, default=names if len(names) > 1 else names,
        key="xml_merge_sources",
    )
    if st.button("Prepare combined XML", key="xml_prepare_merge"):
        if len(merge_sources) < 2:
            st.warning("Choose at least two XML files to combine.")
        else:
            selected_roots = [documents[name]["root"] for name in merge_sources]
            root_tags = {node.tag for node in selected_roots}
            if len(root_tags) != 1:
                st.error("The selected documents have different root elements and cannot be combined.")
            elif any(
                node.attrib != selected_roots[0].attrib
                or normalized_text(node.text) != normalized_text(selected_roots[0].text)
                for node in selected_roots[1:]
            ):
                st.error(
                    "The selected documents have different root attributes or root text. "
                    "They cannot be combined without discarding data."
                )
            else:
                merged_root = copy.deepcopy(selected_roots[0])
                for extra_root in selected_roots[1:]:
                    merged_root.extend(copy.deepcopy(list(extra_root)))
                st.session_state["xml_combined"] = serialize_xml(merged_root)
                st.success(f"Combined {len(merge_sources)} XML files under `{local_name(merged_root.tag)}`.")

with tabs[4]:
    st.subheader("Analyze element content")
    analysis_name = st.selectbox("Document to analyze", names, key="xml_analysis_doc")
    analysis_rows = flatten_elements(
        documents[analysis_name]["root"], source=analysis_name
    )
    analysis_df = pd.DataFrame(analysis_rows)
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Distinct tags", analysis_df["tag"].nunique())
    summary_col2.metric("Attributes", sum(
        len(node.attrib) for node in documents[analysis_name]["root"].iter()
    ))
    summary_col3.metric(
        "Text-bearing elements",
        int(analysis_df["content_text"].astype(bool).sum()),
    )
    tag_summary = (
        analysis_df.groupby("tag", as_index=False)
        .agg(elements=("tag", "size"), text_elements=("content_text", lambda values: values.astype(bool).sum()))
        .sort_values("elements", ascending=False)
    )
    st.dataframe(tag_summary, use_container_width=True, hide_index=True)
    st.bar_chart(tag_summary.set_index("tag")[["elements", "text_elements"]])

    st.divider()
    st.subheader("Compare two XML documents")
    if len(names) < 2:
        st.info("Upload at least two well-formed XML files to compare them.")
    else:
        compare_cols = st.columns(2)
        left_name = compare_cols[0].selectbox("First document", names, index=0, key="xml_compare_left")
        right_name = compare_cols[1].selectbox("Second document", names, index=1, key="xml_compare_right")
        differences = compare_xml_trees(
            flatten_elements(documents[left_name]["root"]),
            flatten_elements(documents[right_name]["root"]),
        )
        st.caption(f"{len(differences):,} structural difference(s) by element path.")
        st.dataframe(differences, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Export XML and extracted data")
    export_name = st.selectbox("Document to export", names, key="xml_export_doc")
    export_doc = documents[export_name]
    export_processed = st.checkbox(
        "Export cleaned version when available",
        value=True,
        key="xml_export_cleaned",
    )
    saved_export = st.session_state.get("xml_processed", {}).get(export_name)
    export_bytes = (
        saved_export["content"]
        if export_processed and saved_export
        and saved_export["digest"] == export_doc["digest"]
        else export_doc["raw"]
    )
    filename_base = export_name.rsplit(".", 1)[0]
    export_cols = st.columns(4)
    export_cols[0].download_button(
        "⬇ Download XML",
        data=export_bytes,
        file_name=f"{filename_base}_cleaned.xml" if export_bytes != export_doc["raw"] else export_name,
        mime="application/xml",
        key="xml_export_raw",
    )
    export_rows = pd.DataFrame(
        flatten_elements(ET.fromstring(export_bytes), source=export_name)
    )
    export_cols[1].download_button(
        "⬇ Download elements CSV",
        data=export_rows.to_csv(index=False).encode("utf-8"),
        file_name=f"{filename_base}_elements.csv",
        mime="text/csv",
        key="xml_export_csv",
    )
    export_cols[2].download_button(
        "⬇ Download elements JSON",
        data=export_rows.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{filename_base}_elements.json",
        mime="application/json",
        key="xml_export_json",
    )
    export_cols[3].download_button(
        "⬇ Download elements Excel",
        data=dataframe_to_xlsx(export_rows),
        file_name=f"{filename_base}_elements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xml_export_xlsx",
    )

    st.divider()
    st.subheader("Split XML into smaller files")
    split_tag_options = sorted({
        local_name(child.tag)
        for child in ET.fromstring(export_bytes)
    })
    if not split_tag_options:
        st.info("The selected XML has no child elements to split.")
    else:
        split_tag = st.selectbox("Repeated child element", split_tag_options, key="xml_split_tag")
        chunk_size = st.number_input(
            "Elements per output file", min_value=1, max_value=10000,
            value=100, step=1, key="xml_split_size",
        )
        if st.button("Prepare split archive", key="xml_prepare_split"):
            root = ET.fromstring(export_bytes)
            selected_children = [
                child for child in root if local_name(child.tag) == split_tag
            ]
            other_children = [child for child in root if local_name(child.tag) != split_tag]
            files = {}
            for start in range(0, len(selected_children), int(chunk_size)):
                chunk_root = copy.deepcopy(root)
                for child in list(chunk_root):
                    chunk_root.remove(child)
                chunk_root.extend(copy.deepcopy(other_children))
                chunk_root.extend(copy.deepcopy(selected_children[start:start + int(chunk_size)]))
                part_number = start // int(chunk_size) + 1
                files[f"{filename_base}_part_{part_number}.xml"] = serialize_xml(chunk_root)
            if files:
                st.session_state["xml_split_archive"] = make_zip(files)
                st.success(f"Prepared {len(files)} part(s), each with up to {int(chunk_size)} `{split_tag}` elements.")
            else:
                st.warning(f"No `{split_tag}` elements were found.")

    if st.session_state.get("xml_split_archive"):
        st.download_button(
            "⬇ Download split XML archive",
            data=st.session_state["xml_split_archive"],
            file_name=f"{filename_base}_split.zip",
            mime="application/zip",
            key="xml_download_split",
        )

    if st.session_state.get("xml_combined"):
        st.download_button(
            "⬇ Download combined XML",
            data=st.session_state["xml_combined"],
            file_name="combined.xml",
            mime="application/xml",
            key="xml_download_combined",
        )
