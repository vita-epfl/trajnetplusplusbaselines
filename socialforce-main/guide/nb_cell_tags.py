import glob
import nbformat as nbf


def main():
    # Collect a list of all notebooks in the content folder
    notebooks = glob.glob("./**/*.ipynb", recursive=True)

    # Text to look for in adding tags
    text_search_dict = {
        "# HIDDEN": "remove-cell",  # Remove the whole cell
        "# NO CODE": "remove-input",  # Remove only the input
        "# HIDE CODE": "hide-input",  # Hide the input w/ a button to show
        "# HIDE OUTPUT": "hide-output",
        "import socialforce": "remove-cell",
    }

    # Search through each notebook and look for the text, add a tag if necessary
    for i_path in notebooks:
        notebook = nbf.read(i_path, nbf.NO_CONVERT)
        modified = False  # avoid writing when unnecessary to preserve cache

        for cell in notebook.cells:
            cell_tags = cell.get('metadata', {}).get('tags', [])

            # remove all tags that were previously set
            for tag in set(text_search_dict.values()):
                if tag in cell_tags:
                    cell_tags.remove(tag)

            for key, val in text_search_dict.items():
                if key in cell['source']:
                    if val not in cell_tags:
                        cell_tags.append(val)

            if cell_tags and ('tags' not in cell['metadata']
                              or set(cell_tags) != set(cell['metadata']['tags'])):
                cell['metadata']['tags'] = cell_tags
                modified = True
            elif 'tags' in cell['metadata'] and not cell_tags:
                del cell['metadata']['tags']
                modified = True

        if modified:
            nbf.write(notebook, i_path)


if __name__ == '__main__':
    main()
