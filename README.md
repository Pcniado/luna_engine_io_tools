# Luna Engine IO Tools

**Luna Engine IO Tools** is a Blender add-on written in Python for working with Luna Engine model and animation assets. It focuses on importing and exporting `.model` files and `.animclip` animation clips directly inside Blender, giving modders and reverse-engineering researchers a cleaner workflow for editing game assets.

## Features

* **Model import**: Import one or more compiled `.model` files into Blender.
* **Model export**: Export selected Blender model data back into the Luna Engine model format.
* **Animation import**: Import `.animclip` files and apply them to an existing armature or camera.
* **Animation export**: Export the selected armature or camera animation as an `.animclip` file.
* **Camera animation support**: Work with camera clips as well as skeletal animation clips.
* **Blender UI integration**: Adds Luna Engine import/export entries to Blender's File menu and provides panels, properties, and operators for model and animation settings.
* **Pure Python add-on**: Runs inside Blender's bundled Python environment with no extra Python packages required.

## Installation

1. Download or clone this repository.
2. Keep the files inside a folder named `luna_engine_io_tools`, or package that folder as a zip.
3. Open Blender 5.0 or newer.
4. Go to **Edit > Preferences > Add-ons**.
5. Click **Install** and select the zip or folder.
6. Enable **Luna Engine IO Tools** in the add-on list.

You can also copy the `luna_engine_io_tools` folder into Blender's add-ons directory and enable it from the Add-ons panel.

## Usage

### Importing models

1. Go to **File > Import > Luna Engine Model**.
2. Select one or more `.model` files.
3. Enable **Import All LODs** if you want lower-detail subsets too.
4. Click **Import**.

### Exporting models

1. Select the model/armature hierarchy you want to export.
2. Configure the model export settings in the Luna Engine model panel.
3. Use **File > Export > Luna Engine Model** or the **Export Luna Engine Model** operator.
4. Save the exported `.model` file.

### Importing animations

1. Select the target armature or camera.
2. Go to **File > Import > Luna Engine Anim**.
3. Select one or more `.animclip` files.
4. The importer will apply the clip to the selected compatible object.

### Exporting animations

1. Select an armature with an active action or a camera with animation data.
2. Go to **File > Export > Luna Engine Anim**.
3. Set the desired frame range and FPS in the scene/export settings.
4. Export the animation as an `.animclip` file.

## Supported file types

| Format | Description | Direction |
|---|---|---|
| `.model` | Luna Engine model data, including meshes, subsets, materials, and skeleton data | Import/export |
| `.animclip` | Animation clip data for armatures or cameras | Import/export |

## Requirements

* Blender 5.0.0 or newer.
* Python 3.x, included with Blender.
* No external Python packages required.

## Development

This project is written in pure Python. After editing the add-on, reload it in Blender or restart Blender to test changes. Keep changes focused, test with real assets when possible, and report issues with enough detail to reproduce them.

## License

This add-on is licensed under the GNU General Public License v3.0. See the [`LICENSE`](LICENSE) file for full details.

## Disclaimer

This project is not affiliated with Insomniac Games.
