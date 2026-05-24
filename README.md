# Anime2SD

**A 99% automatized pipeline to construct training set from anime and more for text-to-image model training**

Demonstration: [https://youtu.be/-Nzj6SEU9XY?si=8-o9vN6ToTRTeGea](https://youtu.be/-Nzj6SEU9XY?si=8-o9vN6ToTRTeGea)

The old scripts and readme have been moved into [scripts_v1](scripts_v1).

Note that the new naming of metadata follows the convention of [waifuc](https://github.com/deepghs/waifuc) and is thus different from the name given to the older version.
For conversion please use [utilities/convert_metadata.py](utilities/convert_metadata.py).

## Local Gradio Web UI

The Frame Lab UI in `app/gradio_ui.py` exposes the existing pipeline as a local workflow editor:

- All CLI settings from `anime2sd/parse_arguments.py` are rendered as controls, grouped by the stage where they take effect. Only settings tabs for enabled stages are displayed; **General** remains available for shared paths and output options.
- Each control displays the original argument description as inline help text, with concise quality, matching, or performance guidance where the setting affects results.
- Stages are selected independently. Stages 1 (frame extraction) and 2 (cropping) are optional; stages 3 through 7 can each be enabled or omitted with checkboxes. Stage 0 is available when downloading source material is desired.
- UI profiles can be exported to and imported from TOML. A single **Save profile** action stores the loaded starting preset values, stage selection, workspace root, and any field edits together in one executable profile beneath `configs/ui/saved/`. These files are intentionally ignored by Git because they commonly contain local paths.
- A **Workspace root** can derive the input, output, reference, and log paths from one Windows folder and create the required working structure with one button.
- Directory and file settings accept Windows paths such as `C:\datasets\anime\output`; paths are normalized before a profile is saved or a run starts.
- Consecutive selected stages are executed as one pipeline segment through `automatic_pipeline.py`, so generated output flows into the next selected stage correctly.
- **Stop pipeline** cancels the active UI run and terminates its pipeline process tree without closing the web interface.
- **Shut down server** terminates an active pipeline if necessary and closes the local Gradio server completely.

The UI listens only on `http://127.0.0.1:7866`. Port `7866` is intentionally fixed for repeatable bookmarks and Pinokio integration. Close another service using that port before launching Frame Lab.
The web UI is tested with Gradio `6.14.0`.

### Workspace Root

Enter one directory in **Workspace root**, for example `C:\datasets\anime\frieren_project`, and choose **Create workspace folders**. While this root is set, it takes precedence over manually entered General path fields for saved profiles and pipeline runs:

```text
C:\datasets\anime\frieren_project\
|-- src\                 # Input for the first enabled stage
|-- ref\                 # Character reference images for Stage 3
|-- logs\                # Pipeline log files
`-- dst\
    |-- intermediate\    # Generated raw, cropped, and classified working data
    `-- training\        # Final selected and captioned training data
```

The content expected in `src` depends on the first enabled stage: videos for Stage 1, raw images for Stage 2, or already cropped images when starting directly at Stage 3. Keep processing stages that depend on each other's output enabled consecutively, for example Stages 2 and 3 together.

### Character Reference Images

Set `--character_ref_dir` under **Stage 3 - Classify**, or use the workspace `ref` directory. Reference images are consumed in Stage 3 to map detected character crops or clusters to known character names; later stages use the resulting metadata rather than reading the reference folder again.

The recommended Windows layout is one subfolder per character:

```text
C:\datasets\anime\references\
|-- frieren\
|   |-- front_01.png
|   `-- portrait_02.webp
`-- fern\
    |-- reference_01.jpg
    `-- reference_02.png
```

Nested folder names become character labels. Images directly inside `references` are also supported: their label is taken from the file name up to the first underscore, for example `frieren_01.png` becomes `frieren`. Subfolders are preferable because the label is explicit and each character can hold multiple reference images.

### Preprocessing Messages

On the first processing pass, raw source images commonly do not yet have companion `_meta.json` files. The pipeline creates default metadata for those images before classification and later processing. This is expected initialization, not a failed match or missing image; the UI now reports it as one informational summary instead of one warning per image.

### Windows Quick Start

Use Python 3.10 because the upstream download dependencies do not support Python 3.11 or newer reliably.

```bat
setup.bat
launch.bat
```

`setup.bat` creates a Python 3.10 `env`, installs the local packages and development test dependencies, runs the application test suite, and uses current CUDA 12.8 PyTorch wheels on NVIDIA PCs. This installation route is intended for both an RTX 3090 and an RTX 5070 Ti. Stage 1 additionally requires `ffmpeg` on `PATH`.

All direct Python CLI calls must use this project environment. In PowerShell, activate it once per terminal session:

```powershell
.\env\Scripts\Activate.ps1
```

To reinstall the development/test dependencies or rerun the verified application suite manually:

```powershell
.\env\Scripts\python.exe -m uv pip install -r requirements-dev.txt
.\env\Scripts\python.exe -m pytest -q
```

The root `pytest.ini` collects the application tests under `tests/` only; the bundled `waifuc` source tree retains its upstream test suite separately. Classification integration tests are skipped unless their local sample-image directories under `data/` are populated.

### Terminal Output

Pipeline logs use severity colors when output is connected to an interactive terminal:

- `INFO` is blue.
- `WARNING` is yellow.
- `ERROR` is red.
- `CRITICAL` is bold red.

Color codes are omitted automatically when output is piped into the Gradio run log or a file. Set `NO_COLOR=1` to disable ANSI coloring in a terminal.

### Pinokio Launcher

This repository is an app launcher located under `PINOKIO_HOME/api/anime-screenshot-pipeline-vibe`. In Pinokio:

1. Choose **Install** to create `env` and install the CUDA/CPU dependencies.
2. Choose **Start** to launch Frame Lab.
3. When the Gradio server is ready, choose **Open Web UI**.
4. Choose **Run Tests** to execute the application test suite inside the managed environment.
5. Use **Update** to pull changes, reinstall dependencies, and rerun verification, or **Reset** to remove only the generated virtual environment.

The Pinokio start script checks the Gradio dependency before launching, captures the local Gradio URL, and opens the same fixed local endpoint used by `launch.bat`. `launch.bat` reports a direct setup instruction when UI dependencies are missing.

### Programmatic UI Access

Choose a **Starting preset**, load and edit it as needed, then save the single resulting profile, for example `configs/ui/saved/my_pipeline.toml`. The named Gradio endpoint `run_saved_profile` accepts that project-relative profile path and an optional comma-separated stage list.

Python:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7866")
result = client.predict(
    "configs/ui/saved/my_pipeline.toml",
    "3,4,5,6,7",
    api_name="/run_saved_profile",
)
print(result)
```

JavaScript:

```javascript
import { Client } from "@gradio/client";

const app = await Client.connect("http://127.0.0.1:7866");
const result = await app.predict("/run_saved_profile", [
  "configs/ui/saved/my_pipeline.toml",
  "3,4,5,6,7"
]);
console.log(result.data);
```

Curl:

```bash
curl -X POST "http://127.0.0.1:7866/call/run_saved_profile" \
  -H "Content-Type: application/json" \
  -d '{"data":["configs/ui/saved/my_pipeline.toml","3,4,5,6,7"]}'
```

The curl response contains an event identifier; consume its streamed result from `GET /call/run_saved_profile/<event_id>`. The original CLI shown below remains available for scripting without a running UI.

The UI controls are also exposed as named API endpoints:

```python
client.predict(api_name="/stop_pipeline")
client.predict(api_name="/shutdown_server")
```

```javascript
await app.predict("/stop_pipeline", []);
await app.predict("/shutdown_server", []);
```

```bash
curl -X POST "http://127.0.0.1:7866/call/stop_pipeline" -H "Content-Type: application/json" -d '{"data":[]}'
curl -X POST "http://127.0.0.1:7866/call/shutdown_server" -H "Content-Type: application/json" -d '{"data":[]}'
```


## Basic Usage

The commands in this section assume that the local `env` virtual environment has been activated. The script `automatic_pipeline.py` allows you to construct a text-to-image training set from anime with minimum effort. All you have to do is

```bash
python automatic_pipeline.py \
    --anime_name name_of_my_favorite_anime \
    --base_config_file configs/pipelines/base.toml \
    --config_file configs/pipelines/screenshots.toml configs/pipelines/booru.toml [...]
```

Providing multiple [configuration files](configs/pipelines) allow for parallel processing of fanarts and animes (and even for parallel processing of multiple animes). You can either create your own configuration files or overwrite existing values by command line arguments.

Of course, you can always go without configuration files if you do not need to run multiple pipelines in parallel.

```bash
python automatic_pipeline.py \
    --start_stage 1 \
    --end_stage 7 \
    --src_dir /path/to/video_dir \
    --dst_dir /path/to/dataset_dir \
    --character_ref_dir /path/to/ref_image_dir \
    --pipeline_type screenshots \
    --crop_with_head \
    --image_prefix my_favorite_anime \
    --ep_init 3 \
    --log_prefix my_favorite_anime
```

:bulb: You can first run from stages 1 to 3 without `--character_ref_dir` to cluster characters. Then you go through the clusters to quickly construct your reference folder and run again from stages 3 to 7 with `--character_ref_dir` now given. See [Wiki](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki) for details.  
:bulb:  Although it is possible to run from stage 0 which downloads anime automatically, it is still recommended to prepare the animes yourself as the downloading part is not fully optimized (may just hang if there are no seeders etc).

There are a lot of arguments (more than 100) that allow you to configure the entire process. See all of them in the aforementioned configuration files or with
```bash
python automatic_pipeline.py --help
```

It is highly recommended to read at least [Main Arguments](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Main-Arguments) so that you know how to set up things correctly.

## Advanced Usage

There are three ways that you can use the script.

- **Use it as a black box:** Type the anime name, go watching several episodes of anime, come back, and the dataset is ready.
- **Use it powerful dataset creation assistant:** You can decide yourself where to start and where to end, with possibility to manually inspect and modify the dataset after each stage and resume. You can provide character reference images, correct character classification results, adjust core tags, edit tags with other tools. This will allow you to construct a better dataset than what we get with the fully automatic process.
- **Use it as a tool box:** Each stage can be run independently for the task in question, with many parameters that you can adjust. Besides the main script, there are also other numerous scripts in this repository that are useful for dataset preparation. However, [waifuc](https://github.com/deepghs/waifuc) which this project heavily makes use of may be more appropriate in this case.

## Pipeline Overview

The script performs all the following automatically.

- [Stage 0] [Anime and fanart downloading](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-0:-Anime-and-Fanart-Downloading)
- [Stage 1] [Frame extraction and similar image removal](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-1:-Frame-Extraction-and-Similar-Image-Removal)
- [Stage 2] [Character detection and cropping ](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-2:-Character-Detection-and-Cropping)
- [Stage 3] [Character classification](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-3:-Character-Classification)
- [Stage 4] [Image selection and resizing](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-4:-Image-Selection-and-Resizing)
- [Stage 5] [Tagging, captioning, and generating wildcards and embedding initialization information](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-5:-Tagging-and-Captioning)
- [Stage 6] [Dataset arrangement](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-6:-Dataset-Arrangement)
- [Stage 7] [Repeat computation for concept balancing](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-7:-Repeat-Computation-for-Concept-Balancing)


## Dataset Organization and Training

- Once we go through the pipeline, the dataset is hierarchically organized in `/path/to/dataset_dir/training` with `multiply.txt` in each subfolder indicating the repeat of the images from this directory. More details on this are provided in [Dataset Organization](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Dataset-Organization).
- Since each trainer reads data differently. Some more steps may be required before training is performed. See [Start Training](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Start-Training) for what to do for [EveryDream2](https://github.com/victorchall/EveryDream2trainer), [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), and [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion).

## Installation

1. Clone this directory
    ```bash
    git clone https://github.com/cyber-meow/anime_screenshot_pipeline
    cd anime_screenshot_pipeline
    ```

2. Depending on your operating system, run either `install.sh` or `install.bat` in terminal. Both create and install into the project-local `env` virtual environment only.
3. Activate `env` before running Python commands:
    ```bash
    source env/bin/activate
    ```
    On Windows PowerShell, use:
    ```powershell
    .\env\Scripts\Activate.ps1
    ```

**Additional Steps and Known Issues**
 
- The first stage of the `screenshots` pipeline uses [ffmpeg](https://ffmpeg.org/) from command line. You can install it with
    - Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    - Windows: `choco install ffmpeg`provided that [Chocolatey](https://chocolatey.org/install) is installed
- If you want to use onnx with gpu, make sure cuda 11.8 is properly installed as [onnxruntime-gpu 1.16 uses cuda 11.8](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) 
- python >= 3.11.0 is not yet supported due to the use of the libtorrent library
- Anime downloading is not working on Windows again due to the use of libtorrent library: https://github.com/arvidn/libtorrent/issues/6689


## Change Logs

### Main

- Update documentation [2023.12.24]
- Fully automatic with only need for specifying anime name [2023.12.02]
- Multi-anime support [2023.12.01]
- Fanart support [2023.12.01]
- .toml support [2023.11.29]
- HCP-diffusion compatibility [2023.10.08]

### Secondary

- Load metadata from text file saved by imgbrd-grabber [2023.12.25]
- Keep tokens separator support for Kohya trainer, possibility to add dropped character tags to the end [2023.12.02]
- Ref directory hierarchy and Character class to account for different appearances of the same character [2023.11.28]
- Embedding initialization with hard tags [2023.11.11]
- Improved classification workflow that takes existing character metadata into account [2023.11.10]
- Core tag-based pruning [2023.10.15]
- Add size to metadata to avoid opening images for size comparison [2023.10.14]


## TODO / Potential improvements

Contributions are welcome

### Secondary

- [ ] Do not crop images that are already cropped before unless otherwise specified
- [ ] Text detection
- [ ] Improve core tag detection by using half body or full body images
- [ ] Bag of words clustering for wildcard
- [ ] Prepare HCP with multiple datasets
- [ ] Arguments to optionally remove subfolders with too few images
- [ ] Replace ffmpeg command by built-in python functions
- [ ] Improved tag pruning (with tag tree?)

### Advanced

- [ ] Beyond character classification: outfits, objects, backgrounds, etc.
- [ ] Image quality filtering 
- [ ] Segmentation and soft mask
- [ ] Graphical interfaces with tagging/captioning tools for manual correction



## Credits

- The new workflow is largely inspired by the fully automatic procedure for single character of [narugo1992](https://github.com/narugo1992) and is largely based on the library [waifuc](https://github.com/deepghs/waifuc)
- The [tag_filtering/overlap_tags.json](tag_filtering/overlap_tags.json) file is provided by gensen2ee
- See the [old readme](scripts_v1/README.md) as well
- The Gradio Frame Lab UI and Pinokio launcher additions are documented in [CREDITS.md](CREDITS.md).

## License

This project remains distributed under the MIT License. See [LICENSE](LICENSE) for the copyright notice and terms that apply to the original project and these modifications.

