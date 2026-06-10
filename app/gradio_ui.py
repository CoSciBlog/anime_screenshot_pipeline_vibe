from __future__ import annotations

import argparse
import html
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import toml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime2sd.parse_arguments import create_parser


PORT = 7866
CONFIG_DIR = ROOT / "configs" / "ui"
SAVED_CONFIG_DIR = CONFIG_DIR / "saved"
GLOBAL_CONFIG = CONFIG_DIR / "configuration.toml"
RUNTIME_CONFIG = SAVED_CONFIG_DIR / "_last_run.toml"
PIPELINE_PROCESS_LOCK = threading.Lock()
PIPELINE_RUN_LOCK = threading.Lock()
PIPELINE_STOP_REQUESTED = threading.Event()
ACTIVE_PIPELINE_PROCESS: subprocess.Popen[str] | None = None
WORKSPACE_DIRECTORIES = ("src", "ref", "logs")
WORKSPACE_RESERVED_DIRECTORIES = (*WORKSPACE_DIRECTORIES, "dst")
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
STAGE_START_RE = re.compile(r"Start stage\s+(\d+)")
TQDM_PROGRESS_RE = re.compile(
    r"(?P<label>[^:\n]+):\s*(?P<percent>\d+)%\|.*?\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<timing>[^\]]+)\]"
)

STAGES = OrderedDict(
    [
        (
            0,
            (
                "Download",
                "Fetch source media from anime torrents or booru sources. Use this page to control source breadth, episode range, ratings, and size limits before any local processing begins.",
            ),
        ),
        (
            1,
            (
                "Frames",
                "Extract frames and remove near duplicates. These settings decide how much motion variety is kept and how aggressively repeated screenshots are filtered before detection.",
            ),
        ),
        (
            2,
            (
                "Detect",
                "Detect characters and create crop intermediates. Detection settings trade speed, recall, and identity evidence such as heads, faces, full bodies, and optional three-stage crops.",
            ),
        ),
        (
            3,
            (
                "Classify",
                "Match detected crops to reference characters or clusters. Reference folders, clustering thresholds, output limits, and cleanup options are configured here.",
            ),
        ),
        (
            4,
            (
                "Select",
                "Build the training image set from classified crops and source frames. This page controls what image variants are exported, resized, filtered, or removed from the final dataset.",
            ),
        ),
        (
            5,
            (
                "Caption",
                "Generate tags and captions for training. Thresholds, pruning, tag ordering, and metadata probabilities change how explicit and stable the training signal becomes.",
            ),
        ),
        (
            6,
            (
                "Arrange",
                "Organize captioned images into concept and character folders. These settings define how combinations are grouped before repeat balancing.",
            ),
        ),
        (
            7,
            (
                "Balance",
                "Calculate repeat weights for training subsets. Use this page to cap oversampling, lift sparse concepts, and apply an optional custom weighting CSV.",
            ),
        ),
    ]
)

FIELD_GROUPS = OrderedDict(
    [
        (
            "General",
            [
                "src_dir",
                "dst_dir",
                "extra_path_component",
                "log_dir",
                "log_prefix",
                "pipeline_type",
                "image_type",
                "remove_intermediate",
                "remove_src_files_after_pipeline",
                "overwrite_path",
                "load_grabber_ext",
                "load_aux",
                "save_aux",
            ],
        ),
        (
            "Stage 0 - Download",
            [
                "anime_name",
                "candidate_submitters",
                "anime_resolution",
                "min_download_episode",
                "max_download_episode",
                "anime_name_booru",
                "character_info_file",
                "download_for_characters",
                "booru_download_limit",
                "booru_download_limit_per_character",
                "allowed_ratings",
                "allowed_image_classes",
                "max_download_size",
            ],
        ),
        (
            "Stage 1 - Frames",
            [
                "extract_key",
                "image_prefix",
                "ep_init",
                "no_remove_similar",
                "detect_duplicate_model",
                "detect_duplicate_batch_size",
                "similar_thresh",
            ],
        ),
        (
            "Stage 2 - Detect",
            [
                "min_crop_size",
                "crop_with_head",
                "crop_with_face",
                "detect_level",
                "use_3stage_crop",
            ],
        ),
        (
            "Stage 3 - Classify",
            [
                "character_ref_dir",
                "n_add_to_ref_per_character",
                "max_images_per_character",
                "max_images_per_character_per_episode",
                "remove_classified_aux_files",
                "remove_stage2_crops_after_classification",
                "remove_noise_folder_after_classification",
                "ignore_character_metadata",
                "no_extract_from_noise",
                "no_filter_characters",
                "keep_unnamed_clusters",
                "accept_multiple_candidates",
                "cluster_merge_threshold",
                "cluster_min_samples",
                "classification_chunk_size",
                "same_threshold_rel",
                "same_threshold_abs",
            ],
        ),
        (
            "Stage 4 - Select",
            [
                "overwrite_emb_init_info",
                "character_overwrite_uncropped",
                "character_remove_unclassified",
                "no_cropped_in_dataset",
                "no_original_in_dataset",
                "no_resize",
                "max_size",
                "image_save_ext",
                "n_anime_reg",
                "filter_again",
            ],
        ),
        (
            "Stage 5 - Caption",
            [
                "overwrite_tags",
                "tagging_method",
                "tag_threshold",
                "sort_mode",
                "append_dropped_character_tags",
                "max_tag_number",
                "blacklist_tags_file",
                "overlap_tags_file",
                "character_tags_file",
                "process_from_original_tags",
                "prune_mode",
                "drop_difficulty",
                "compute_core_tag_up_levels",
                "core_frequency_thresh",
                "use_existing_core_tag_file",
                "drop_all_core",
                "emb_min_difficulty",
                "emb_max_difficulty",
                "emb_init_all_core",
                "append_dropped_character_tags_wildcard",
                "caption_ordering",
                "caption_inner_sep",
                "caption_outer_sep",
                "character_sep",
                "character_inner_sep",
                "character_outer_sep",
                "keep_tokens_sep",
                "keep_tokens_before",
                "use_npeople_prob",
                "use_character_prob",
                "use_copyright_prob",
                "use_image_type_prob",
                "use_artist_prob",
                "use_rating_prob",
                "use_crop_info_prob",
                "use_tags_prob",
            ],
        ),
        (
            "Stage 6 - Arrange",
            [
                "rearrange_up_levels",
                "arrange_format",
                "max_character_number",
                "min_images_per_combination",
            ],
        ),
        (
            "Stage 7 - Balance",
            [
                "compute_multiply_up_levels",
                "min_multiply",
                "max_multiply",
                "weight_csv",
            ],
        ),
    ]
)

DEFAULT_ENABLED_STAGES = ["3", "4", "5", "6", "7"]
GROUP_STAGES = {
    f"Stage {number} - {title}": number
    for number, (title, _description) in STAGES.items()
}
PATH_FIELDS = {
    "src_dir",
    "dst_dir",
    "log_dir",
    "character_info_file",
    "character_ref_dir",
    "blacklist_tags_file",
    "overlap_tags_file",
    "character_tags_file",
    "weight_csv",
}
FIELD_GUIDANCE = {
    "src_dir": (
        "This is the source folder for the first enabled stage. It should contain videos, raw images, or generated intermediates depending on where the run starts. "
        r"When a workspace root is used, this is mapped to <root>\src; reference images belong in the separate ref folder."
    ),
    "dst_dir": r"Pipeline output root. Intermediate data is written below <root>\dst\intermediate and final training data below <root>\dst\training when a workspace root is used.",
    "character_ref_dir": (
        "Reference image root used only in Stage 3. Create one subfolder per character and place clear reference crops or portraits inside it, "
        r"for example <root>\ref\frieren\*.png. Better references reduce cluster mistakes."
    ),
    "candidate_submitters": "Use a narrow list for consistent encodes or a broader list when availability matters more than uniform video style.",
    "anime_resolution": "Higher resolutions preserve small faces and costume details, but downloads, frame extraction, and detection take longer.",
    "booru_download_limit": "More images improve coverage of poses and outfits, but they increase download time, filtering cost, and manual review volume.",
    "booru_download_limit_per_character": "Higher per-character limits help rare characters, while lower limits keep large casts from dominating disk usage.",
    "allowed_ratings": "Restricting ratings changes the visual distribution of the dataset and can remove otherwise useful images.",
    "allowed_image_classes": "Stricter classes improve consistency; broader classes add variety but may introduce screenshots, comics, or 3D content.",
    "max_download_size": "Smaller values save disk and processing time. Larger values keep detail that may help detection and later training.",
    "extract_key": "Key-frame extraction is faster and usually less repetitive, but it may skip short expressions or fast motion poses.",
    "no_remove_similar": "Enable only when you want maximum coverage. It keeps repeated frames and can make the dataset visually redundant.",
    "detect_duplicate_model": "Stronger models may catch subtle duplicates more reliably, but they need more startup time and compute.",
    "detect_duplicate_batch_size": "Increase for throughput when VRAM/RAM is available; reduce it when duplicate detection runs out of memory.",
    "similar_thresh": "Higher thresholds keep more near-duplicates. Lower thresholds remove repetition more aggressively and may discard useful variants.",
    "min_crop_size": "Raise this to reject tiny or blurry detections; lower it when small or distant characters are important.",
    "crop_with_head": "Useful for identity-focused datasets because it keeps crops with head evidence, but it drops partial body crops.",
    "crop_with_face": "Most strict identity filter. It improves character certainty but reduces recall for back views, masks, and stylized faces.",
    "detect_level": "Larger detector levels can find more difficult crops, while smaller levels are faster and use less memory.",
    "use_3stage_crop": "Adds person, half-body, and head variants. Use it once when close-up training coverage matters; it increases runtime and intermediate size.",
    "n_add_to_ref_per_character": "Adds confident matches back into the reference set. This can improve future runs, but bad matches will also be reinforced.",
    "max_images_per_character": "Caps saved images per recognized character to reduce imbalance. Set 0 when you want every accepted match.",
    "max_images_per_character_per_episode": "Caps repeated matches per character within each SxxExx episode, reducing near-duplicate dominance from long scenes.",
    "remove_classified_aux_files": "Deletes JSON metadata and NPY feature caches when they are no longer needed. This saves space but prevents reuse without recomputation.",
    "remove_stage2_crops_after_classification": (
        "Deletes generated Stage 2 crop intermediates after Stage 3 succeeds. The cleanup is skipped when Stage 3 used a user-provided input folder."
    ),
    "remove_noise_folder_after_classification": (
        "Deletes classified 0_noise/0_noisy output after Stage 3. Use it only when rejected crops are not needed for review, debugging, or later reuse."
    ),
    "remove_src_files_after_pipeline": (
        "Deletes files inside the configured source folder after the full pipeline finishes successfully. "
        "Use only when the source media has already been copied, backed up, or no longer needs to be reused."
    ),
    "no_filter_characters": "Disabling consistency filtering retains more samples, but the final classified folders may contain more label noise.",
    "keep_unnamed_clusters": "Keeps unmatched clusters as extra material. They add coverage but do not become named reference characters.",
    "cluster_merge_threshold": "Controls when similar clusters are joined. Lower values merge more aggressively and can mix different characters.",
    "cluster_min_samples": "Higher values suppress tiny clusters and noise, but they can lose rare characters with few crops.",
    "classification_chunk_size": (
        "Bounds the quadratic Stage 3 similarity step. Lower values use less RAM/VRAM but may split unnamed clusters across chunks."
    ),
    "same_threshold_rel": "Controls relative similarity strictness for noise extraction and filtering. Higher values are stricter.",
    "same_threshold_abs": "Controls absolute support needed for larger clusters during noise extraction and filtering.",
    "no_cropped_in_dataset": "Excluding crops produces a more scene-focused dataset and reduces close-up identity examples.",
    "no_original_in_dataset": "Excluding originals produces a crop-focused dataset and removes wider composition/context frames.",
    "no_resize": "Keeps native image sizes. This preserves detail but increases storage and shifts resizing work to the trainer.",
    "max_size": "Larger exports preserve more detail. Smaller exports save disk and can speed up later training prep.",
    "filter_again": "Runs duplicate filtering after selection for a cleaner final dataset, at the cost of extra processing time.",
    "tagging_method": "Choose the tagging model. Accuracy, tag vocabulary, startup cost, and inference speed differ by model.",
    "tag_threshold": "Higher thresholds keep only confident tags. Lower thresholds add more detail but can introduce weak or noisy tags.",
    "max_tag_number": "Limits caption length. More tags add specificity, while fewer tags keep captions focused.",
    "prune_mode": "Controls how redundant or overly broad tags are removed. This changes which concepts dominate training captions.",
    "core_frequency_thresh": "Higher values mark fewer tags as core and keep more details in image-specific captions.",
    "use_character_prob": "Lower values reduce how often character names appear in captions, weakening identity conditioning.",
    "use_tags_prob": "Lower values reduce descriptive tags and make captions rely more on selected metadata fields.",
    "arrange_format": "Defines the folder hierarchy used for concept separation before repeat balancing.",
    "min_images_per_combination": "Higher values merge sparse combinations into broader groups, improving stability but losing granularity.",
    "min_multiply": "Sets the minimum repeat exposure so small groups are not ignored during training.",
    "max_multiply": "Caps oversampling. Lower caps reduce overfitting risk for small or rare groups.",
    "weight_csv": "Optional manual weight table. Use it when automatic balancing needs project-specific overrides.",
}

DEFAULT_LANGUAGE = "en"
LANGUAGE_CHOICES = [("English", "en"), ("Deutsch", "de")]

STAGE_TRANSLATIONS = {
    "de": {
        0: (
            "Download",
            "Lädt Quellmedien aus Anime- oder Booru-Quellen. Auf dieser Seite steuerst du Quellenbreite, Episodenbereich, Ratings und Größenlimits, bevor lokale Verarbeitung startet.",
        ),
        1: (
            "Frames",
            "Extrahiert Einzelbilder und entfernt nahe Duplikate. Diese Einstellungen bestimmen, wie viel Bewegungsvarianz erhalten bleibt und wie streng wiederholte Screenshots gefiltert werden.",
        ),
        2: (
            "Erkennen",
            "Erkennt Charaktere und erzeugt Crop-Zwischendaten. Die Optionen steuern Geschwindigkeit, Trefferquote und Identitätsmerkmale wie Kopf, Gesicht, Körper und Drei-Stufen-Crops.",
        ),
        3: (
            "Klassifizieren",
            "Ordnet erkannte Crops Referenzcharakteren oder Clustern zu. Referenzordner, Clustering-Schwellen, Ausgabelimits und Cleanup-Optionen werden hier gesetzt.",
        ),
        4: (
            "Auswählen",
            "Erstellt den Trainingsdatensatz aus klassifizierten Crops und Quellbildern. Diese Seite steuert, welche Varianten exportiert, skaliert, gefiltert oder entfernt werden.",
        ),
        5: (
            "Beschriften",
            "Erzeugt Tags und Captions für das Training. Schwellen, Pruning, Tag-Reihenfolge und Metadaten-Wahrscheinlichkeiten verändern die Trainingssignale.",
        ),
        6: (
            "Anordnen",
            "Sortiert beschriftete Bilder in Konzept- und Charakterordner. Diese Einstellungen definieren, wie Kombinationen vor dem Balancing gruppiert werden.",
        ),
        7: (
            "Balancieren",
            "Berechnet Repeat-Gewichte für Trainingsgruppen. Nutze diese Seite, um Oversampling zu begrenzen, kleine Konzepte anzuheben oder eine eigene Gewichtungs-CSV anzuwenden.",
        ),
    }
}

GROUP_LABELS_DE = {
    "General": "Allgemein",
    "Stage 0 - Download": "Stage 0 - Download",
    "Stage 1 - Frames": "Stage 1 - Frames",
    "Stage 2 - Detect": "Stage 2 - Erkennen",
    "Stage 3 - Classify": "Stage 3 - Klassifizieren",
    "Stage 4 - Select": "Stage 4 - Auswählen",
    "Stage 5 - Caption": "Stage 5 - Beschriften",
    "Stage 6 - Arrange": "Stage 6 - Anordnen",
    "Stage 7 - Balance": "Stage 7 - Balancieren",
}

UI_TEXT = {
    "en": {
        "workflow": "Workflow",
        "stages_to_run": "Stages to run",
        "stages_info": "Only enabled stages are run and shown in Settings by stage. Stage 3 uses reference character folders for matching.",
        "stage_guide": "Stage guide",
        "run_pipeline": "Run pipeline",
        "stop_pipeline": "Stop pipeline",
        "ready": "Ready",
        "ready_message": "Select stages and run the pipeline.",
        "run_log": "Run log",
        "configuration": "Configuration",
        "language": "Language",
        "language_info": "Changes labels, stage names, page descriptions, and setting help text. Save configuration to keep this preference.",
        "workspace_root": "Workspace root",
        "workspace_placeholder": r"C:\datasets\anime\my_project",
        "workspace_info": r"Optional working root. Runs and saved configuration use <root>\src, <root>\dst, <root>\ref, and <root>\logs.",
        "create_workspace": "Create workspace folders",
        "clear_output": "Clear generated output",
        "workspace_help": "`src` = first-stage input, `ref` = character reference images, `dst/intermediate` and `dst/training` = generated pipeline data. Creating a workspace keeps existing contents and leaves `dst` uncreated until data is written; clearing output removes only `dst` results.",
        "import_configuration": "Import configuration",
        "load_configuration": "Load configuration",
        "configuration_help": "One global TOML file stores language, stage selection, workspace paths, and all settings. It is restored automatically when the application starts.",
        "save_configuration": "Save configuration",
        "export_settings": "Export settings",
        "settings_by_stage": "Settings by stage",
        "additional_settings": "Additional settings",
        "shutdown": "Shut down server",
        "shutdown_confirm": "**Shut down the server?** This stops an active pipeline and closes the Frame Lab web interface.",
        "cancel": "Cancel",
        "confirm_shutdown": "Yes, shut down server",
        "web_interface": "Web interface",
        "fixed_port": "fixed port",
        "impact": "Effect",
        "setting_fallback": "Pipeline setting used by the command-line workflow.",
        "list_help": "Enter one option per line or separate values with commas.",
        "config_saved": "Global configuration saved. Export is ready:",
        "config_loaded_global": "Global configuration loaded:",
        "config_loaded_imported": "Imported configuration loaded:",
        "config_invalid": "Choose a valid TOML configuration file to import.",
        "config_missing": "No global configuration saved yet. Adjust settings and select Save configuration.",
        "waiting": "Waiting to start",
        "current": "Current",
        "all_complete": "All selected stages complete",
        "complete_count": "{done} of {total} stages complete",
    },
    "de": {
        "workflow": "Workflow",
        "stages_to_run": "Auszuführende Stages",
        "stages_info": "Nur aktivierte Stages werden ausgeführt und unter Einstellungen nach Stage angezeigt. Stage 3 nutzt Referenzordner für die Charakterzuordnung.",
        "stage_guide": "Stage-Übersicht",
        "run_pipeline": "Pipeline starten",
        "stop_pipeline": "Pipeline stoppen",
        "ready": "Bereit",
        "ready_message": "Wähle Stages aus und starte die Pipeline.",
        "run_log": "Ausführungslog",
        "configuration": "Konfiguration",
        "language": "Sprache",
        "language_info": "Ändert Labels, Stage-Namen, Seitenbeschreibungen und Hilfetexte der Einstellungen. Speichere die Konfiguration, um die Sprache beizubehalten.",
        "workspace_root": "Workspace-Stammordner",
        "workspace_placeholder": r"C:\datasets\anime\mein_projekt",
        "workspace_info": r"Optionaler Arbeitsordner. Läufe und gespeicherte Konfiguration verwenden <root>\src, <root>\dst, <root>\ref und <root>\logs.",
        "create_workspace": "Workspace-Ordner erstellen",
        "clear_output": "Generierte Ausgabe löschen",
        "workspace_help": "`src` = Eingabe für die erste Stage, `ref` = Referenzbilder, `dst/intermediate` und `dst/training` = generierte Pipeline-Daten. Beim Erstellen bleiben vorhandene Inhalte erhalten; `dst` entsteht erst, wenn Daten geschrieben werden. Ausgabe löschen entfernt nur `dst`-Ergebnisse.",
        "import_configuration": "Konfiguration importieren",
        "load_configuration": "Konfiguration laden",
        "configuration_help": "Eine globale TOML-Datei speichert Sprache, Stage-Auswahl, Workspace-Pfade und alle Einstellungen. Sie wird beim Start automatisch wiederhergestellt.",
        "save_configuration": "Konfiguration speichern",
        "export_settings": "Einstellungen exportieren",
        "settings_by_stage": "Einstellungen nach Stage",
        "additional_settings": "Weitere Einstellungen",
        "shutdown": "Server herunterfahren",
        "shutdown_confirm": "**Server herunterfahren?** Das stoppt eine aktive Pipeline und schließt die Frame-Lab-Weboberfläche.",
        "cancel": "Abbrechen",
        "confirm_shutdown": "Ja, Server herunterfahren",
        "web_interface": "Weboberfläche",
        "fixed_port": "fester Port",
        "impact": "Auswirkung",
        "setting_fallback": "Pipeline-Einstellung für den Kommandozeilen-Workflow.",
        "list_help": "Gib eine Option pro Zeile ein oder trenne Werte mit Kommas.",
        "config_saved": "Globale Konfiguration gespeichert. Export ist bereit:",
        "config_loaded_global": "Globale Konfiguration geladen:",
        "config_loaded_imported": "Importierte Konfiguration geladen:",
        "config_invalid": "Wähle eine gültige TOML-Konfigurationsdatei zum Importieren.",
        "config_missing": "Es wurde noch keine globale Konfiguration gespeichert. Passe Einstellungen an und wähle Konfiguration speichern.",
        "waiting": "Wartet auf Start",
        "current": "Aktuell",
        "all_complete": "Alle ausgewählten Stages abgeschlossen",
        "complete_count": "{done} von {total} Stages abgeschlossen",
    },
}

FIELD_GUIDANCE_DE = {
    "src_dir": r"Quellordner für die erste aktivierte Stage. Je nach Startpunkt enthält er Videos, Rohbilder oder Zwischendaten. Mit Workspace-Stammordner wird er auf <root>\src gesetzt; Referenzbilder gehören nach ref.",
    "dst_dir": r"Ausgabe-Stammordner der Pipeline. Mit Workspace-Stammordner landen Zwischendaten unter <root>\dst\intermediate und Trainingsdaten unter <root>\dst\training.",
    "character_ref_dir": r"Referenzordner nur für Stage 3. Lege pro Charakter einen Unterordner an, z. B. <root>\ref\frieren\*.png. Gute Referenzen reduzieren falsche Cluster.",
    "candidate_submitters": "Eine engere Liste erzeugt konsistentere Encodes; eine breitere Liste erhöht die Chance, alle Episoden zu finden.",
    "anime_resolution": "Höhere Auflösung erhält kleine Gesichter und Kostümdetails, erhöht aber Download-, Extraktions- und Erkennungszeit.",
    "booru_download_limit": "Mehr Bilder verbessern Posen- und Outfit-Abdeckung, erhöhen aber Downloadzeit, Filteraufwand und Review-Menge.",
    "booru_download_limit_per_character": "Höhere Limits helfen seltenen Charakteren; niedrigere Limits verhindern, dass große Casts den Speicher dominieren.",
    "allowed_ratings": "Rating-Filter ändern die Bildverteilung und können sonst nützliche Bilder entfernen.",
    "allowed_image_classes": "Strengere Klassen verbessern Konsistenz; breitere Klassen bringen Vielfalt, aber auch Comics, 3D oder uneinheitliche Quellen.",
    "max_download_size": "Kleinere Werte sparen Speicher und Laufzeit. Größere Werte erhalten Details für Erkennung und Training.",
    "extract_key": "Keyframes sind schneller und meist weniger repetitiv, können aber kurze Gesichtsausdrücke oder schnelle Posen überspringen.",
    "no_remove_similar": "Nur aktivieren, wenn maximale Abdeckung wichtiger ist als Redundanz. Ähnliche Frames bleiben erhalten.",
    "detect_duplicate_model": "Stärkere Modelle erkennen subtile Duplikate besser, brauchen aber mehr Startzeit und Rechenleistung.",
    "detect_duplicate_batch_size": "Erhöhe den Wert für mehr Durchsatz bei genug VRAM/RAM; senke ihn bei Speicherfehlern.",
    "similar_thresh": "Höhere Werte behalten mehr nahe Duplikate. Niedrigere Werte filtern Wiederholungen aggressiver.",
    "min_crop_size": "Erhöhen, um kleine oder unscharfe Erkennungen zu verwerfen; senken, wenn entfernte Charaktere wichtig sind.",
    "crop_with_head": "Gut für Identitätssicherheit, weil Kopfmerkmale erforderlich sind; verwirft aber gültige Teilkörper-Crops.",
    "crop_with_face": "Strengster Identitätsfilter. Erhöht Sicherheit, senkt aber Recall bei Rückenansichten, Masken oder stilisierten Gesichtern.",
    "detect_level": "Größere Detektoren finden schwierigere Crops, kleinere sind schneller und sparsamer.",
    "use_3stage_crop": "Erzeugt Personen-, Halbbody- und Kopfvarianten. Sinnvoll für Close-ups, aber langsam und speicherintensiv.",
    "n_add_to_ref_per_character": "Fügt sichere Treffer den Referenzen hinzu. Kann spätere Läufe verbessern, verstärkt aber auch falsche Zuordnungen.",
    "max_images_per_character": "Begrenzt gespeicherte Bilder pro erkanntem Charakter. 0 speichert alle akzeptierten Treffer.",
    "max_images_per_character_per_episode": "Begrenzt wiederholte Treffer pro Charakter und SxxExx-Episode, damit lange Szenen nicht dominieren.",
    "remove_classified_aux_files": "Löscht JSON-Metadaten und NPY-Feature-Caches, wenn sie nicht mehr benötigt werden. Spart Platz, verhindert aber Wiederverwendung ohne Neuberechnung.",
    "remove_stage2_crops_after_classification": "Löscht generierte Stage-2-Crops nach erfolgreicher Stage 3. Bei benutzerdefinierten Stage-3-Eingaben wird nichts gelöscht.",
    "remove_noise_folder_after_classification": "Löscht `0_noise`/`0_noisy` nach Stage 3. Nur nutzen, wenn verworfene Crops nicht für Review, Debugging oder spätere Nutzung gebraucht werden.",
    "remove_src_files_after_pipeline": "Löscht Dateien im konfigurierten Quellordner erst nach erfolgreichem Abschluss der gesamten Pipeline. Nur nutzen, wenn die Quellen gesichert sind oder nicht erneut gebraucht werden.",
    "no_filter_characters": "Deaktiviert Konsistenzfilter. Dadurch bleiben mehr Samples erhalten, aber mit höherem Label-Noise-Risiko.",
    "keep_unnamed_clusters": "Behält unbenannte Cluster als Zusatzmaterial. Sie erhöhen Abdeckung, erhalten aber keine Referenznamen.",
    "cluster_merge_threshold": "Steuert, wann ähnliche Cluster zusammengeführt werden. Niedrigere Werte mergen aggressiver und können Charaktere vermischen.",
    "cluster_min_samples": "Höhere Werte unterdrücken kleine Cluster und Noise, können aber seltene Charaktere mit wenigen Crops verlieren.",
    "classification_chunk_size": "Begrenzt den speicherintensiven Ähnlichkeitsschritt in Stage 3. Niedrigere Werte sparen RAM/VRAM, können aber unbenannte Cluster aufteilen.",
    "same_threshold_rel": "Relative Ähnlichkeitsstrenge für Noise-Extraktion und Filterung. Höhere Werte sind strenger.",
    "same_threshold_abs": "Absolute Mindestunterstützung für größere Cluster bei Noise-Extraktion und Filterung.",
    "no_cropped_in_dataset": "Crops werden nicht in den Trainingsdatensatz übernommen. Ergebnis wird stärker szenenorientiert.",
    "no_original_in_dataset": "Originalbilder werden nicht übernommen. Ergebnis wird stärker crop- und identitätsorientiert.",
    "no_resize": "Behält native Bildgrößen. Erhält Details, erhöht aber Speicherbedarf und verlagert Resize-Arbeit ins Training.",
    "max_size": "Größere Exporte erhalten mehr Details. Kleinere Exporte sparen Speicher und beschleunigen spätere Vorbereitung.",
    "filter_again": "Filtert nach der Auswahl erneut Duplikate. Sauberer Datensatz, aber zusätzliche Laufzeit.",
    "tagging_method": "Wählt das Tagging-Modell. Genauigkeit, Tag-Vokabular, Startkosten und Geschwindigkeit unterscheiden sich.",
    "tag_threshold": "Höhere Schwellen behalten nur sichere Tags. Niedrigere Schwellen liefern mehr Details, aber mehr schwache Tags.",
    "max_tag_number": "Begrenzt die Caption-Länge. Mehr Tags sind spezifischer, weniger Tags fokussieren wichtige Konzepte.",
    "prune_mode": "Steuert, wie redundante oder zu breite Tags entfernt werden. Das verändert, welche Konzepte in Captions dominieren.",
    "core_frequency_thresh": "Höhere Werte markieren weniger Tags als Core-Tags und lassen mehr Details bildspezifisch.",
    "use_character_prob": "Niedrigere Werte reduzieren, wie oft Charakternamen in Captions erscheinen.",
    "use_tags_prob": "Niedrigere Werte reduzieren beschreibende Tags und nutzen stärker ausgewählte Metadatenfelder.",
    "arrange_format": "Definiert die Ordnerhierarchie zur Konzepttrennung vor dem Repeat-Balancing.",
    "min_images_per_combination": "Höhere Werte mergen seltene Kombinationen in breitere Gruppen. Stabiler, aber weniger granular.",
    "min_multiply": "Mindest-Repeat, damit kleine Gruppen im Training nicht ignoriert werden.",
    "max_multiply": "Begrenzt Oversampling. Niedrigere Caps senken Overfitting-Risiko bei kleinen Gruppen.",
    "weight_csv": "Optionale manuelle Gewichtungstabelle für projektspezifische Korrekturen.",
}


def normalized_language(language: str | None) -> str:
    return language if language in UI_TEXT else DEFAULT_LANGUAGE


def ui_text(language: str | None, key: str) -> str:
    language = normalized_language(language)
    return UI_TEXT[language].get(key, UI_TEXT[DEFAULT_LANGUAGE][key])


def stage_details(number: int, language: str | None = DEFAULT_LANGUAGE) -> tuple[str, str]:
    language = normalized_language(language)
    return STAGE_TRANSLATIONS.get(language, {}).get(number, STAGES[number])


def stage_choices(language: str | None = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    return [
        (f"{number} - {stage_details(number, language)[0]}", str(number))
        for number in STAGES
    ]


def stage_guide_html(language: str | None = DEFAULT_LANGUAGE) -> str:
    return (
        "<div class='stage-grid'><table>"
        + "".join(
            f"<tr><td>{number} - {html.escape(title)}</td><td>{html.escape(description)}</td></tr>"
            for number, (title, description) in (
                (number, stage_details(number, language)) for number in STAGES
            )
        )
        + "</table></div>"
    )


def group_label(group_name: str, language: str | None = DEFAULT_LANGUAGE) -> str:
    if normalized_language(language) == "de":
        return GROUP_LABELS_DE.get(group_name, group_name)
    return group_name


def setting_info(action: argparse.Action, language: str | None = DEFAULT_LANGUAGE) -> str:
    language = normalized_language(language)
    base = action.help or ui_text(language, "setting_fallback")
    guidance = FIELD_GUIDANCE_DE.get(action.dest) if language == "de" else FIELD_GUIDANCE.get(action.dest)
    if language == "de":
        if guidance:
            return f"{ui_text(language, 'impact')}: {guidance}"
        return ui_text(language, "setting_fallback")
    if guidance:
        return f"{base} {ui_text(language, 'impact')}: {guidance}"
    return base

STYLE = """
.gradio-container {
  --frame-bg: #f2f4f2;
  --panel-bg: #ffffff;
  --panel-raised: #f7f8f6;
  --line: #d8ded9;
  --ink: #172127;
  --muted: #5b6b70;
  --accent: #bf5a20;
  --accent-soft: #f8e7d7;
  background:
    radial-gradient(circle at 3% 0%, #fff9ee 0, transparent 35rem),
    linear-gradient(180deg, #fafbf8 0%, var(--frame-bg) 100%);
  color: var(--ink);
  font-family: "Aptos", "Trebuchet MS", sans-serif;
  min-height: 100vh;
}
.dark .gradio-container,
.gradio-container.dark {
  --frame-bg: #0c141a;
  --panel-bg: #121d24;
  --panel-raised: #18262e;
  --line: #2b3e47;
  --ink: #f2efe9;
  --muted: #b4c1c4;
  --accent: #f2a34b;
  --accent-soft: #2b241c;
  background:
    radial-gradient(circle at 3% 0%, #20394b 0, transparent 36rem),
    linear-gradient(180deg, #0e1921 0%, var(--frame-bg) 100%);
}
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container .prose {
  color: var(--ink);
}
.gradio-container .secondary-text,
.gradio-container .info {
  color: var(--muted);
}
.workspace {
  padding-top: .5rem;
}
.run-panel {
  border: 1px solid var(--line);
  border-radius: .7rem;
  background: var(--panel-bg);
  padding: 1rem;
  box-shadow: 0 10px 34px rgba(15, 26, 34, .06);
}
.dark .run-panel { box-shadow: 0 14px 38px rgba(0, 0, 0, .22); }
.panel-label {
  margin: 0 0 .75rem;
  color: var(--muted);
  font-size: .74rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 700;
}
.stage-grid {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  overflow: hidden;
  margin: .6rem 0 .9rem;
}
.stage-grid table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.stage-grid td { padding: .62rem .7rem; border-bottom: 1px solid var(--line); color: var(--muted); }
.stage-grid tr:last-child td { border-bottom: 0; }
.stage-grid td:first-child { color: var(--accent); font-weight: 700; white-space: nowrap; }
.settings-label {
  color: var(--muted);
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin: 1.4rem 0 .65rem;
}
.settings-tabs {
  background: var(--panel-bg);
  border: 1px solid var(--line);
  border-radius: .7rem;
  padding: .55rem;
  box-shadow: 0 10px 34px rgba(15, 26, 34, .05);
}
.settings-tabs button {
  color: var(--muted);
  font-weight: 600;
}
.settings-tabs button.selected {
  color: var(--accent);
  border-color: var(--accent);
}
.settings-grid {
  padding-top: .7rem;
}
.workspace-card {
  margin: .2rem 0 .85rem;
  padding: .8rem;
  border-radius: .55rem;
  border: 1px solid var(--line);
  background: var(--panel-raised);
}
.workspace-card .prose {
  color: var(--muted);
  font-size: .89rem;
}
.pipeline-status {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  margin: .85rem 0 .5rem;
  padding: .75rem .85rem;
}
.pipeline-status h3 {
  color: var(--ink);
  font-size: 1rem;
  margin: 0 0 .2rem;
}
.pipeline-status p {
  color: var(--muted);
  margin: 0;
}
.pipeline-progress {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  margin-bottom: .7rem;
  padding: .65rem .8rem;
}
.pipeline-progress-header {
  color: var(--muted);
  display: flex;
  font-size: .85rem;
  justify-content: space-between;
  margin-bottom: .45rem;
}
.pipeline-progress-track {
  background: var(--line);
  border-radius: 999px;
  height: .55rem;
  overflow: hidden;
}
.pipeline-progress-fill {
  background: var(--accent);
  border-radius: 999px;
  height: 100%;
  transition: width .22s ease;
}
.config-accordion {
  margin-top: 1rem;
}
.compact-upload {
  min-height: 6.2rem !important;
  max-height: 7.8rem !important;
}
.compact-upload .upload-container,
.compact-upload [data-testid="dropzone"] {
  min-height: 5rem !important;
  padding: .35rem !important;
}
.compact-upload .upload-container svg,
.compact-upload [data-testid="dropzone"] svg {
  width: 1.2rem !important;
  height: 1.2rem !important;
}
.run-log textarea {
  font-family: "Cascadia Mono", "Consolas", monospace;
  line-height: 1.42;
}
.gradio-container .primary {
  background: var(--accent);
  border-color: var(--accent);
}
.settings-save-button {
  margin-top: .85rem;
}
.shutdown-button {
  margin-top: .55rem;
}
.shutdown-confirmation {
  background: var(--accent-soft);
  border: 1px solid var(--accent);
  border-radius: .55rem;
  margin-top: .6rem;
  padding: .75rem .85rem;
}
.shutdown-confirmation .prose {
  margin-bottom: .6rem;
}
.gradio-container a { color: var(--accent); }
"""


def parser_actions() -> list[argparse.Action]:
    skipped = {"help", "base_config_file", "config_file", "start_stage", "end_stage"}
    return [
        action
        for action in create_parser()._actions
        if action.dest not in skipped
    ]


def flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if key == "ui":
            flattened[key] = value
        elif isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def clean_value(value: Any) -> Any:
    return None if value == {} else value


def defaults(include_screenshots: bool = True) -> dict[str, Any]:
    values = {action.dest: clean_value(action.default) for action in parser_actions()}
    paths = [ROOT / "configs" / "pipelines" / "base.toml"]
    if include_screenshots:
        paths.append(ROOT / "configs" / "pipelines" / "screenshots.toml")
    for path in paths:
        if path.exists():
            values.update(
                {
                    key: clean_value(value)
                    for key, value in flatten_toml(toml.load(path)).items()
                    if key != "ui"
                }
            )
    return values


def text_for_list(value: Any) -> str:
    if not value:
        return ""
    return "\n".join(str(item) for item in value)


def display_value(action: argparse.Action, value: Any) -> Any:
    value = clean_value(value)
    if action.nargs in ("*", "+"):
        return text_for_list(value)
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    return "" if value is None else value


def component_value(action: argparse.Action, value: Any) -> Any:
    shown = display_value(action, value)
    if action.type in (int, float):
        return None if shown == "" else action.type(shown)
    return None if action.choices and shown == "" else shown


def normalize_path_value(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    return os.path.normpath(expanded) if expanded else ""


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]+", value or "") if item.strip()]


def config_value(action: argparse.Action, value: Any) -> Any:
    if action.nargs in ("*", "+"):
        return parse_list(str(value))
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    if value == "" or value is None:
        return None
    if action.dest in PATH_FIELDS:
        return normalize_path_value(str(value))
    if action.type in (int, float):
        return action.type(value)
    return value


def build_config(values: Iterable[Any], actions: list[argparse.Action]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for action, value in zip(actions, values):
        converted = config_value(action, value)
        if converted is not None:
            config[action.dest] = converted
    return config


def workspace_paths(root_value: str) -> dict[str, str]:
    root_path = normalize_path_value(root_value)
    if not root_path:
        return {}
    root = Path(root_path)
    return {
        "src_dir": str(root / "src"),
        "dst_dir": str(root / "dst"),
        "character_ref_dir": str(root / "ref"),
        "log_dir": str(root / "logs"),
    }


def apply_workspace_paths(root_value: str, config: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(config)
    mapped.update(workspace_paths(root_value))
    return mapped


def create_workspace_structure(root_value: str):
    paths = workspace_paths(root_value)
    if not paths:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Enter a workspace root before creating folders.",
        )
    root = Path(normalize_path_value(root_value))
    existing_directories = [
        relative_path for relative_path in WORKSPACE_RESERVED_DIRECTORIES
        if (root / relative_path).is_dir()
    ]
    for relative_path in WORKSPACE_RESERVED_DIRECTORIES:
        path = root / relative_path
        if path.exists() and not path.is_dir():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                f"Cannot create workspace: `{path}` exists but is not a folder.",
            )
    for relative_path in WORKSPACE_DIRECTORIES:
        path = root / relative_path
        path.mkdir(parents=True, exist_ok=True)
    if existing_directories:
        status = (
            f"Workspace ready: `{root}`. Existing folders and their contents were kept; "
            "missing input, reference, and log folders were created. Output folders "
            "are created only when a stage writes data."
        )
    else:
        status = (
            f"Workspace created: `{root}`. Place first-stage input in `src`, "
            "character references in `ref`, and outputs will be written under `dst`."
        )
    return (
        gr.update(value=paths["src_dir"]),
        gr.update(value=paths["dst_dir"]),
        gr.update(value=paths["character_ref_dir"]),
        gr.update(value=paths["log_dir"]),
        status,
    )


def clear_workspace_output(root_value: str) -> str:
    paths = workspace_paths(root_value)
    if not paths:
        return "Enter a workspace root before clearing generated output."
    with PIPELINE_PROCESS_LOCK:
        process = ACTIVE_PIPELINE_PROCESS
        if process is not None and process.poll() is None:
            return "Stop the active pipeline before clearing generated output."
    root = Path(normalize_path_value(root_value))
    dst = Path(paths["dst_dir"])
    if dst.resolve().parent != root.resolve():
        return "Output cleanup stopped because the destination is outside the workspace root."
    if dst.exists() and not dst.is_dir():
        return f"Cannot clear generated output: `{dst}` is not a folder."
    removed = 0
    if dst.exists():
        for item in dst.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
    if removed:
        shutil.rmtree(dst)
        return (
            f"Generated output cleared: `{dst}`. Output folders will be created "
            "when a stage writes data."
        )
    if dst.exists():
        dst.rmdir()
    return (
        f"No generated output to clear in `{dst}`. Output folders will be created "
        "when a stage writes data."
    )


def clean_run_line(line: str) -> str:
    clean_fragments = (
        ANSI_ESCAPE_RE.sub("", line).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    )
    return clean_fragments[-1].strip() if clean_fragments else ""


def append_run_history(history: list[str], line: str) -> None:
    progress_match = TQDM_PROGRESS_RE.search(line)
    if progress_match:
        label = progress_match.group("label").strip()
        for index in range(len(history) - 1, max(-1, len(history) - 10), -1):
            if history[index].startswith(f"{label}:"):
                history[index] = line
                return
    history.append(line)
    del history[:-160]


def run_detail_from_line(line: str) -> str | None:
    match = TQDM_PROGRESS_RE.search(line)
    if not match:
        return None
    return (
        f"{match.group('label').strip()}: {match.group('percent')}% "
        f"({match.group('done')}/{match.group('total')}) [{match.group('timing')}]"
    )


def status_markup(title: str, message: str) -> str:
    return (
        "<div class='pipeline-status'>"
        f"<h3>{html.escape(title)}</h3><p>{html.escape(message)}</p>"
        "</div>"
    )


def progress_markup(
    selected: list[int],
    completed: set[int],
    current: int | None,
    detail: str = "",
    language: str | None = DEFAULT_LANGUAGE,
) -> str:
    language = normalized_language(language)
    total = len(selected)
    complete_count = len(completed.intersection(selected))
    percent = round((complete_count / total) * 100) if total else 0
    current_text = ui_text(language, "waiting")
    if current is not None:
        current_text = f"{ui_text(language, 'current')}: Stage {current} - {stage_details(current, language)[0]}"
    elif total and complete_count == total:
        current_text = ui_text(language, "all_complete")
    label = ui_text(language, "complete_count").format(done=complete_count, total=total)
    if detail:
        current_text = f"{current_text} | {detail}"
    return (
        "<div class='pipeline-progress'>"
        "<div class='pipeline-progress-header'>"
        f"<span>{html.escape(current_text)}</span><span>{html.escape(label)}</span>"
        "</div>"
        "<div class='pipeline-progress-track'>"
        f"<div class='pipeline-progress-fill' style='width:{percent}%'></div>"
        "</div></div>"
    )


def mirror_run_output(line: str) -> None:
    print(line, end="" if line.endswith("\n") else "\n", flush=True)


def process_output_with_heartbeat(process: subprocess.Popen[str]):
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    while True:
        try:
            message = messages.get(timeout=1)
        except queue.Empty:
            yield ""
            continue
        if message is None:
            return
        yield message


def stage_values(selected: Iterable[Any] | None) -> list[int]:
    return sorted({int(stage) for stage in (selected or [])})


def stage_ranges(stages: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for stage in sorted(stages):
        if ranges and stage == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], stage)
        else:
            ranges.append((stage, stage))
    return ranges


def stage_tab_updates(selected: Iterable[Any] | None, language: str | None = DEFAULT_LANGUAGE):
    enabled = set(stage_values(selected))
    return [
        gr.update(visible=number in enabled, label=group_label(f"Stage {number} - {STAGES[number][0]}", language))
        for number in STAGES
    ]


def save_configuration(
    language: str,
    workspace_root: str,
    selected: list[str],
    *values: Any,
):
    if language not in UI_TEXT:
        values = (selected, *values)
        selected = workspace_root
        workspace_root = language
        language = DEFAULT_LANGUAGE
    language = normalized_language(language)
    GLOBAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config = apply_workspace_paths(workspace_root, build_config(values, ACTIONS))
    config["ui"] = {
        "enabled_stages": stage_values(selected),
        "fixed_port": PORT,
        "language": language,
        "workspace_root": normalize_path_value(workspace_root),
    }
    with GLOBAL_CONFIG.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)
    relative_path = str(GLOBAL_CONFIG.relative_to(ROOT)).replace("\\", "/")
    return (
        f"{ui_text(language, 'config_saved')} `{relative_path}`",
        str(GLOBAL_CONFIG),
    )


def configuration_state(uploaded_path: str | None = None):
    config = defaults()
    ui_config: dict[str, Any] = {}
    path = Path(uploaded_path) if uploaded_path else GLOBAL_CONFIG
    language = DEFAULT_LANGUAGE
    if path and path.exists():
        loaded = flatten_toml(toml.load(path))
        ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
        language = normalized_language(ui_config.get("language", DEFAULT_LANGUAGE))
        config.update({key: clean_value(value) for key, value in loaded.items()})
        source = "config_loaded_imported" if uploaded_path else "config_loaded_global"
        status = f"{ui_text(language, source)} `{path.name}`"
    elif uploaded_path:
        status = ui_text(language, "config_invalid")
    else:
        status = ui_text(language, "config_missing")
    selected = [str(stage) for stage in ui_config.get("enabled_stages", DEFAULT_ENABLED_STAGES)]
    workspace_root = ui_config.get("workspace_root", "")
    return config, selected, workspace_root, language, status


def load_configuration(uploaded_path: str | None = None):
    config, selected, workspace_root, language, status = configuration_state(uploaded_path)
    updates = [
        gr.update(value=language),
        gr.update(value=workspace_root),
        gr.update(value=selected, choices=stage_choices(language)),
    ]
    updates.extend(gr.update(value=component_value(action, config.get(action.dest))) for action in ACTIONS)
    updates.append(status)
    return updates


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            text=True,
        )
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def stop_pipeline() -> str:
    global ACTIVE_PIPELINE_PROCESS
    with PIPELINE_PROCESS_LOCK:
        process = ACTIVE_PIPELINE_PROCESS
        if process is None or process.poll() is not None:
            return "No pipeline process is currently running."
        PIPELINE_STOP_REQUESTED.set()
        terminate_process_tree(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return "Pipeline stopped."


def shutdown_server() -> str:
    stop_pipeline()

    def exit_process() -> None:
        time.sleep(0.35)
        os._exit(0)

    threading.Thread(target=exit_process, daemon=True).start()
    return "Server shutdown requested. This browser connection will close shortly."


def show_shutdown_confirmation():
    return gr.update(visible=True)


def hide_shutdown_confirmation():
    return gr.update(visible=False)


def execute_config(stages: list[int], config: dict[str, Any]):
    global ACTIVE_PIPELINE_PROCESS
    completed: set[int] = set()
    current_stage: int | None = None
    detail = ""
    clustering_started: float | None = None

    def update(title: str, message: str, history: list[str]):
        return (
            status_markup(title, message),
            progress_markup(stages, completed, current_stage, detail),
            "\n".join(history),
        )

    if not stages:
        yield (
            status_markup("No stages selected", "Enable at least one stage before running."),
            progress_markup([], set(), None),
            "No stage selected. Enable at least one checkbox.",
        )
        return
    if not PIPELINE_RUN_LOCK.acquire(blocking=False):
        yield (
            status_markup("Pipeline already running", "Stop the active run before starting another."),
            progress_markup(stages, set(), None),
            "A pipeline run is already active. Stop it before starting another run.",
        )
        return

    PIPELINE_STOP_REQUESTED.clear()
    try:
        SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("w", encoding="utf-8") as handle:
            toml.dump(config, handle)

        history = [
            f"Runtime configuration: {RUNTIME_CONFIG.relative_to(ROOT)}",
            f"Stages: {', '.join(str(stage) for stage in stages)}",
        ]
        yield update("Pipeline starting", f"Preparing {len(stages)} selected stage(s).", history)
        for start_stage, end_stage in stage_ranges(stages):
            if PIPELINE_STOP_REQUESTED.is_set():
                append_run_history(history, "Pipeline stopped by user.")
                yield update("Pipeline stopped", "Processing was stopped by the user.", history)
                return
            titles = ", ".join(
                f"{stage}: {STAGES[stage][0]}" for stage in range(start_stage, end_stage + 1)
            )
            append_run_history(history, f"--- Stages {titles} ---")
            yield update("Pipeline running", f"Starting {titles}.", history)
            command = [
                sys.executable,
                str(ROOT / "automatic_pipeline.py"),
                "--base_config_file",
                str(RUNTIME_CONFIG),
                "--start_stage",
                str(start_stage),
                "--end_stage",
                str(end_stage),
            ]
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **popen_options,
            )
            with PIPELINE_PROCESS_LOCK:
                ACTIVE_PIPELINE_PROCESS = process
            for raw_line in process_output_with_heartbeat(process):
                if not raw_line:
                    if clustering_started is not None:
                        elapsed = int(time.monotonic() - clustering_started)
                        detail = (
                            f"Clustering active; elapsed {elapsed}s; "
                            "ETA unavailable during OPTICS fitting."
                        )
                        yield update(
                            "Pipeline running",
                            "Classification clustering is still processing.",
                            history,
                        )
                    continue
                mirror_run_output(raw_line)
                line = clean_run_line(raw_line)
                if not line:
                    continue
                stage_match = STAGE_START_RE.search(line)
                if stage_match:
                    next_stage = int(stage_match.group(1))
                    if current_stage is not None and current_stage != next_stage:
                        completed.add(current_stage)
                    current_stage = next_stage
                    detail = ""
                    clustering_started = None
                parsed_detail = run_detail_from_line(line)
                if parsed_detail:
                    detail = parsed_detail
                if "Clustering OPTICS:" in line:
                    clustering_started = time.monotonic()
                    detail = "Clustering active; ETA unavailable during OPTICS fitting."
                append_run_history(history, line)
                running_stage = current_stage if current_stage is not None else start_stage
                yield update(
                    "Pipeline running",
                    f"Processing Stage {running_stage} - {STAGES[running_stage][0]}.",
                    history,
                )
            return_code = process.wait()
            with PIPELINE_PROCESS_LOCK:
                if ACTIVE_PIPELINE_PROCESS is process:
                    ACTIVE_PIPELINE_PROCESS = None
            if PIPELINE_STOP_REQUESTED.is_set():
                append_run_history(history, "Pipeline stopped by user.")
                yield update("Pipeline stopped", "Processing was stopped by the user.", history)
                return
            if return_code != 0:
                append_run_history(
                    history,
                    f"Stages {start_stage}-{end_stage} failed with exit code {return_code}."
                )
                yield update("Pipeline failed", "Review the run log for the reported error.", history)
                return
            completed.update(stage for stage in stages if start_stage <= stage <= end_stage)
            current_stage = None
            detail = ""
            clustering_started = None
            append_run_history(history, f"Stages {start_stage}-{end_stage} completed.")
            yield update("Pipeline running", "Continuing with remaining selected stages.", history)
        append_run_history(history, "All selected stages completed.")
        yield update(
            "Pipeline complete",
            f"Completed {len(stages)} selected stage(s).",
            history,
        )
    finally:
        with PIPELINE_PROCESS_LOCK:
            process = ACTIVE_PIPELINE_PROCESS
            ACTIVE_PIPELINE_PROCESS = None
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        PIPELINE_STOP_REQUESTED.clear()
        PIPELINE_RUN_LOCK.release()


def run_selected_stages(selected: list[str], workspace_root: str, *values: Any):
    stages = stage_values(selected)
    if normalize_path_value(workspace_root):
        *_updates, workspace_status = create_workspace_structure(workspace_root)
        if workspace_status.startswith("Cannot create workspace"):
            yield (
                status_markup("Workspace unavailable", workspace_status),
                progress_markup(stages, set(), None),
                workspace_status,
            )
            return
    config = apply_workspace_paths(workspace_root, build_config(values, ACTIONS))
    yield from execute_config(stages, config)


def run_saved_profile(config_path: str, stages_text: str):
    requested_path = (ROOT / config_path).resolve()
    if ROOT not in requested_path.parents or not requested_path.exists():
        yield "The profile file must exist inside the project directory."
        return
    loaded = flatten_toml(toml.load(requested_path))
    ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
    if stages_text.strip():
        stages = stage_values(parse_list(stages_text))
    else:
        stages = stage_values(ui_config.get("enabled_stages", []))
    workspace_root = ui_config.get("workspace_root", "")
    if normalize_path_value(workspace_root):
        *_updates, workspace_status = create_workspace_structure(str(workspace_root))
        if workspace_status.startswith("Cannot create workspace"):
            yield workspace_status
            return
        loaded = apply_workspace_paths(str(workspace_root), loaded)
    for _status, _progress, log in execute_config(stages, loaded):
        yield log


def field_label(action: argparse.Action) -> str:
    return f"--{action.dest}"


def control_update(action: argparse.Action, language: str | None = DEFAULT_LANGUAGE):
    info = setting_info(action, language)
    if action.nargs in ("*", "+"):
        info = f"{info} {ui_text(language, 'list_help')}"
    return gr.update(label=field_label(action), info=info)


def make_component(action: argparse.Action, value: Any, language: str | None = DEFAULT_LANGUAGE):
    label = f"--{action.dest}"
    info = setting_info(action, language)
    shown = component_value(action, value)
    if isinstance(action, argparse._StoreTrueAction):
        return gr.Checkbox(label=label, value=shown, info=info)
    if action.choices:
        return gr.Dropdown(label=label, choices=list(action.choices), value=shown, info=info)
    if action.type in (int, float):
        precision = 0 if action.type is int else None
        return gr.Number(label=label, value=None if shown == "" else shown, precision=precision, info=info)
    if action.nargs in ("*", "+"):
        return gr.Textbox(label=label, value=shown, lines=2, info=f"{info} {ui_text(language, 'list_help')}")
    return gr.Textbox(label=label, value=shown, info=info)


def interface_language_updates(language: str, selected: Iterable[Any] | None):
    language = normalized_language(language)
    selected_values = [str(stage) for stage in stage_values(selected)]
    updates = [
        gr.update(value=f"<p class='panel-label'>{ui_text(language, 'workflow')}</p>"),
        gr.update(
            label=ui_text(language, "stages_to_run"),
            info=ui_text(language, "stages_info"),
            choices=stage_choices(language),
            value=selected_values,
        ),
        gr.update(value=stage_guide_html(language)),
        gr.update(value=ui_text(language, "run_pipeline")),
        gr.update(value=ui_text(language, "stop_pipeline")),
        gr.update(value=status_markup(ui_text(language, "ready"), ui_text(language, "ready_message"))),
        gr.update(value=progress_markup(stage_values(selected), set(), None, language=language)),
        gr.update(label=ui_text(language, "run_log")),
        gr.update(label=ui_text(language, "language"), info=ui_text(language, "language_info")),
        gr.update(
            label=ui_text(language, "workspace_root"),
            placeholder=ui_text(language, "workspace_placeholder"),
            info=ui_text(language, "workspace_info"),
        ),
        gr.update(value=ui_text(language, "create_workspace")),
        gr.update(value=ui_text(language, "clear_output")),
        gr.update(value=ui_text(language, "workspace_help")),
        gr.update(label=ui_text(language, "import_configuration")),
        gr.update(value=ui_text(language, "load_configuration")),
        gr.update(value=ui_text(language, "configuration_help")),
        gr.update(value=ui_text(language, "save_configuration")),
        gr.update(label=ui_text(language, "export_settings")),
        gr.update(
            value=(
                f"{ui_text(language, 'web_interface')}: `http://127.0.0.1:{PORT}` "
                f"({ui_text(language, 'fixed_port')})."
            )
        ),
        gr.update(value=f"<p class='settings-label'>{ui_text(language, 'settings_by_stage')}</p>"),
    ]
    enabled = set(stage_values(selected_values))
    updates.extend(
        gr.update(
            label=group_label(group_name, language),
            visible=GROUP_STAGES.get(group_name) is None or GROUP_STAGES[group_name] in enabled,
        )
        for group_name in FIELD_GROUPS
    )
    updates.extend(gr.update(value=stage_details(number, language)[1]) for number in STAGES)
    updates.extend(control_update(action, language) for action in ACTIONS)
    updates.extend(
        [
            gr.update(value=ui_text(language, "save_configuration")),
            gr.update(value=ui_text(language, "shutdown")),
            gr.update(value=ui_text(language, "shutdown_confirm")),
            gr.update(value=ui_text(language, "cancel")),
            gr.update(value=ui_text(language, "confirm_shutdown")),
        ]
    )
    return updates


def build_interface() -> gr.Blocks:
    initial, initial_stages, initial_workspace_root, initial_language, initial_status = configuration_state()
    controls: list[Any] = []
    all_setting_tabs: list[Any] = []
    stage_tabs: list[Any] = []
    stage_descriptions: list[Any] = []
    grouped = {name: set(keys) for name, keys in FIELD_GROUPS.items()}
    known_fields = {key for keys in grouped.values() for key in keys}

    with gr.Blocks(
        title="Anime2SD Frame Lab",
        elem_classes="workspace",
    ) as demo:
        with gr.Column(elem_classes="run-panel"):
            workflow_label = gr.HTML(f"<p class='panel-label'>{ui_text(initial_language, 'workflow')}</p>")
            stage_selector = gr.CheckboxGroup(
                choices=stage_choices(initial_language),
                value=initial_stages,
                label=ui_text(initial_language, "stages_to_run"),
                info=ui_text(initial_language, "stages_info"),
            )
            with gr.Accordion(ui_text(initial_language, "stage_guide"), open=False):
                stage_guide = gr.HTML(stage_guide_html(initial_language))
            with gr.Row():
                run_button = gr.Button(ui_text(initial_language, "run_pipeline"), variant="primary")
                stop_button = gr.Button(ui_text(initial_language, "stop_pipeline"), variant="stop")
            run_status = gr.HTML(
                status_markup(ui_text(initial_language, "ready"), ui_text(initial_language, "ready_message"))
            )
            stage_progress = gr.HTML(
                progress_markup(stage_values(initial_stages), set(), None, language=initial_language)
            )
            output = gr.Textbox(
                label=ui_text(initial_language, "run_log"), lines=18, interactive=False, elem_classes="run-log"
            )

        with gr.Accordion(ui_text(initial_language, "configuration"), open=False, elem_classes="run-panel config-accordion"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="workspace-card"):
                        language_selector = gr.Dropdown(
                            label=ui_text(initial_language, "language"),
                            choices=LANGUAGE_CHOICES,
                            value=initial_language,
                            info=ui_text(initial_language, "language_info"),
                        )
                        workspace_root = gr.Textbox(
                            label=ui_text(initial_language, "workspace_root"),
                            value=initial_workspace_root,
                            placeholder=ui_text(initial_language, "workspace_placeholder"),
                            info=ui_text(initial_language, "workspace_info"),
                        )
                        create_workspace_button = gr.Button(ui_text(initial_language, "create_workspace"))
                        clear_output_button = gr.Button(ui_text(initial_language, "clear_output"), variant="stop")
                        workspace_help = gr.Markdown(ui_text(initial_language, "workspace_help"))
                with gr.Column():
                    uploaded = gr.File(
                        label=ui_text(initial_language, "import_configuration"),
                        file_types=[".toml"],
                        type="filepath",
                        elem_classes="compact-upload",
                    )
                    load_button = gr.Button(ui_text(initial_language, "load_configuration"))
                    configuration_help = gr.Markdown(ui_text(initial_language, "configuration_help"))
                    with gr.Row():
                        save_button = gr.Button(ui_text(initial_language, "save_configuration"), variant="primary")
                        export_button = gr.DownloadButton(
                            ui_text(initial_language, "export_settings"),
                            value=str(GLOBAL_CONFIG) if GLOBAL_CONFIG.exists() else None,
                        )
            status = gr.Markdown(
                f"{ui_text(initial_language, 'web_interface')}: `http://127.0.0.1:{PORT}` ({ui_text(initial_language, 'fixed_port')}).\n\n{initial_status}"
            )

        settings_label = gr.HTML(f"<p class='settings-label'>{ui_text(initial_language, 'settings_by_stage')}</p>")
        with gr.Tabs(elem_classes="settings-tabs"):
            for group_name, fields in FIELD_GROUPS.items():
                stage_number = GROUP_STAGES.get(group_name)
                visible = stage_number is None or str(stage_number) in initial_stages
                with gr.Tab(group_label(group_name, initial_language), visible=visible) as tab:
                    all_setting_tabs.append(tab)
                    if stage_number is not None:
                        stage_tabs.append(tab)
                        stage_descriptions.append(gr.Markdown(stage_details(stage_number, initial_language)[1]))
                    group_actions = [item for item in ACTIONS if item.dest in fields]
                    with gr.Column(elem_classes="settings-grid"):
                        for offset in range(0, len(group_actions), 3):
                            with gr.Row():
                                for action in group_actions[offset:offset + 3]:
                                    controls.append(make_component(action, initial.get(action.dest), initial_language))

            remaining = [action for action in ACTIONS if action.dest not in known_fields]
            if remaining:
                with gr.Tab(ui_text(initial_language, "additional_settings")):
                    with gr.Column(elem_classes="settings-grid"):
                        for offset in range(0, len(remaining), 3):
                            with gr.Row():
                                for action in remaining[offset:offset + 3]:
                                    controls.append(make_component(action, initial.get(action.dest), initial_language))

        save_settings_button = gr.Button(
            ui_text(initial_language, "save_configuration"),
            variant="primary",
            elem_classes="settings-save-button",
        )
        shutdown_button = gr.Button(ui_text(initial_language, "shutdown"), variant="stop", elem_classes="shutdown-button")
        with gr.Column(visible=False, elem_classes="shutdown-confirmation") as shutdown_confirmation:
            shutdown_text = gr.Markdown(ui_text(initial_language, "shutdown_confirm"))
            with gr.Row():
                cancel_shutdown_button = gr.Button(ui_text(initial_language, "cancel"))
                confirm_shutdown_button = gr.Button(ui_text(initial_language, "confirm_shutdown"), variant="stop")

        ordered_controls = {action.dest: control for action, control in zip(
            [item for group in FIELD_GROUPS.values() for item in ACTIONS if item.dest in group] + remaining,
            controls,
        )}
        controls_in_action_order = [ordered_controls[action.dest] for action in ACTIONS]
        workspace_path_outputs = [
            ordered_controls["src_dir"],
            ordered_controls["dst_dir"],
            ordered_controls["character_ref_dir"],
            ordered_controls["log_dir"],
        ]

        create_workspace_button.click(
            create_workspace_structure,
            inputs=workspace_root,
            outputs=[*workspace_path_outputs, status],
            api_name="create_workspace",
        )
        clear_output_button.click(
            clear_workspace_output,
            inputs=workspace_root,
            outputs=status,
            api_name="clear_workspace_output",
        )

        save_button.click(
            save_configuration,
            inputs=[language_selector, workspace_root, stage_selector, *controls_in_action_order],
            outputs=[status, export_button],
            api_name="save_configuration",
        )
        save_settings_button.click(
            save_configuration,
            inputs=[language_selector, workspace_root, stage_selector, *controls_in_action_order],
            outputs=[status, export_button],
        )
        language_outputs = [
            workflow_label,
            stage_selector,
            stage_guide,
            run_button,
            stop_button,
            run_status,
            stage_progress,
            output,
            language_selector,
            workspace_root,
            create_workspace_button,
            clear_output_button,
            workspace_help,
            uploaded,
            load_button,
            configuration_help,
            save_button,
            export_button,
            status,
            settings_label,
            *all_setting_tabs,
            *stage_descriptions,
            *controls_in_action_order,
            save_settings_button,
            shutdown_button,
            shutdown_text,
            cancel_shutdown_button,
            confirm_shutdown_button,
        ]
        language_selector.change(
            interface_language_updates,
            inputs=[language_selector, stage_selector],
            outputs=language_outputs,
        )
        load_button.click(
            load_configuration,
            inputs=uploaded,
            outputs=[language_selector, workspace_root, stage_selector, *controls_in_action_order, status],
        ).then(
            interface_language_updates,
            inputs=[language_selector, stage_selector],
            outputs=language_outputs,
        )
        api_load_button = gr.Button(visible=False)
        api_load_button.click(
            load_configuration,
            outputs=[language_selector, workspace_root, stage_selector, *controls_in_action_order, status],
            api_name="load_global_configuration",
        ).then(
            interface_language_updates,
            inputs=[language_selector, stage_selector],
            outputs=language_outputs,
        )
        stage_selector.change(stage_tab_updates, inputs=[stage_selector, language_selector], outputs=stage_tabs)
        run_event = run_button.click(
            run_selected_stages,
            inputs=[stage_selector, workspace_root, *controls_in_action_order],
            outputs=[run_status, stage_progress, output],
            api_name="run_from_form",
            concurrency_limit=1,
            concurrency_id="pipeline_run",
        )
        stop_button.click(
            stop_pipeline,
            outputs=output,
            api_name="stop_pipeline",
            cancels=[run_event],
            concurrency_limit=None,
            concurrency_id="pipeline_control",
        )
        shutdown_button.click(
            show_shutdown_confirmation,
            outputs=shutdown_confirmation,
        )
        cancel_shutdown_button.click(
            hide_shutdown_confirmation,
            outputs=shutdown_confirmation,
        )
        confirm_shutdown_button.click(
            shutdown_server,
            outputs=status,
            api_name="shutdown_server",
            concurrency_limit=None,
        )
        api_profile = gr.Textbox(value="configs/ui/configuration.toml", visible=False, container=False)
        api_stages = gr.Textbox(value="", visible=False, container=False)
        api_output = gr.Textbox(visible=False, container=False)
        api_button = gr.Button(visible=False)
        api_button.click(
            run_saved_profile,
            inputs=[api_profile, api_stages],
            outputs=api_output,
            api_name="run_saved_profile",
        )
    return demo


ACTIONS = parser_actions()


if __name__ == "__main__":
    build_interface().queue().launch(
        server_name="127.0.0.1",
        server_port=PORT,
        show_error=True,
        css=STYLE,
    )
