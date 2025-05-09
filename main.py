import os
import glob
import json
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from loguru import logger

custom_torch_hub_dir = "/mnt/disk01/data/av.beliaev5/torch_home"
os.makedirs(custom_torch_hub_dir, exist_ok=True)  # Убедимся, что директория существует
os.environ["TORCH_HOME"] = custom_torch_hub_dir

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ (Минимальные) ---
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# --- ИМПОРТ КОНФИГУРАЦИИ ---
try:
    from config import (
        PATH_TO_SPLIT,
        PATH_TO_PREDICTIONS,
        SIZE_REQUIREMENTS,
        OUR_TO_MODEL_CLASSES,
        MODEL_MAPPING,
        CLASSES_GROUPS,
        CVAT_LINK,
    )
except ImportError:
    logger.critical(
        "Критическая ошибка: Не удалось импортировать конфигурацию из config.py."
    )
    # Fallback to dummy config for demonstration if config.py is missing
    logger.warning("Using DUMMY config values as config.py was not found.")
    PATH_TO_SPLIT = "dummy_split_data"
    PATH_TO_PREDICTIONS = "dummy_predictions.json"
    SIZE_REQUIREMENTS = {"person": (30, 80)}
    OUR_TO_MODEL_CLASSES = {"person": {"person", "pedestrian"}, "car": {"car"}}
    MODEL_MAPPING = {
        "person": {"person_model_v1", "human"},
        "car": {"vehicle", "car_model_v2"},
    }
    CLASSES_GROUPS = {"person_related": ["person", "safety_vest", "helmet"]}
    CVAT_LINK = "http://localhost:8080"
    os.makedirs(PATH_TO_SPLIT, exist_ok=True)
    # Create a dummy predictions file
    dummy_preds_data = {
        "image1.jpg": [
            {"label": "person_model_v1", "score": 0.9, "bbox": [10, 10, 60, 110]},
            {"label": "car_model_v2", "score": 0.8, "bbox": [100, 100, 200, 200]},
        ],
        "image2.jpg": [
            {"label": "person_model_v1", "score": 0.95, "bbox": [20, 20, 70, 120]}
        ],
    }
    with open(PATH_TO_PREDICTIONS, "w") as f_dummy_preds:
        json.dump(dummy_preds_data, f_dummy_preds)

    # Create a dummy CSV file for testing
    dummy_csv_data = {
        "image_path": [f"{PATH_TO_SPLIT}/image1.jpg", f"{PATH_TO_SPLIT}/image2.jpg"],
        "image_name": ["image1.jpg", "image2.jpg"],
        "image_width": [640, 640],
        "image_height": [480, 480],
        "instance_label": ["person", "person"],
        "bbox_x_tl": [10, 20],
        "bbox_y_tl": [10, 20],
        "bbox_x_br": [50, 60],
        "bbox_y_br": [100, 110],
        "task_id": [1, 1],
        "job_id": [101, 101],
        "image_id": [1, 2],
    }
    dummy_df = pd.DataFrame(dummy_csv_data)
    dummy_df.to_csv(os.path.join(PATH_TO_SPLIT, "person.csv"), index=False)
    # Create dummy image files (empty, just for os.path.exists)
    open(f"{PATH_TO_SPLIT}/image1.jpg", "w").close()
    open(f"{PATH_TO_SPLIT}/image2.jpg", "w").close()
    # exit() # If you want to stop if config.py is not found, uncomment this

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}
SKIP_CLASSES_FOR_IOU_EVAL = {
    "safety",
    "no_safety",
    "chin_strap",
    "chin_strap_off",
    "glasses",
    "glasses_off",
}
INCLUSION_THRESHOLD_GT_COVERED = 0.8


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_group_for_label(label: str) -> str:
    for group, labels in CLASSES_GROUPS.items():
        if label in labels:
            return group
    return label


def map_prediction_label_to_canonical(pred_label_from_model: str) -> str:
    for target, sources in MODEL_MAPPING.items():
        if pred_label_from_model in sources:
            return target
    return pred_label_from_model


def map_our_gt_label_to_model_label_set(our_gt_label: str) -> set:
    return OUR_TO_MODEL_CLASSES.get(our_gt_label, {our_gt_label})


def get_abs_bbox_from_normalized(norm_bbox, img_width, img_height):
    x, y, w, h_norm = norm_bbox
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h_norm) * img_height
    return [x1, y1, x2, y2]


def calculate_area(bbox_abs):
    if None in bbox_abs or bbox_abs[2] < bbox_abs[0] or bbox_abs[3] < bbox_abs[1]:
        return 0.0
    return (bbox_abs[2] - bbox_abs[0]) * (bbox_abs[3] - bbox_abs[1])


def calculate_intersection_area(bbox1_abs, bbox2_abs):
    x_left = max(bbox1_abs[0], bbox2_abs[0])
    y_top = max(bbox1_abs[1], bbox2_abs[1])
    x_right = min(bbox1_abs[2], bbox2_abs[2])
    y_bottom = min(bbox1_abs[3], bbox2_abs[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


# --- КАСТОМНАЯ ОЦЕНКА ПО ВХОЖДЕНИЮ ---
def evaluate_by_inclusion(
    dataset, gt_field="ground_truth", pred_field="predictions", gt_covered_threshold=0.8
):
    logger.info(f"Оценка по вхождению для '{dataset.name}'...")
    view = dataset.view()
    for sample in view.iter_samples(autosave=True, progress=True):
        if sample.metadata is None or sample.metadata.width is None:
            try:
                sample.compute_metadata(overwrite=False)
            except Exception:
                logger.warning(
                    f"Пропуск сэмпла в evaluate_by_inclusion (нет метаданных): {sample.filepath}"
                )
                continue
            if sample.metadata is None or sample.metadata.width is None:
                logger.warning(
                    f"Пропуск сэмпла в evaluate_by_inclusion (нет метаданных): {sample.filepath}"
                )
                continue
        img_w, img_h = sample.metadata.width, sample.metadata.height
        gts = (
            sample[gt_field].detections
            if sample[gt_field] and sample[gt_field].detections
            else []
        )
        preds = (
            sample[pred_field].detections
            if sample[pred_field] and sample[pred_field].detections
            else []
        )
        abs_gts = [
            {"d": gt, "bb": get_abs_bbox_from_normalized(gt.bounding_box, img_w, img_h)}
            for gt in gts
        ]
        abs_preds = [
            {"d": p, "bb": get_abs_bbox_from_normalized(p.bounding_box, img_w, img_h)}
            for p in preds
        ]
        for gt_item in abs_gts:
            gt_d, gt_bb = gt_item["d"], gt_item["bb"]
            gt_area = calculate_area(gt_bb)
            gt_d["max_pred_inclusion_in_gt"] = 0.0
            gt_d["gt_covered_by_inclusion"] = False
            if gt_area > 0 and abs_preds:
                for pred_item in abs_preds:
                    incl = calculate_intersection_area(gt_bb, pred_item["bb"]) / gt_area
                    if incl > gt_d["max_pred_inclusion_in_gt"]:
                        gt_d["max_pred_inclusion_in_gt"] = incl
                if gt_d["max_pred_inclusion_in_gt"] >= gt_covered_threshold:
                    gt_d["gt_covered_by_inclusion"] = True
        for pred_item in abs_preds:
            pred_d, pred_bb = pred_item["d"], pred_item["bb"]
            pred_area = calculate_area(pred_bb)
            pred_d["max_gt_inclusion_in_pred"] = 0.0
            pred_d["max_gt_coverage_by_pred"] = 0.0
            if pred_area > 0 and abs_gts:
                for gt_item in abs_gts:
                    inter = calculate_intersection_area(gt_item["bb"], pred_bb)
                    if (
                        pred_area > 0
                        and inter / pred_area > pred_d["max_gt_inclusion_in_pred"]
                    ):
                        pred_d["max_gt_inclusion_in_pred"] = inter / pred_area
                    gt_ac = calculate_area(gt_item["bb"])
                    if gt_ac > 0 and inter / gt_ac > pred_d["max_gt_coverage_by_pred"]:
                        pred_d["max_gt_coverage_by_pred"] = inter / gt_ac
    logger.info(f"Оценка по вхождению для '{dataset.name}' завершена.")


# --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ПАТЧЕЙ ---
def compute_and_save_patch_embeddings(
    dataset_or_view,
    zoo_model_name,
    patches_field="ground_truth",
    embeddings_field_on_detection="embedding",
):
    if not dataset_or_view.has_sample_field(patches_field):
        logger.warning(
            f"Поле '{patches_field}' не найдено в '{dataset_or_view.name}'. Пропуск эмбеддингов патчей."
        )
        return

    logger.info(
        f"Вычисление эмбеддингов для объектов из '{patches_field}' в '{dataset_or_view.name}' используя модель Zoo: '{zoo_model_name}'."
    )
    logger.info(
        f"Результат будет сохранен в поле '{embeddings_field_on_detection}' каждого объекта Detection из поля '{patches_field}'."
    )

    try:
        # Check if there are any detections in the specified patches_field
        # This query is more efficient than iterating through samples
        if dataset_or_view.count(f"{patches_field}.detections") == 0:
            logger.info(
                f"В поле '{patches_field}' датасета '{dataset_or_view.name}' нет объектов. Пропуск эмбеддингов патчей."
            )
            return

        model_instance = foz.load_zoo_model(zoo_model_name)

        dataset_or_view.compute_embeddings(
            model_instance,
            embeddings_field=embeddings_field_on_detection,
            patches_field=patches_field,
            batch_size=4,  # Adjust batch_size based on your VRAM
        )

        logger.info(
            f"Эмбеддинги патчей успешно вычислены для '{dataset_or_view.name}'."
        )
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()  # Save changes to the field schema and data
    except Exception as e:
        logger.exception(
            f"Ошибка при вычислении эмбеддингов патчей для '{dataset_or_view.name}' с моделью '{zoo_model_name}'."
        )
        logger.error(
            f"Убедитесь, что имя модели ('{zoo_model_name}') корректно, есть интернет, и зависимости установлены."
        )


# --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ИЗОБРАЖЕНИЙ ---
def compute_and_save_image_embeddings(
    dataset_or_view,
    zoo_model_name,
    embeddings_field_on_sample="image_embedding",  # Field on the sample itself
):
    logger.info(
        f"Вычисление эмбеддингов для изображений в '{dataset_or_view.name}' используя модель Zoo: '{zoo_model_name}'."
    )
    logger.info(
        f"Результат будет сохранен в поле '{embeddings_field_on_sample}' каждого сэмпла."
    )

    try:
        if dataset_or_view.count() == 0:
            logger.info(
                f"Датасет '{dataset_or_view.name}' не содержит сэмплов. Пропуск эмбеддингов изображений."
            )
            return

        model_instance = foz.load_zoo_model(zoo_model_name)

        dataset_or_view.compute_embeddings(
            model_instance,
            embeddings_field=embeddings_field_on_sample,
            # No patches_field here, so it computes for the whole image
            batch_size=4,  # Adjust batch_size based on your VRAM
        )

        logger.info(
            f"Эмбеддинги изображений успешно вычислены для '{dataset_or_view.name}'."
        )
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()  # Save changes to the field schema and data
    except Exception as e:
        logger.exception(
            f"Ошибка при вычислении эмбеддингов изображений для '{dataset_or_view.name}' с моделью '{zoo_model_name}'."
        )
        logger.error(
            f"Убедитесь, что имя модели ('{zoo_model_name}') корректно, есть интернет, и зависимости установлены."
        )


# --- ЗАГРУЗКА ДАННЫХ И СОЗДАНИЕ ДАТАСЕТА ДЛЯ КЛАССА ---
def load_class_dataset_from_csv(csv_file, all_predictions_dict, config_params):
    our_class_name_from_csv = os.path.splitext(os.path.basename(csv_file))[0]
    logger.info(
        f"Обработка: {our_class_name_from_csv} (из {os.path.basename(csv_file)})"
    )
    dataset_name = our_class_name_from_csv
    if dataset_name in fo.list_datasets():
        logger.info(
            f"Датасет '{dataset_name}' уже существует. Удаление и создание заново."
        )
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name, persistent=True)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Ошибка чтения {csv_file}: {e}")
        return None
    df = df.dropna(subset=["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"])
    req_cols = {
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "instance_label",
        "task_id",
        "job_id",
        "image_id",
    }
    if not req_cols.issubset(df.columns):
        logger.error(
            f"CSV {csv_file} не содержит колонки: {req_cols - set(df.columns)}. Пропуск."
        )
        return None

    target_model_labels = map_our_gt_label_to_model_label_set(our_class_name_from_csv)
    display_gt_label = (
        list(target_model_labels)[0] if target_model_labels else our_class_name_from_csv
    )

    img_data = {}
    for _, r in df.iterrows():
        img_p, img_n, gt_l = r["image_path"], r["image_name"], r["instance_label"]
        if gt_l != our_class_name_from_csv or not os.path.exists(img_p):
            if gt_l == our_class_name_from_csv and not os.path.exists(img_p):
                logger.warning(
                    f"Файл изображения не найден: {img_p} для класса {our_class_name_from_csv}. Пропуск строки."
                )
            continue
        try:
            w, h = int(r["image_width"]), int(r["image_height"])
            if w <= 0 or h <= 0:
                raise ValueError("Image dimensions must be positive")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Размеры для {img_n} в {csv_file} некорректны ({e}). Пропуск строки."
            )
            continue

        if img_p not in img_data:
            img_data[img_p] = {"gts": [], "preds": [], "s": (w, h), "n": img_n}
        x1, y1, x2, y2 = r["bbox_x_tl"], r["bbox_y_tl"], r["bbox_x_br"], r["bbox_y_br"]
        img_data[img_p]["gts"].append(
            {
                "label": display_gt_label,
                "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                "box_width_abs": float(x2 - x1),
                "box_height_abs": float(y2 - y1),
                "cvat_task": config_params["CVAT_LINK"]
                + f'/tasks/{r["task_id"]}/jobs/{int(r["job_id"])}?frame={r["image_id"]}',
            }
        )

    for img_p, data in img_data.items():
        w, h = data["s"]
        for pred_raw in all_predictions_dict.get(data["n"], []):
            if not all(k in pred_raw for k in ["label", "score", "bbox"]):
                continue
            pred_l_canon = map_prediction_label_to_canonical(pred_raw["label"])
            if pred_l_canon not in target_model_labels:
                continue
            grp = get_group_for_label(pred_l_canon)
            if grp in config_params["SIZE_REQUIREMENTS"]:
                mw, mh = config_params["SIZE_REQUIREMENTS"][grp]
                x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred_raw["bbox"]
                if (x2p_abs - x1p_abs) < mw or (y2p_abs - y1p_abs) < mh:
                    continue
            x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred_raw["bbox"]
            data["preds"].append(
                {
                    "label": pred_l_canon,
                    "bounding_box": [
                        x1p_abs / w,
                        y1p_abs / h,
                        (x2p_abs - x1p_abs) / w,
                        (y2p_abs - y1p_abs) / h,
                    ],
                    "confidence": pred_raw["score"],
                    "box_width_abs": float(x2p_abs - x1p_abs),
                    "box_height_abs": float(y2p_abs - y1p_abs),
                }
            )

    samples = [
        fo.Sample(
            filepath=fp,
            ground_truth=fo.Detections(
                detections=[fo.Detection(**gt_d) for gt_d in data["gts"]]
            ),
            predictions=fo.Detections(
                detections=[fo.Detection(**pred_d) for pred_d in data["preds"]]
            ),
        )
        for fp, data in img_data.items()
        if data["gts"]  # Only add samples that have GT for this class
    ]

    if not samples:
        logger.warning(f"Нет данных для '{dataset_name}'. Удаление пустого датасета.")
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
        return None  # Return None if dataset is empty
    dataset.add_samples(samples)
    dataset.save()  # Save after adding samples
    logger.info(f"Добавлено {len(samples)} сэмплов в '{dataset_name}'.")

    if our_class_name_from_csv in config_params["SKIP_CLASSES_FOR_IOU_EVAL"]:
        evaluate_by_inclusion(
            dataset,
            gt_covered_threshold=config_params["INCLUSION_THRESHOLD_GT_COVERED"],
        )
    else:
        if dataset.count("ground_truth.detections") > 0:
            for iou_thr, iou_tag in config_params["IOU_DICT"].items():
                try:
                    logger.info(
                        f"Оценка IoU={iou_thr} для '{our_class_name_from_csv}' (метка: {display_gt_label})"
                    )
                    dataset.evaluate_detections(
                        "predictions",
                        gt_field="ground_truth",
                        eval_key=iou_tag,
                        method="coco",
                        iou=iou_thr,
                        compute_mAP=False,  # Usually True for full COCO, but can be False for per-class
                        classwise=False,  # Set True if you want separate TP/FP/FN per class (useful if multiple GT classes)
                        classes=[
                            display_gt_label
                        ],  # Evaluate only for the current class
                        progress=True,
                    )
                    dataset.save()  # Save evaluation results
                except Exception as e:
                    logger.error(
                        f"Ошибка оценки IoU={iou_thr} для '{our_class_name_from_csv}': {e}"
                    )
        else:
            logger.warning(
                f"Пропуск оценки IoU для '{our_class_name_from_csv}': нет GT."
            )

    if config_params["COMPUTE_GT_PATCH_EMBEDDINGS"]:
        logger.info(f"Вычисление эмбеддингов GT патчей для '{dataset_name}'...")
        compute_and_save_patch_embeddings(
            dataset,
            zoo_model_name=config_params["ZOO_MODEL_NAME_FOR_EMBEDDINGS"],
            patches_field="ground_truth",
            embeddings_field_on_detection=config_params[
                "PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION"
            ],
        )

    if config_params["COMPUTE_IMAGE_EMBEDDINGS"]:
        logger.info(f"Вычисление эмбеддингов изображений для '{dataset_name}'...")
        compute_and_save_image_embeddings(
            dataset,
            zoo_model_name=config_params["ZOO_MODEL_NAME_FOR_EMBEDDINGS"],
            embeddings_field_on_sample=config_params[
                "IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE"
            ],
        )

    logger.info(f"Обработка '{dataset_name}' завершена.")
    if config_params["LAUNCH_APP_FOR_EACH"] and samples:
        try:
            session = (
                fo.launch_app(  # Store session to potentially close later if needed
                    dataset,
                    address="0.0.0.0",
                    port=config_params["current_port"],
                    auto=False,  # Keeps Python script running
                )
            )
            logger.info(
                f"App для '{dataset_name}' доступен на порту {config_params['current_port']}. Сессия: {session}"
            )
        except Exception as e:
            logger.error(f"Не удалось запустить App для '{dataset_name}': {e}")
    return dataset


# --- ОСНОВНАЯ ФУНКЦИЯ ---
def main():
    APP_CONFIG = {
        "LAUNCH_APP_FOR_EACH": False,
        "COMPUTE_GT_PATCH_EMBEDDINGS": True,
        "COMPUTE_IMAGE_EMBEDDINGS": True,  # New flag for image embeddings
        "ZOO_MODEL_NAME_FOR_EMBEDDINGS": "dinov2-vitb14-torch",
        # "ZOO_MODEL_NAME_FOR_EMBEDDINGS": "clip-vit-base32-torch",
        "PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION": "gt_patch_embedding",  # Renamed for clarity
        "IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE": "image_embedding",  # New field name
        "START_PORT": 30082,
        "TEST_MODE_SINGLE_CSV": False,  # Set to True to process only the first CSV
    }

    LOADER_PARAMS = {
        "CVAT_LINK": CVAT_LINK,
        "SIZE_REQUIREMENTS": SIZE_REQUIREMENTS,
        "SKIP_CLASSES_FOR_IOU_EVAL": SKIP_CLASSES_FOR_IOU_EVAL,
        "INCLUSION_THRESHOLD_GT_COVERED": INCLUSION_THRESHOLD_GT_COVERED,
        "IOU_DICT": IOU_DICT,
        "COMPUTE_GT_PATCH_EMBEDDINGS": APP_CONFIG["COMPUTE_GT_PATCH_EMBEDDINGS"],
        "COMPUTE_IMAGE_EMBEDDINGS": APP_CONFIG[
            "COMPUTE_IMAGE_EMBEDDINGS"
        ],  # Pass new flag
        "ZOO_MODEL_NAME_FOR_EMBEDDINGS": APP_CONFIG["ZOO_MODEL_NAME_FOR_EMBEDDINGS"],
        "PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION": APP_CONFIG[
            "PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION"
        ],
        "IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE": APP_CONFIG[  # Pass new field name
            "IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE"
        ],
        "LAUNCH_APP_FOR_EACH": APP_CONFIG["LAUNCH_APP_FOR_EACH"],
    }

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            all_preds = json.load(f)
    except FileNotFoundError:
        logger.critical(f"Файл предсказаний не найден: {PATH_TO_PREDICTIONS}")
        return
    except json.JSONDecodeError:
        logger.critical(f"Ошибка декодирования JSON: {PATH_TO_PREDICTIONS}")
        return

    csv_files = sorted(glob.glob(os.path.join(PATH_TO_SPLIT, "*.csv")))
    if not csv_files:
        logger.warning(f"CSV файлы не найдены в {PATH_TO_SPLIT}")
        return

    if APP_CONFIG["TEST_MODE_SINGLE_CSV"]:
        logger.info("--- РЕЖИМ ТЕСТИРОВАНИЯ: Обработка только первого CSV файла ---")
        csv_files = csv_files[:1]
        if not csv_files:
            logger.warning("Нет CSV файлов для тестового режима.")
            return

    loaded_datasets_summary = []
    for i, csv_f in enumerate(csv_files):
        LOADER_PARAMS["current_port"] = APP_CONFIG["START_PORT"] + i
        ds = load_class_dataset_from_csv(csv_f, all_preds, LOADER_PARAMS)
        if ds:  # ds can be None if no samples were added
            info = {"name": ds.name, "patch_emb": False, "image_emb": False}

            if APP_CONFIG["COMPUTE_GT_PATCH_EMBEDDINGS"]:
                # Check if the field exists on detections (more robust)
                # Need to ensure there is at least one detection to check
                if (
                    ds.count(
                        f"ground_truth.detections.{APP_CONFIG['PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION']}"
                    )
                    > 0
                ):
                    info["patch_emb"] = True
                # Fallback: check first sample's first detection (less robust but faster if above fails for some reason)
                elif (
                    ds.first()
                    and ds.first().ground_truth
                    and ds.first().ground_truth.detections
                    and APP_CONFIG["PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION"]
                    in ds.first().ground_truth.detections[0]
                ):
                    info["patch_emb"] = True

            if APP_CONFIG["COMPUTE_IMAGE_EMBEDDINGS"]:
                # Check if the sample field exists (more robust)
                if ds.has_sample_field(
                    APP_CONFIG["IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE"]
                ):
                    info["image_emb"] = True

            if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and ds.count() > 0:
                info["app_port"] = LOADER_PARAMS["current_port"]
            loaded_datasets_summary.append(info)

    logger.info("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_summary:
        logger.info("Созданы/обновлены датасеты:")
        for info in loaded_datasets_summary:
            emb_msgs = []
            if info["patch_emb"]:
                emb_msgs.append("с эмб. GT патчей")
            if info["image_emb"]:
                emb_msgs.append("с эмб. изображений")

            emb_str = ""
            if emb_msgs:
                emb_str = f" ({', '.join(emb_msgs)})"

            app_str = (
                f" (App на порту {info['app_port']})" if "app_port" in info else ""
            )
            logger.info(f"- {info['name']}{emb_str}{app_str}")

        if loaded_datasets_summary:
            first_ds_name = loaded_datasets_summary[0]["name"]
            patch_emb_field_example = APP_CONFIG[
                "PATCH_EMBEDDINGS_FIELD_NAME_IN_DETECTION"
            ]
            image_emb_field_example = APP_CONFIG[
                "IMAGE_EMBEDDINGS_FIELD_NAME_ON_SAMPLE"
            ]

            logger.info(f"\nДля просмотра (если App не был запущен):")
            logger.info(f"import fiftyone as fo")
            logger.info(f"dataset = fo.load_dataset('{first_ds_name}')")
            logger.info(
                f"session = fo.launch_app(dataset, port={APP_CONFIG['START_PORT']})"
            )
            logger.info(f"--- Для ЭМБЕДДИНГОВ ПАТЧЕЙ (GT): ---")
            logger.info(
                f"В панели Embeddings -> Plot Type: Labels, Label field: ground_truth.detections, Field with embeddings: {patch_emb_field_example}"
            )
            logger.info(f"--- Для ЭМБЕДДИНГОВ ИЗОБРАЖЕНИЙ: ---")
            logger.info(
                f"В панели Embeddings -> Plot Type: Samples, Field with embeddings: {image_emb_field_example}"
            )
    else:
        logger.warning("Не было создано ни одного датасета.")


if __name__ == "__main__":
    main()
